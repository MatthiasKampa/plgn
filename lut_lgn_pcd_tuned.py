"""
Tuned persistent CD for the deep LGN (chip-native, no off-chip backprop).

Diagnosis of the first PCD attempt (stalled at chance): LR too high (model lurched
faster than the persistent chain could track), weak weight decay (model sharpened ->
negative chain froze -> biased gradient), centering destabilized, and only the negative
chain was persisted.

Fixes here:
  - KEEP THE MODEL SOFT while training (strong weight decay) so both chains keep mixing;
    sharpen only at inference (eval rescales the couplings). A LUT only needs the right
    SIGN of the conditional, not huge weights.
  - LOW learning rate so the persistent fantasy particles track the slowly-moving model.
  - PERSIST BOTH chains (positive data-clamped & negative model), with a burn-in first.
  - short k-step updates on the persistent chains (proper PCD), no centering.
  - best-checkpoint (FPGA keeps the best weights).
Network: (HA,HB,HC) = (8,8,4) -> 29 nodes, interior-gate degree 12 (fits Z-1 <=12).
Task: f(x0..x5) = MAJ3(x0,x1,x2) XOR MAJ3(x3,x4,x5).
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
HA = HB = 8; HC = 4; BETA = 1.0; P = 64

I_IN = list(range(6)); I_OA, I_OB, I_OC = 6, 7, 8
I_HA = list(range(9, 9 + HA)); I_HB = list(range(9 + HA, 9 + HA + HB)); I_HC = list(range(9 + HA + HB, 9 + HA + HB + HC))
N_NODES = 9 + HA + HB + HC; ALL_HID = I_HA + I_HB + I_HC
edge_idx = []
for h in I_HA:
    for u in [0, 1, 2, I_OA]: edge_idx.append((u, h))
for h in I_HB:
    for u in [3, 4, 5, I_OB]: edge_idx.append((u, h))
for h in I_HC:
    for u in [I_OA, I_OB, I_OC]: edge_idx.append((u, h))
EI = jnp.array([a for a, b in edge_idx]); EJ = jnp.array([b for a, b in edge_idx]); N_EDGES = len(edge_idx)
deg = [0] * N_NODES
for a, b in edge_idx: deg[a] += 1; deg[b] += 1
print(f"network: {N_NODES} nodes, {N_EDGES} couplings, interior-gate degree {deg[I_OA]} (<=12: {'OK' if max(deg)<=12 else 'NO'})")

NODES = [SpinNode() for _ in range(N_NODES)]
EDGES = [(NODES[a], NODES[b]) for a, b in edge_idx]
key = jax.random.key(0); key, kb, kw = jax.random.split(key, 3)
biases = jnp.concatenate([jnp.zeros(9), 0.1 * jax.random.normal(kb, (HA + HB + HC,))])
weights = 0.2 * jax.random.normal(kw, (N_EDGES,))
BASE = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))
def rebuild(b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), BASE, (b, w))

bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1)
X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
Y = (maj(bits[:, 0], bits[:, 1], bits[:, 2]) ^ maj(bits[:, 3], bits[:, 4], bits[:, 5]))
T = 2.0 * Y.astype(jnp.float32) - 1.0
print(f"task balance: {int(Y.sum())}/{P} ones")

def blk(idxs): return Block([NODES[i] for i in idxs])
NEG_CLAMP = [blk(I_IN)]; NEG_FREE = [blk([I_OA, I_OB, I_OC]), blk(ALL_HID)]; NEG_OBS = blk([I_OA, I_OB, I_OC] + ALL_HID); NEG_OBS_IDX = [I_OA, I_OB, I_OC] + ALL_HID
POS_CLAMP = [blk(I_IN + [I_OC])]; POS_FREE = [blk([I_OA, I_OB]), blk(ALL_HID)]; POS_OBS = blk([I_OA, I_OB] + ALL_HID); POS_OBS_IDX = [I_OA, I_OB] + ALL_HID
POS_CB = (jnp.concatenate([X, T[:, None]], 1) > 0); NEG_CB = (X > 0)
POS_CLAMP_VALS = jnp.concatenate([X, T[:, None]], 1)

SCHED_STEP = SamplingSchedule(n_warmup=0, n_samples=40, steps_per_sample=2)    # persistent: short k-step update (more samples -> lower-noise gradient)
SCHED_BURN = SamplingSchedule(n_warmup=300, n_samples=1, steps_per_sample=1)   # initial burn-in
SCHED_EVAL = SamplingSchedule(n_warmup=200, n_samples=300, steps_per_sample=2)

@eqx.filter_jit
def sample_phase(program, sched, keys, init, clamp, obs):
    def one(k, i, c): return sample_states(k, program, sched, i, [c], [obs])[0]
    return jax.vmap(one)(keys, init, clamp)

def assemble(sampled, obs_idx, clamp_idx, clamp_vals):
    S = sampled.shape[1]; full = jnp.zeros((P, S, N_NODES))
    full = full.at[:, :, jnp.array(clamp_idx)].set(jnp.broadcast_to(clamp_vals[:, None, :], (P, S, len(clamp_idx))))
    full = full.at[:, :, jnp.array(obs_idx)].set(2.0 * sampled.astype(jnp.float32) - 1.0)
    return full
def moments(full): return full.mean(1).mean(0), (full[:, :, EI] * full[:, :, EJ]).mean(1).mean(0)
def persist(sampled, n_unit): last = sampled[:, -1, :]; return [last[:, :n_unit], last[:, n_unit:]]

def eval_acc(b, w, key, sharpen=3.0):
    ebm = rebuild(b * sharpen, w * sharpen); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    s = sample_phase(prog, SCHED_EVAL, jax.random.split(key, P), hinton_init(key, ebm, NEG_FREE, (P,)), NEG_CB, NEG_OBS)
    oc = s[:, :, 2].astype(jnp.float32).mean(1)
    return float(jnp.mean(((oc > 0.5).astype(jnp.int32) == Y).astype(jnp.float32)))

# ============== tuned persistent CD ==============
print("\ntuned PCD (persist both chains, soft model via weight decay, low LR):")
STEPS, WD = 600, 2e-3   # WD that gave plain CD its best (preserves the +/- contrast)
b, w = biases, weights
mb = jnp.zeros(N_NODES); vb = jnp.zeros(N_NODES); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
# init + burn-in the persistent chains
pos_state = hinton_init(key, BASE, POS_FREE, (P,)); neg_state = hinton_init(key, BASE, NEG_FREE, (P,))
ebm = rebuild(b, w)
pos_state = persist(sample_phase(IsingSamplingProgram(ebm, POS_FREE, POS_CLAMP), SCHED_BURN, jax.random.split(key, P), pos_state, POS_CB, POS_OBS), 2)
neg_state = persist(sample_phase(IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP), SCHED_BURN, jax.random.split(key, P), neg_state, NEG_CB, NEG_OBS), 3)
print(f"{'step':>5}{'acc':>8}{'best':>7}{'|w|':>8}")
best_acc, best = -1.0, (b, w); traj = []
t0 = time.time()
for step in range(1, STEPS + 1):
    key, kp, kn = jax.random.split(key, 3)
    ebm = rebuild(b, w)
    sp = sample_phase(IsingSamplingProgram(ebm, POS_FREE, POS_CLAMP), SCHED_STEP, jax.random.split(kp, P), pos_state, POS_CB, POS_OBS)
    pos_state = persist(sp, 2); sb_p, ee_p = moments(assemble(sp, POS_OBS_IDX, I_IN + [I_OC], POS_CLAMP_VALS))
    sn = sample_phase(IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP), SCHED_STEP, jax.random.split(kn, P), neg_state, NEG_CB, NEG_OBS)
    neg_state = persist(sn, 3); sb_n, ee_n = moments(assemble(sn, NEG_OBS_IDX, I_IN, X))
    gb = -BETA * (sb_p - sb_n) + WD * b
    gw = -BETA * (ee_p - ee_n) + WD * w
    lr = 0.005 + 0.015 * (1.0 - step / STEPS)   # moderate: below CD's 0.05 (chain lags), above the too-low 0.012
    b1, b2, eps, t = 0.9, 0.999, 1e-8, step
    mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
    mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
    b = b - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
    w = w - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
    if step % 25 == 0 or step == 1:
        key, ke = jax.random.split(key); acc = eval_acc(b, w, ke)
        if acc > best_acc: best_acc, best = acc, (b, w)
        traj.append(int(round(acc * P)))
        print(f"{step:>5}{int(round(acc*P)):>5}/{P}{int(round(best_acc*P)):>5}/{P}{float(jnp.linalg.norm(w)):>8.2f}", flush=True)
print(f"trained in {time.time()-t0:.0f}s ; BEST = {int(round(best_acc*P))}/{P}")

b, w = best
key, ke = jax.random.split(key)
print(f"\ntrajectory(/64): {traj}")
print(f"TUNED-PCD best-checkpoint accuracy: {int(round(eval_acc(b, w, ke, sharpen=3.0)*P))}/{P}  "
      f"(vs untuned PCD 33/64, plain CD ~44/64)")
