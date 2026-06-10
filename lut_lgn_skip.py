"""
Skip-edge fix for the deep-LGN credit-assignment stall (the highest-leverage
recommendation from the multi-agent review).

Diagnosis: the label reaches the latent gate oA only via a 2-hop path oC->hidC->oA,
so the contrastive gradient on oA is ~0 (measured <oA>+ ~= <oA>-). FIX: add DIRECT
skip couplings oA-oC and oB-oC, making supervision 1-hop. In the mean-field positive
phase the clamped output oC now directly fields oA/oB.

Consequences handled:
  - oA-oC, oB-oC are unit-unit edges -> the graph is no longer 2-colorable. The negative
    phase uses a valid 3-colouring {oA,oB} | {oC} | hidden.
  - rebalance HC=3 so oA degree = 8(hidA)+3(hidC)+1(oC) = 12  (stays within Z-1 <=12).
  - thermodynamic hygiene (reviewer B): fixed beta=1.0 (near-critical, no backwards
    curriculum), eval beta = train beta, lr 3e-3->1e-3, Glorot init.
Recipe otherwise = MF-positive + PCD-negative (DBM), best-checkpoint. Chip-native.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
HA = HB = 8; HC = 3; BETA = 1.0; P = 64

I_IN = list(range(6)); I_OA, I_OB, I_OC = 6, 7, 8
I_HA = list(range(9, 9 + HA)); I_HB = list(range(9 + HA, 9 + HA + HB)); I_HC = list(range(9 + HA + HB, 9 + HA + HB + HC))
N = 9 + HA + HB + HC; ALL_HID = I_HA + I_HB + I_HC
edge_idx = []
for h in I_HA:
    for u in [0, 1, 2, I_OA]: edge_idx.append((u, h))
for h in I_HB:
    for u in [3, 4, 5, I_OB]: edge_idx.append((u, h))
for h in I_HC:
    for u in [I_OA, I_OB, I_OC]: edge_idx.append((u, h))
edge_idx.append((I_OA, I_OC)); edge_idx.append((I_OB, I_OC))     # <-- SKIP: 1-hop supervision to latents
EI = jnp.array([a for a, b in edge_idx]); EJ = jnp.array([b for a, b in edge_idx]); N_EDGES = len(edge_idx)
deg = [0] * N
for a, b in edge_idx: deg[a] += 1; deg[b] += 1
assert max(deg) <= 12, f"degree {max(deg)} > 12"
print(f"network: {N} nodes, {N_EDGES} couplings (incl 2 skip oA-oC,oB-oC), max degree {max(deg)} (<=12 OK)")

NODES = [SpinNode() for _ in range(N)]; EDGES = [(NODES[a], NODES[b]) for a, b in edge_idx]
key = jax.random.key(0); key, kw = jax.random.split(key)
GAIN = 1.0; deg_arr = jnp.array(deg, jnp.float32)
sigma_e = GAIN * jnp.sqrt(2.0 / (deg_arr[EI] + deg_arr[EJ]))
weights = sigma_e * jax.random.normal(kw, (N_EDGES,)); biases = jnp.zeros(N)
BASE = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))
def rebuild(b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), BASE, (b, w))

bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1); X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
Y = (maj(bits[:, 0], bits[:, 1], bits[:, 2]) ^ maj(bits[:, 3], bits[:, 4], bits[:, 5])); T = 2.0 * Y.astype(jnp.float32) - 1.0
print(f"task balance: {int(Y.sum())}/{P} ones")

def blk(idxs): return Block([NODES[i] for i in idxs])
NEG_CLAMP = [blk(I_IN)]; NEG_FREE = [blk([I_OA, I_OB]), blk([I_OC]), blk(ALL_HID)]   # 3-colour (skip edges)
NEG_OBS = blk([I_OA, I_OB, I_OC] + ALL_HID); NEG_OBS_IDX = [I_OA, I_OB, I_OC] + ALL_HID; NEG_CB = (X > 0)
free_pos = jnp.zeros(N, bool).at[jnp.array([I_OA, I_OB] + ALL_HID)].set(True)
clamp_vals_pos = jnp.zeros((P, N)).at[:, 0:6].set(X).at[:, I_OC].set(T)

@jax.jit
def mf_positive(b, w, n_iter=25):
    Wm = jnp.zeros((N, N)).at[EI, EJ].set(w).at[EJ, EI].set(w)
    mu0 = jnp.where(free_pos[None, :], 0.0, clamp_vals_pos)
    def body(mu, _):
        nm = jnp.tanh(BETA * (b[None, :] + mu @ Wm))
        return jnp.where(free_pos[None, :], nm, clamp_vals_pos), None
    mu, _ = jax.lax.scan(body, mu0, None, length=n_iter)
    return mu.mean(0), (mu[:, EI] * mu[:, EJ]).mean(0)

SCHED_STEP = SamplingSchedule(n_warmup=0, n_samples=50, steps_per_sample=2)
SCHED_BURN = SamplingSchedule(n_warmup=300, n_samples=1, steps_per_sample=1)
SCHED_EVAL = SamplingSchedule(n_warmup=200, n_samples=300, steps_per_sample=2)
@eqx.filter_jit
def sample_phase(program, sched, keys, init, clamp, obs):
    def one(k, i, c): return sample_states(k, program, sched, i, [c], [obs])[0]
    return jax.vmap(one)(keys, init, clamp)
def assemble(sampled, obs_idx, clamp_idx, clamp_vals):
    S = sampled.shape[1]; full = jnp.zeros((P, S, N))
    full = full.at[:, :, jnp.array(clamp_idx)].set(jnp.broadcast_to(clamp_vals[:, None, :], (P, S, len(clamp_idx))))
    full = full.at[:, :, jnp.array(obs_idx)].set(2.0 * sampled.astype(jnp.float32) - 1.0)
    return full
def neg_moments(full): return full.mean(1).mean(0), (full[:, :, EI] * full[:, :, EJ]).mean(1).mean(0)
def persist3(sampled): last = sampled[:, -1, :]; return [last[:, :2], last[:, 2:3], last[:, 3:]]  # [oA,oB]|[oC]|hidden
def eval_acc(b, w, key):
    ebm = rebuild(b, w); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    s = sample_phase(prog, SCHED_EVAL, jax.random.split(key, P), hinton_init(key, ebm, NEG_FREE, (P,)), NEG_CB, NEG_OBS)
    oc = s[:, :, 2].astype(jnp.float32).mean(1)
    return float(jnp.mean(((oc > 0.5).astype(jnp.int32) == Y).astype(jnp.float32)))

print("\nSKIP-EDGE recipe: oA-oC,oB-oC (1-hop) + Glorot init + fixed beta=1.0 + MF-positive + PCD-negative")
STEPS, WD = 500, 1e-3
b, w = biases, weights
mb = jnp.zeros(N); vb = jnp.zeros(N); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
neg_state = hinton_init(key, BASE, NEG_FREE, (P,))
neg_state = persist3(sample_phase(IsingSamplingProgram(rebuild(b, w), NEG_FREE, NEG_CLAMP), SCHED_BURN, jax.random.split(key, P), neg_state, NEG_CB, NEG_OBS))
print(f"{'step':>5}{'acc':>8}{'best':>7}{'|w|':>7}")
best_acc, best, traj = -1.0, (b, w), []
t0 = time.time()
for step in range(1, STEPS + 1):
    key, kn, ke = jax.random.split(key, 3)
    sb_p, ee_p = mf_positive(b, w)
    sn = sample_phase(IsingSamplingProgram(rebuild(b, w), NEG_FREE, NEG_CLAMP), SCHED_STEP, jax.random.split(kn, P), neg_state, NEG_CB, NEG_OBS)
    neg_state = persist3(sn); sb_n, ee_n = neg_moments(assemble(sn, NEG_OBS_IDX, I_IN, X))
    gb = -(sb_p - sb_n) + WD * b; gw = -(ee_p - ee_n) + WD * w
    lr = 0.001 + 0.002 * (1.0 - step / STEPS)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, step
    mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
    mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
    b = b - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
    w = w - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
    if step % 20 == 0 or step == 1:
        acc = eval_acc(b, w, ke)
        if acc > best_acc: best_acc, best = acc, (b, w)
        traj.append(int(round(acc * P)))
        print(f"{step:>5}{int(round(acc*P)):>5}/{P}{int(round(best_acc*P)):>5}/{P}{float(jnp.linalg.norm(w)):>7.2f}", flush=True)
print(f"trained in {time.time()-t0:.0f}s ; BEST = {int(round(best_acc*P))}/{P}")
b, w = best; key, ke = jax.random.split(key)
print(f"\ntrajectory(/64): {traj}")
# did the now-1-hop-supervised latent oA learn MAJ3? (clamp x0..x2, x3..x5=0, sample oA)
ebm = rebuild(b, w); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
agree = 0
for p3 in range(8):
    b3 = [(p3 >> k) & 1 for k in range(3)]
    k1, k2 = jax.random.split(jax.random.fold_in(key, p3))
    s = sample_states(k2, prog, SCHED_EVAL, hinton_init(k1, ebm, NEG_FREE, ()), [jnp.array(b3 + [0, 0, 0], dtype=bool)], [Block([NODES[I_OA]])])[0]
    agree += (int(float(jnp.mean(s.astype(jnp.float32))) > 0.5) == int(sum(b3) >= 2))
print(f"SKIP-EDGE best-checkpoint: {int(round(eval_acc(b, w, ke)*P))}/{P}   "
      f"(prior latent attempts ~33-44/64);  latent oA matches MAJ3 on {agree}/8")
