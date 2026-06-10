"""
Winning-recipe attempt: the Deep Boltzmann Machine training recipe
(Salakhutdinov & Hinton 2009) adapted to chip-native training.

Diagnosis: all sampling-based attempts (CD, PCD-untuned, PCD-tuned at 3 WD/LR regimes)
stalled near chance. They share one weakness: the POSITIVE (data-clamped) statistics are
estimated by short Gibbs sampling THROUGH the latent units oA,oB -> too noisy/biased ->
the data gradient is useless.

Fix (the DBM recipe):
  - POSITIVE phase  = MEAN-FIELD fixed point (deterministic, low-variance) for the latent
    units given clamped inputs+output.  < s_i s_j>_+ ~= mu_i mu_j.
  - NEGATIVE phase  = PCD sampling on the chip (persistent fantasy particles).
  - keep the model soft enough (weight decay) that the negative chain keeps mixing;
    sharpen only at inference.
Mapping to hardware: the FPGA runs the cheap mean-field inference for the data statistics;
the Z-1 chip runs the negative-phase sampling. Still no off-chip backprop.

Network (8,8,4) -> 29 nodes, interior-gate degree 12 (fits Z-1 <=12).
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
N = 9 + HA + HB + HC; ALL_HID = I_HA + I_HB + I_HC
edge_idx = []
for h in I_HA:
    for u in [0, 1, 2, I_OA]: edge_idx.append((u, h))
for h in I_HB:
    for u in [3, 4, 5, I_OB]: edge_idx.append((u, h))
for h in I_HC:
    for u in [I_OA, I_OB, I_OC]: edge_idx.append((u, h))
EI = jnp.array([a for a, b in edge_idx]); EJ = jnp.array([b for a, b in edge_idx]); N_EDGES = len(edge_idx)
deg = [0] * N
for a, b in edge_idx: deg[a] += 1; deg[b] += 1
print(f"network: {N} nodes, {N_EDGES} couplings, interior-gate degree {deg[I_OA]} (<=12: {'OK' if max(deg)<=12 else 'NO'})")

NODES = [SpinNode() for _ in range(N)]; EDGES = [(NODES[a], NODES[b]) for a, b in edge_idx]
key = jax.random.key(0); key, kb, kw = jax.random.split(key, 3)
# Glorot/fan-in init: sigma_e = GAIN*sqrt(2/(deg_i+deg_j)) keeps every node's field O(1)
# (responsive, unsaturated) so the latent units oA,oB are DRIVEN by their inputs from step 0.
GAIN = 1.0
deg_arr = jnp.array(deg, dtype=jnp.float32)
sigma_e = GAIN * jnp.sqrt(2.0 / (deg_arr[EI] + deg_arr[EJ]))
weights = sigma_e * jax.random.normal(kw, (N_EDGES,))
biases = jnp.zeros(N)   # zero bias => centered +/-1 units (P(s=1)=0.5)
BASE = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))
def rebuild(b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), BASE, (b, w))

bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1); X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
Y = (maj(bits[:, 0], bits[:, 1], bits[:, 2]) ^ maj(bits[:, 3], bits[:, 4], bits[:, 5])); T = 2.0 * Y.astype(jnp.float32) - 1.0
print(f"task balance: {int(Y.sum())}/{P} ones")

def blk(idxs): return Block([NODES[i] for i in idxs])
NEG_CLAMP = [blk(I_IN)]; NEG_FREE = [blk([I_OA, I_OB, I_OC]), blk(ALL_HID)]; NEG_OBS = blk([I_OA, I_OB, I_OC] + ALL_HID); NEG_OBS_IDX = [I_OA, I_OB, I_OC] + ALL_HID
NEG_CB = (X > 0)

# ---- POSITIVE phase: mean-field (clamp inputs + oC; infer oA,oB + hidden) ----
free_pos = jnp.zeros(N, bool).at[jnp.array([I_OA, I_OB] + ALL_HID)].set(True)   # free mask
clamp_vals_pos = jnp.zeros((P, N)).at[:, 0:6].set(X).at[:, I_OC].set(T)          # clamped values
@jax.jit
def mf_positive(b, w, n_iter=25):
    Wm = jnp.zeros((N, N)).at[EI, EJ].set(w).at[EJ, EI].set(w)
    mu = jnp.where(free_pos[None, :], 0.0, clamp_vals_pos)
    def body(mu, _):
        nm = jnp.tanh(BETA * (b[None, :] + mu @ Wm))
        return jnp.where(free_pos[None, :], nm, clamp_vals_pos), None
    mu, _ = jax.lax.scan(body, mu, None, length=n_iter)
    return mu.mean(0), (mu[:, EI] * mu[:, EJ]).mean(0)

# ---- NEGATIVE phase: PCD sampling on THRML ----
SCHED_STEP = SamplingSchedule(n_warmup=0, n_samples=60, steps_per_sample=2)
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
def persist(sampled, n_unit): last = sampled[:, -1, :]; return [last[:, :n_unit], last[:, n_unit:]]
def eval_acc(b, w, key, sharpen=3.0):
    ebm = rebuild(b * sharpen, w * sharpen); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    s = sample_phase(prog, SCHED_EVAL, jax.random.split(key, P), hinton_init(key, ebm, NEG_FREE, (P,)), NEG_CB, NEG_OBS)
    oc = s[:, :, 2].astype(jnp.float32).mean(1)
    return float(jnp.mean(((oc > 0.5).astype(jnp.int32) == Y).astype(jnp.float32)))

# ============== train: MF positive + PCD negative ==============
print("\nMF-positive + PCD-negative (the DBM recipe), chip-native:")
STEPS, WD = 500, 1.5e-3
b, w = biases, weights
mb = jnp.zeros(N); vb = jnp.zeros(N); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
neg_state = hinton_init(key, BASE, NEG_FREE, (P,))
neg_state = persist(sample_phase(IsingSamplingProgram(rebuild(b, w), NEG_FREE, NEG_CLAMP), SCHED_BURN, jax.random.split(key, P), neg_state, NEG_CB, NEG_OBS), 3)
print(f"{'step':>5}{'acc':>8}{'best':>7}{'|w|':>7}")
best_acc, best, traj = -1.0, (b, w), []
t0 = time.time()
for step in range(1, STEPS + 1):
    key, kn, ke = jax.random.split(key, 3)
    sb_p, ee_p = mf_positive(b, w)
    sn = sample_phase(IsingSamplingProgram(rebuild(b, w), NEG_FREE, NEG_CLAMP), SCHED_STEP, jax.random.split(kn, P), neg_state, NEG_CB, NEG_OBS)
    neg_state = persist(sn, 3); sb_n, ee_n = neg_moments(assemble(sn, NEG_OBS_IDX, I_IN, X))
    gb = -BETA * (sb_p - sb_n) + WD * b; gw = -BETA * (ee_p - ee_n) + WD * w
    lr = 0.004 + 0.016 * (1.0 - step / STEPS)
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
print(f"MF+PCD best-checkpoint: {int(round(eval_acc(b, w, ke)*P))}/{P}   "
      f"(sampling-only attempts: CD ~44, PCD ~33-38, all near chance through the latent layer)")
