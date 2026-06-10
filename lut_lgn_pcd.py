"""
Chip-native deep-LGN training, stabilized: PERSISTENT contrastive divergence
(+ centering + best-checkpoint).  Same task / network as lut_lgn.py.

Why this is still 100% chip-native (FPGA-managed, no off-chip backprop):
  - gradient is the same free-vs-clamped correlation difference <ss>_+ - <ss>_- ,
    every term sampled on the chip;
  - PCD = the FPGA simply KEEPS the negative-phase chip state between weight updates
    instead of resetting it, so the model chain keeps mixing (-> approaches the exact
    ML gradient, kills the CD-bias that made plain CD peak-then-regress);
  - centering = subtract running per-node means (a local FPGA-side reparam) -> better
    conditioning of a deep EBM;
  - best-checkpoint = the FPGA keeps the best weights seen (we threw away a 94% model before).

Task: f(x0..x5) = MAJ3(x0,x1,x2) XOR MAJ3(x3,x4,x5)   (needs composition; oA,oB latent)
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
HA = HB = 4; HC = 3; BETA = 1.0

# ---- node layout (0-5 inputs, 6 oA, 7 oB, 8 oC, then hidden) ----
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
print(f"network: {N_NODES} nodes, {N_EDGES} couplings, max degree {max(deg)} (<=12 OK)")

NODES = [SpinNode() for _ in range(N_NODES)]
EDGES = [(NODES[a], NODES[b]) for a, b in edge_idx]
key = jax.random.key(0); key, kb, kw = jax.random.split(key, 3)
biases = jnp.concatenate([jnp.zeros(9), 0.1 * jax.random.normal(kb, (HA + HB + HC,))])
weights = 0.3 * jax.random.normal(kw, (N_EDGES,))
BASE = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))
def rebuild(b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), BASE, (b, w))

# ---- data ----
P = 64
bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1)
X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
Y = (maj(bits[:, 0], bits[:, 1], bits[:, 2]) ^ maj(bits[:, 3], bits[:, 4], bits[:, 5]))
T = 2.0 * Y.astype(jnp.float32) - 1.0
print(f"task balance: {int(Y.sum())}/{P} ones")

def blk(idxs): return Block([NODES[i] for i in idxs])
NEG_CLAMP = [blk(I_IN)]; NEG_FREE = [blk([I_OA, I_OB, I_OC]), blk(ALL_HID)]; NEG_OBS_IDX = [I_OA, I_OB, I_OC] + ALL_HID
POS_CLAMP = [blk(I_IN + [I_OC])]; POS_FREE = [blk([I_OA, I_OB]), blk(ALL_HID)]; POS_OBS_IDX = [I_OA, I_OB] + ALL_HID
NEG_OBS, POS_OBS = blk(NEG_OBS_IDX), blk(POS_OBS_IDX)
SCHED_POS = SamplingSchedule(n_warmup=80, n_samples=100, steps_per_sample=1)
SCHED_NEG = SamplingSchedule(n_warmup=5, n_samples=120, steps_per_sample=1)   # short: chain PERSISTS across steps
SCHED_EVAL = SamplingSchedule(n_warmup=200, n_samples=300, steps_per_sample=2)

@eqx.filter_jit
def sample_phase(program, sched, keys, init, clamp, obs):
    def one(k, i, c): return sample_states(k, program, sched, i, [c], [obs])[0]
    return jax.vmap(one)(keys, init, clamp)

def assemble(sampled, obs_idx, clamp_idx, clamp_vals):
    S = sampled.shape[1]
    full = jnp.zeros((P, S, N_NODES))
    full = full.at[:, :, jnp.array(clamp_idx)].set(jnp.broadcast_to(clamp_vals[:, None, :], (P, S, len(clamp_idx))))
    full = full.at[:, :, jnp.array(obs_idx)].set(2.0 * sampled.astype(jnp.float32) - 1.0)
    return full
def moments(full):
    return full.mean(1).mean(0), (full[:, :, EI] * full[:, :, EJ]).mean(1).mean(0)

POS_CLAMP_BOOL = (jnp.concatenate([X, T[:, None]], 1) > 0); NEG_CLAMP_BOOL = (X > 0)

def pos_phase(b, w, key):
    ebm = rebuild(b, w); prog = IsingSamplingProgram(ebm, POS_FREE, POS_CLAMP)
    sampled = sample_phase(prog, SCHED_POS, jax.random.split(key, P), hinton_init(key, ebm, POS_FREE, (P,)), POS_CLAMP_BOOL, POS_OBS)
    return moments(assemble(sampled, POS_OBS_IDX, I_IN + [I_OC], jnp.concatenate([X, T[:, None]], 1)))

def neg_phase(b, w, key, neg_state):                       # PCD: continue from persisted state
    ebm = rebuild(b, w); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    sampled = sample_phase(prog, SCHED_NEG, jax.random.split(key, P), neg_state, NEG_CLAMP_BOOL, NEG_OBS)
    last = sampled[:, -1, :]                               # persist final chip state
    new_state = [last[:, :3], last[:, 3:]]
    sb, ee = moments(assemble(sampled, NEG_OBS_IDX, I_IN, X))
    return sb, ee, new_state

def eval_acc(b, w, key, sharpen=2.0):
    ebm = rebuild(b * sharpen, w * sharpen); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    sampled = sample_phase(prog, SCHED_EVAL, jax.random.split(key, P), hinton_init(key, ebm, NEG_FREE, (P,)), NEG_CLAMP_BOOL, NEG_OBS)
    oc = sampled[:, :, 2].astype(jnp.float32).mean(1)      # P(oC=1); oC is 3rd in NEG_OBS
    return float(jnp.mean(((oc > 0.5).astype(jnp.int32) == Y).astype(jnp.float32)))

# ============== persistent-CD training ==============
print("\nPCD + centering + best-checkpoint (persistent negative chain on THRML):")
print(f"{'step':>5}{'acc':>8}{'best':>7}")
STEPS, WD = 320, 1e-3
mb = jnp.zeros(N_NODES); vb = jnp.zeros(N_NODES); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
mu = jnp.zeros(N_NODES)                                    # centering running means
neg_state = hinton_init(key, BASE, NEG_FREE, (P,))         # persistent chain init
best_acc, best = -1.0, (biases, weights)
t0 = time.time()
for step in range(1, STEPS + 1):
    key, kp, kn = jax.random.split(key, 3)
    sb_p, ee_p = pos_phase(biases, weights, kp)
    sb_n, ee_n, neg_state = neg_phase(biases, weights, kn, neg_state)
    raw_gb = sb_p - sb_n; raw_gw = ee_p - ee_n
    mu = 0.95 * mu + 0.05 * 0.5 * (sb_p + sb_n)            # EMA of node means
    gw_c = raw_gw - mu[EI] * raw_gb[EJ] - mu[EJ] * raw_gb[EI]   # centered weight gradient
    gb = -BETA * raw_gb + WD * biases
    gw = -BETA * gw_c + WD * weights
    lr = 0.01 + 0.03 * (1.0 - step / STEPS)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, step
    mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
    mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
    biases = biases - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
    weights = weights - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
    if step % 15 == 0 or step == 1:
        key, ke = jax.random.split(key)
        acc = eval_acc(biases, weights, ke)
        if acc > best_acc: best_acc, best = acc, (biases, weights)
        print(f"{step:>5}{int(round(acc*P)):>5}/{P}{int(round(best_acc*P)):>5}/{P}{'  <- new best' if acc>=best_acc else ''}")
print(f"trained in {time.time()-t0:.1f}s ; BEST checkpoint = {int(round(best_acc*P))}/{P}")

# ============== best checkpoint: final accuracy + did the latent gate learn MAJ3? ==============
biases, weights = best
key, ke = jax.random.split(key)
final = eval_acc(biases, weights, ke, sharpen=2.5)
print(f"\nBEST-checkpoint whole-network accuracy: {int(round(final*P))}/{P}")
print("\ninterpretability — latent oA vs MAJ3 (clamp x0..x2, x3..x5=0):")
ebm = rebuild(biases * 2.5, weights * 2.5); prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
print("  x2x1x0 | MAJ3 | P(oA=1)")
agree = 0
for p3 in range(8):
    b3 = [(p3 >> k) & 1 for k in range(3)]
    k1, k2 = jax.random.split(jax.random.fold_in(key, p3))
    s = sample_states(k2, prog, SCHED_EVAL, hinton_init(k1, ebm, NEG_FREE, ()), [jnp.array(b3 + [0, 0, 0], dtype=bool)], [Block([NODES[I_OA]])])[0]
    pa = float(jnp.mean(s.astype(jnp.float32))); m = int(sum(b3) >= 2); agree += (int(pa > 0.5) == m)
    print(f"   {b3[2]}{b3[1]}{b3[0]}   |   {m}  |  {pa:4.2f}")
verdict = "learned MAJ3" if agree >= 7 else ("learned an equivalent code (8-agree under relabeling)" if min(agree, 8 - agree) <= 1 else "partial / distributed code")
print(f"  oA agrees with MAJ3 on {agree}/8  -> {verdict}")
print(f"\nchip-native deep-LGN result: {int(round(final*P))}/{P}  (PCD, no off-chip backprop, {N_NODES} nodes)")
