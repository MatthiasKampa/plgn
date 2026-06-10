"""
2-layer Logic Gate Network as ONE bipartite deep EBM, trained end-to-end by
THRML whole-network contrastive divergence (the chip-native path).

Task:  f(x0..x5) = MAJ3(x0,x1,x2) XOR MAJ3(x3,x4,x5)
  - 6 inputs => a single LUT4 (fan-in 4) CANNOT do it; needs composition.
  - layer 1: gate A (x0,x1,x2 -> latent oA),  gate B (x3,x4,x5 -> latent oB)
  - layer 2: gate C (oA,oB -> output oC)
  - oA, oB are LATENT (no targets) -> this is the credit-assignment test:
    can the interior gates learn useful sub-functions from output supervision alone?

Whole-network CD:  dW = <s_i s_j>_+ - <s_i s_j>_- ,  both phases sampled on THRML.
  positive: clamp inputs + oC=f(x); sample oA,oB + all hidden
  negative: clamp inputs only;      sample oA,oB,oC + all hidden
The graph is bipartite (units {x,o} vs hidden), so it runs as the chip's 2-color cycle.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
HA = HB = 4
HC = 3
BETA = 1.0

# ---- node layout ----
N_INP = 6
I_IN = list(range(6))                     # 0..5 inputs
I_OA, I_OB, I_OC = 6, 7, 8
I_HA = list(range(9, 9 + HA))
I_HB = list(range(9 + HA, 9 + HA + HB))
I_HC = list(range(9 + HA + HB, 9 + HA + HB + HC))
N_NODES = 9 + HA + HB + HC
ALL_HID = I_HA + I_HB + I_HC

# ---- edges (unit<->hidden only => bipartite) ----
edge_idx = []
for h in I_HA:                            # gate A: x0,x1,x2 and oA  <-> hidA
    for u in [0, 1, 2, I_OA]: edge_idx.append((u, h))
for h in I_HB:                            # gate B: x3,x4,x5 and oB  <-> hidB
    for u in [3, 4, 5, I_OB]: edge_idx.append((u, h))
for h in I_HC:                            # gate C: oA,oB and oC      <-> hidC
    for u in [I_OA, I_OB, I_OC]: edge_idx.append((u, h))
EI = jnp.array([a for a, b in edge_idx]); EJ = jnp.array([b for a, b in edge_idx])
N_EDGES = len(edge_idx)

# degree report (connectivity neglected per request, but we still print it)
deg = [0] * N_NODES
for a, b in edge_idx: deg[a] += 1; deg[b] += 1
print(f"network: {N_NODES} nodes, {N_EDGES} couplings, max degree {max(deg)}  "
      f"(oA/oB degree {deg[I_OA]} = {HA}+{HC})")

# ---- THRML nodes / base EBM ----
NODES = [SpinNode() for _ in range(N_NODES)]
EDGES = [(NODES[a], NODES[b]) for a, b in edge_idx]
key = jax.random.key(0)
key, kb, kw = jax.random.split(key, 3)
biases = jnp.concatenate([jnp.zeros(9), 0.1 * jax.random.normal(kb, (HA + HB + HC,))])
weights = 0.3 * jax.random.normal(kw, (N_EDGES,))
BASE = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))
def rebuild(b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), BASE, (b, w))

# ---- data: all 2^6 patterns ----
P = 64
bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1)            # (64,6) {0,1}
X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
fA = maj(bits[:, 0], bits[:, 1], bits[:, 2])
fB = maj(bits[:, 3], bits[:, 4], bits[:, 5])
Y = (fA ^ fB)                                                    # target (64,)
T = 2.0 * Y.astype(jnp.float32) - 1.0
print(f"task balance: {int(Y.sum())}/{P} ones")

# ---- block structure for the two phases ----
def blk(idxs): return Block([NODES[i] for i in idxs])
# negative: clamp inputs; free units {oA,oB,oC} | hidden
NEG_CLAMP = [blk(I_IN)]
NEG_FREE = [blk([I_OA, I_OB, I_OC]), blk(ALL_HID)]
NEG_OBS_IDX = [I_OA, I_OB, I_OC] + ALL_HID
# positive: clamp inputs + oC; free units {oA,oB} | hidden
POS_CLAMP = [blk(I_IN + [I_OC])]
POS_FREE = [blk([I_OA, I_OB]), blk(ALL_HID)]
POS_OBS_IDX = [I_OA, I_OB] + ALL_HID
NEG_OBS, POS_OBS = blk(NEG_OBS_IDX), blk(POS_OBS_IDX)
SCHED = SamplingSchedule(n_warmup=120, n_samples=160, steps_per_sample=2)

@eqx.filter_jit
def sample_phase(program, keys, init, clamp, obs):
    def one(k, i, c): return sample_states(k, program, SCHED, i, [c], [obs])[0]
    return jax.vmap(one)(keys, init, clamp)

def assemble(sampled, obs_idx, clamp_idx, clamp_vals):
    # sampled: (P,S,nobs) bool ; build full +/-1 spins (P,S,N)
    S = sampled.shape[1]
    full = jnp.zeros((P, S, N_NODES))
    cv = jnp.broadcast_to(clamp_vals[:, None, :], (P, S, len(clamp_idx)))
    full = full.at[:, :, jnp.array(clamp_idx)].set(cv)
    full = full.at[:, :, jnp.array(obs_idx)].set(2.0 * sampled.astype(jnp.float32) - 1.0)
    return full

def moments(full):
    sbar = full.mean(axis=1).mean(axis=0)                         # (N,)
    eebar = (full[:, :, EI] * full[:, :, EJ]).mean(axis=1).mean(axis=0)  # (n_edges,)
    return sbar, eebar

# clamp value arrays (+/-1)
XIN = X                                                          # (64,6)
POS_CLAMP_VALS = jnp.concatenate([X, T[:, None]], axis=1)        # (64,7) inputs + oC
NEG_CLAMP_VALS = X                                               # (64,6)
# bool clamps for THRML (True=+1)
POS_CLAMP_BOOL = (POS_CLAMP_VALS > 0)
NEG_CLAMP_BOOL = (NEG_CLAMP_VALS > 0)

def phase_moments(b, w, key, which):
    ebm = rebuild(b, w)
    if which == "pos":
        free, obs, obs_idx, cl_idx, cl_bool, cl_val = POS_FREE, POS_OBS, POS_OBS_IDX, I_IN + [I_OC], POS_CLAMP_BOOL, POS_CLAMP_VALS
        prog = IsingSamplingProgram(ebm, POS_FREE, POS_CLAMP)
    else:
        free, obs, obs_idx, cl_idx, cl_bool, cl_val = NEG_FREE, NEG_OBS, NEG_OBS_IDX, I_IN, NEG_CLAMP_BOOL, NEG_CLAMP_VALS
        prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    keys = jax.random.split(key, P)
    init = hinton_init(key, ebm, free, (P,))
    sampled = sample_phase(prog, keys, init, cl_bool, obs)        # (P,S,nobs)
    full = assemble(sampled, obs_idx, cl_idx, cl_val)
    return moments(full)

# ---- exact functional accuracy monitor: clamp inputs, settle, read oC by majority ----
def eval_acc(b, w, key, sharpen=1.0):
    ebm = rebuild(b * sharpen, w * sharpen)
    prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
    keys = jax.random.split(key, P)
    init = hinton_init(key, ebm, NEG_FREE, (P,))
    sampled = sample_phase(prog, keys, init, NEG_CLAMP_BOOL, NEG_OBS)   # (P,S,nobs); col0 = oC? order [oA,oB,oC]+hid
    oc = sampled[:, :, 2].astype(jnp.float32).mean(axis=1)        # P(oC=1) per pattern (oC is 3rd in NEG_OBS)
    pred = (oc > 0.5).astype(jnp.int32)
    return float(jnp.mean((pred == Y).astype(jnp.float32)))

# ============== train: whole-network CD + Adam ==============
print("\nwhole-network CD training (both phases sampled on THRML):")
print(f"{'step':>5}{'acc':>8}{'|gw|':>9}")
STEPS, WD = 220, 2e-3
mb = jnp.zeros(N_NODES); vb = jnp.zeros(N_NODES); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
t0 = time.time()
for step in range(1, STEPS + 1):
    key, kpos, kneg = jax.random.split(key, 3)
    sb_p, ee_p = phase_moments(biases, weights, kpos, "pos")
    sb_n, ee_n = phase_moments(biases, weights, kneg, "neg")
    gb = -BETA * (sb_p - sb_n) + WD * biases
    gw = -BETA * (ee_p - ee_n) + WD * weights
    lr = 0.01 + 0.04 * (1.0 - step / STEPS)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, step
    mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
    mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
    biases = biases - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
    weights = weights - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
    if step % 20 == 0 or step == 1:
        key, ke = jax.random.split(key)
        acc = eval_acc(biases, weights, ke, sharpen=2.0)
        print(f"{step:>5}{int(round(acc*P)):>5}/{P}{float(jnp.linalg.norm(gw)):>9.4f}")
print(f"trained in {time.time()-t0:.1f}s")

# ============== did the LATENT interior gates learn MAJ3? ==============
print("\ninterpretability — what did latent oA learn? (clamp x0..x2, x3..x5=0, sample oA)")
ebm = rebuild(biases * 2.0, weights * 2.0)
prog = IsingSamplingProgram(ebm, NEG_FREE, NEG_CLAMP)
print(f"  x2x1x0 | MAJ3 | P(oA=1)")
agree = 0
for p3 in range(8):
    b3 = [(p3 >> k) & 1 for k in range(3)]
    xin = jnp.array(b3 + [0, 0, 0], dtype=bool)
    k1, k2 = jax.random.split(jax.random.fold_in(key, p3))
    init = hinton_init(k1, ebm, NEG_FREE, ())
    s = sample_states(k2, prog, SCHED, init, [xin], [Block([NODES[I_OA]])])[0]
    pa = float(jnp.mean(s.astype(jnp.float32)))
    m = int(sum(b3) >= 2)
    agree += (int(pa > 0.5) == m)
    print(f"   {b3[2]}{b3[1]}{b3[0]}   |   {m}  |  {pa:4.2f}")
print(f"  oA matches MAJ3 on {agree}/8 inputs "
      f"({'learned the sub-function' if agree>=7 else 'learned a different (still valid) code' if agree>=5 else 'did not align'})")
key, ke = jax.random.split(key)
print(f"\nFINAL whole-network accuracy: {int(round(eval_acc(biases, weights, ke, sharpen=2.5)*P))}/{P}"
      f"   ({N_NODES} nodes, trained end-to-end via sampling; interior outputs never supervised)")
