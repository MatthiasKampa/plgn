"""
Move to THRML: learn a LUT4 by SAMPLING-BASED training (contrastive divergence),
with the negative phase run on THRML's block-Gibbs sampler -- the hardware-native
path (what a Z-1 TSU would actually do).

Conditional CD (inputs clamped in BOTH phases -> trains P(output | inputs)):
  positive phase: clamp inputs + output to data; hidden mean is exact (RBM closed form)
  negative phase: clamp inputs only; sample output+hidden with THRML block-Gibbs
  gradient:  dW = <s_i s_j>_+  -  <s_i s_j>_- ,   db = <s_i>_+ - <s_i>_-
Then verify the learned circuit by clamped THRML sampling.

Target = PARITY (the worst case). H=6 hidden (the robust pick from the search).
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
N_IN, H = 4, 8
N_VIS = N_IN + 1
N_NODES = N_VIS + H
BETA = 1.0

IDX = jnp.arange(16)
XBITS = ((IDX[:, None] >> jnp.arange(N_IN)) & 1)                 # (16,4) {0,1}
X_SPIN = 2.0 * XBITS.astype(jnp.float32) - 1.0
PARITY = jnp.array([bin(i).count("1") & 1 for i in range(16)])
T_SPIN = 2.0 * PARITY.astype(jnp.float32) - 1.0                  # (16,) target output spin
CLAMP_IN = XBITS.astype(bool)                                    # (16,4) input bits as spins (True=+1)

def logcosh(x):
    a = jnp.abs(x); return a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)

# ---------- THRML graph (built once; params swapped per step via eqx.tree_at) ----------
ins = [SpinNode() for _ in range(N_IN)]
out = SpinNode()
hid = [SpinNode() for _ in range(H)]
NODES = ins + [out] + hid                                        # idx 0-3 in, 4 out, 5.. hidden
VIS = ins + [out]
EDGES = [(VIS[i], hid[j]) for i in range(N_VIS) for j in range(H)]   # weight k = i*H+j
FREE_BLOCKS = [Block([out]), Block(hid)]                         # 2-color free set
CLAMPED_BLOCKS = [Block(ins)]
FREE_ALL = [out] + hid
EI = jnp.array([i for i in range(N_VIS) for j in range(H)])      # full-node idx of edge ends
EJ = jnp.array([N_VIS + j for i in range(N_VIS) for j in range(H)])
N_EDGES = len(EDGES)

# --- enforce the Z-1 hardware budget: <=12 couplings per node (+1 self-bias) ---
_deg = [0] * N_NODES
for _a, _b in EDGES:
    _deg[NODES.index(_a)] += 1; _deg[NODES.index(_b)] += 1
MAXDEG = max(_deg)
assert MAXDEG <= 12, f"node degree {MAXDEG} exceeds the Z-1 budget of 12 couplings"
print(f"connectivity check: max node degree = {MAXDEG}  (<=12 couplings + 1 self-bias: fits Z-1)")
print(f"  per-node degrees -> inputs:{H}  output:{H}  hidden:{N_VIS}   (RBM caps H<=12 to stay in budget)")

key = jax.random.key(0)
key, kb, kw = jax.random.split(key, 3)
biases = jnp.concatenate([jnp.zeros(N_VIS), 0.1 * jax.random.normal(kb, (H,))])   # input/out biases start 0
weights = 0.3 * jax.random.normal(kw, (N_EDGES,))
BASE_EBM = IsingEBM(NODES, EDGES, biases, weights, jnp.array(BETA))

def make_program(b, w):
    ebm = eqx.tree_at(lambda e: (e.biases, e.weights), BASE_EBM, (b, w))
    return ebm, IsingSamplingProgram(ebm, FREE_BLOCKS, CLAMPED_BLOCKS)

SCHED_NEG = SamplingSchedule(n_warmup=250, n_samples=350, steps_per_sample=2)

# batched negative-phase sampler: vmap over the 16 input patterns.
# eqx.filter_jit partitions the program into traced arrays (weights/biases) vs static
# structure (blocks/nodes), so the THRML program can be passed through jit cleanly.
@eqx.filter_jit
def neg_sample(program, keys, init, clamp16):
    def one(k, i, c):
        return sample_states(k, program, SCHED_NEG, i, [c], [Block(FREE_ALL)])[0]
    return jax.vmap(one)(keys, init, clamp16)

# ---------- exact functional accuracy monitor (closed form, no sampling) ----------
@jax.jit
def exact_acc(b, w):
    b_out, c, Wvh = b[4], b[5:], w.reshape(N_VIS, H)
    def lg(xr):
        vp = jnp.concatenate([xr, jnp.array([1.0])]); vn = jnp.concatenate([xr, jnp.array([-1.0])])
        return (b_out + jnp.sum(logcosh(c + vp @ Wvh))) - (-b_out + jnp.sum(logcosh(c + vn @ Wvh)))
    L = jax.vmap(lg)(X_SPIN)
    return jnp.mean((jnp.sign(L) == T_SPIN).astype(jnp.float32))

# ---------- positive phase (exact RBM moments) ----------
@jax.jit
def pos_moments(b, w):
    c, Wvh = b[5:], w.reshape(N_VIS, H)
    V = jnp.concatenate([X_SPIN, T_SPIN[:, None]], axis=1)        # (16,5) clamped visibles
    hmean = jnp.tanh(BETA * (c[None, :] + V @ Wvh))              # (16,H) exact <h|v>
    full = jnp.concatenate([X_SPIN, T_SPIN[:, None], hmean], axis=1)   # (16,Nnodes)
    sbar = full.mean(0)
    eebar = (full[:, EI] * full[:, EJ]).mean(0)
    return sbar, eebar

def neg_moments(b, w, key):
    ebm, prog = make_program(b, w)
    keys = jax.random.split(key, 16)
    init = hinton_init(key, ebm, FREE_BLOCKS, (16,))             # list: [(16,1),(16,H)]
    states = neg_sample(prog, keys, init, CLAMP_IN)             # (16, S, 1+H) bool
    spin = 2.0 * states.astype(jnp.float32) - 1.0
    inb = jnp.broadcast_to(X_SPIN[:, None, :], (16, states.shape[1], N_IN))
    full = jnp.concatenate([inb, spin[:, :, :1], spin[:, :, 1:]], axis=2)   # (16,S,Nnodes)
    sbar = full.mean(axis=(0, 1))
    eebar = (full[:, :, EI] * full[:, :, EJ]).mean(axis=(0, 1))
    return sbar, eebar

# ---------- train: Adam on (biases, weights), gradient from CD moments ----------
print("=" * 66)
print(f"THRML sampling-based training (conditional CD) — PARITY, H={H}")
print(f"negative phase = THRML block-Gibbs ({SCHED_NEG.n_samples} samples x16 clamps/step)")
print("=" * 66)
STEPS, WD = 500, 2e-3   # weight decay keeps the model soft so the CD negative chain keeps mixing
mb = jnp.zeros(N_NODES); vb = jnp.zeros(N_NODES); mw = jnp.zeros(N_EDGES); vw = jnp.zeros(N_EDGES)
print(f"\n{'step':>5}{'exact acc':>12}{'|grad_w|':>12}")
t0 = time.time()
for step in range(1, STEPS + 1):
    key, sk = jax.random.split(key)
    sb_p, ee_p = pos_moments(biases, weights)
    sb_n, ee_n = neg_moments(biases, weights, sk)
    gb = -BETA * (sb_p - sb_n) + WD * biases   # loss-grad: -(<.>+ - <.>-) + weight decay
    gw = -BETA * (ee_p - ee_n) + WD * weights
    lr = 0.008 + 0.032 * (1.0 - step / STEPS)  # decay 0.04 -> 0.008
    b1, b2, eps, t = 0.9, 0.999, 1e-8, step
    mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
    mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
    biases = biases - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
    weights = weights - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
    if step % 40 == 0 or step == 1:
        acc = float(exact_acc(biases, weights))
        print(f"{step:>5}{int(round(acc*16)):>10}/16{float(jnp.linalg.norm(gw)):>12.4f}")
print(f"\ntrained in {time.time()-t0:.1f}s ; final exact functional accuracy = "
      f"{int(round(float(exact_acc(biases, weights))*16))}/16")

# ---------- verify on THRML block-Gibbs (clamped inference), like the hardware would run ----------
print("\n" + "-" * 66)
print("verifying the THRML-trained circuit by clamped block-Gibbs inference:")
# rescale to moderate sharpness so the inference chain mixes
b_out, c, Wvh = weights, None, weights.reshape(N_VIS, H)
Vp = jnp.concatenate([X_SPIN, jnp.ones((16, 1))], 1); Vn = jnp.concatenate([X_SPIN, -jnp.ones((16, 1))], 1)
def F(V):
    return -(V @ jnp.concatenate([biases[:4], biases[4:5]])) - jnp.sum(logcosh(biases[5:] + V @ weights.reshape(N_VIS, H)), 1)
logit = F(Vn) - F(Vp)
alpha = float(3.5 / jnp.median(jnp.abs(logit)))
ebm_inf = IsingEBM(NODES, EDGES, alpha * biases, alpha * weights, jnp.array(1.0))
prog_inf = IsingSamplingProgram(ebm_inf, FREE_BLOCKS, CLAMPED_BLOCKS)
sched = SamplingSchedule(n_warmup=600, n_samples=800, steps_per_sample=3)
ncorrect = 0
for idx in range(16):
    clamp = [jnp.array([bool((idx >> k) & 1) for k in range(N_IN)])]
    ps = []
    for ci in range(4):
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.fold_in(key, idx), ci))
        init = hinton_init(k1, ebm_inf, FREE_BLOCKS, ())
        s = sample_states(k2, prog_inf, sched, init, clamp, [Block([out])])
        ps.append(jnp.mean(s[0].astype(jnp.float32)))
    p1 = float(jnp.mean(jnp.array(ps))); pred = int(p1 > 0.5); par = int(PARITY[idx])
    ncorrect += pred == par
print(f"  THRML clamped-sampling LUT accuracy: {ncorrect}/16   "
      f"(circuit = {N_NODES} nodes, {N_EDGES} couplings, trained entirely via sampling)")
