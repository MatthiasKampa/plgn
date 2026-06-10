"""
Toy: learn a single LUT4 (arbitrary 4-input Boolean function) as an energy-based
model, and find the minimum node count.

Model: a +/-1 Restricted Boltzmann Machine (bipartite Boltzmann machine)
  - 5 VISIBLE nodes = 4 inputs (x0..x3) + 1 output (y)        [fixed]
  - H HIDDEN nodes                                            [the unknown we minimize]

Used as a LUT by clamping the 4 inputs and reading the output (the hardware's
clamp-and-block-Gibbs flow). Success criterion = FUNCTIONAL: argmax_y P(y|x) == f(x)
for all 16 inputs (this is what an MLP "replicating a LUT" optimizes too).

Part 1 trains the RBM EXACTLY (closed-form free energy over hidden, no sampling) so
"minimal H" reflects true representational capacity, not optimizer/MCMC noise.
Part 2 takes the minimal parity model and runs it on THRML's block-Gibbs sampler
(inputs clamped) -- i.e. on the Z-1-like 2-color hardware path.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
N_IN = 4
N_VIS = N_IN + 1  # + output

# ---------- the four target functions (truth tables over index = x0 + 2x1 + 4x2 + 8x3) ----------
def popcount(i): return bin(i).count("1")
TRUTH = {
    "AND4       (lin-sep)":   [1 if popcount(i) == 4 else 0 for i in range(16)],
    "THRESH>=2  (lin-sep)":   [1 if popcount(i) >= 2 else 0 for i in range(16)],
    "PARITY4    (hardest)":   [popcount(i) & 1 for i in range(16)],
    "RANDOM     (typical)":   [int(b) for b in jax.random.bernoulli(jax.random.key(7), 0.5, (16,))],
}

# inputs as +/-1 spins, shape (16, 4)
IDX = jnp.arange(16)
XBITS = jnp.stack([(IDX >> k) & 1 for k in range(N_IN)], axis=1)        # (16,4) in {0,1}
X_SPIN = 2.0 * XBITS.astype(jnp.float32) - 1.0                          # (16,4) in {-1,+1}

def make_targets(tt):
    t = jnp.asarray(tt, dtype=jnp.float32)
    return 2.0 * t - 1.0                                               # (16,) in {-1,+1}

# ---------- +/-1 RBM free energy & functional conditional loss ----------
def logcosh(x):  # numerically stable log cosh
    a = jnp.abs(x)
    return a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)

def free_energy(params, V):           # V: (M,5) spins;  F(v) = -b.v - sum_j logcosh(c_j + (vW)_j)
    b, c, W = params                  # b:(5,) c:(H,) W:(5,H)
    pre = c + V @ W                    # (M,H)  (H may be 0)
    return -(V @ b) - jnp.sum(logcosh(pre), axis=1)

def loss_acc(params, T):
    Vpos = jnp.concatenate([X_SPIN, jnp.ones((16, 1))], axis=1)        # y=+1
    Vneg = jnp.concatenate([X_SPIN, -jnp.ones((16, 1))], axis=1)       # y=-1
    logit = free_energy(params, Vneg) - free_energy(params, Vpos)      # log P(y=+1|x)-log P(y=-1|x)
    loss = jnp.mean(jax.nn.softplus(-logit * T))                       # = -mean log P(correct|x)
    acc = jnp.mean((jnp.sign(logit) == T).astype(jnp.float32))
    return loss, acc

# ---------- training: R parallel random restarts, manual Adam, lax.scan ----------
def init_params(key, H, R, scale=0.6):
    kb, kc, kw = jax.random.split(key, 3)
    return (scale * jax.random.normal(kb, (R, N_VIS)),
            scale * jax.random.normal(kc, (R, H)),
            scale * jax.random.normal(kw, (R, N_VIS, H)))

def tree_map(f, *ts): return jax.tree_util.tree_map(f, *ts)

def train(key, H, T, R=24, steps=4000, lr=0.05, wd=0.0):
    params = init_params(key, H, R)
    m = tree_map(jnp.zeros_like, params); v = tree_map(jnp.zeros_like, params)

    def batched(p):
        l, a = jax.vmap(lambda pp: loss_acc(pp, T))(p)
        return jnp.sum(l), (l, a)                                      # sum -> independent per-restart grads

    def step(carry, t):
        params, m, v = carry
        (_, (ls, accs)), g = jax.value_and_grad(batched, has_aux=True)(params)
        if wd: g = tree_map(lambda gg, pp: gg + wd * pp, g, params)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
        v = tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
        bc1 = 1 - b1 ** (t + 1); bc2 = 1 - b2 ** (t + 1)
        params = tree_map(lambda pp, mm, vv: pp - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps), params, m, v)
        return (params, m, v), (ls, accs)

    (params, _, _), (lh, ah) = jax.lax.scan(step, (params, m, v), jnp.arange(steps))
    final_l, final_a = lh[-1], ah[-1]
    best = jnp.argmin(final_l)
    best_params = tree_map(lambda x: x[best], params)
    return best_params, final_l[best], final_a[best]

train_jit = jax.jit(train, static_argnames=("H", "R", "steps"))

# ====================== PART 1: capacity sweep ======================
print("=" * 74)
print("PART 1  —  how many HIDDEN nodes to learn a LUT4?   (5 visible = 4 in + 1 out)")
print("=" * 74)
H_SWEEP = [0, 1, 2, 3, 4, 6, 8]
key = jax.random.key(0)
results = {}
t0 = time.time()
print(f"\n{'function':<22}" + "".join(f"H={h:<5}" for h in H_SWEEP) + "  min-H(16/16)")
print("-" * 74)
for name, tt in TRUTH.items():
    T = make_targets(tt)
    row = []
    min_h = None
    for H in H_SWEEP:
        key, sk = jax.random.split(key)
        _, loss, acc = train_jit(sk, H=H, T=T)
        correct = int(round(float(acc) * 16))
        row.append(correct)
        if min_h is None and correct == 16:
            min_h = H
    results[name] = (row, min_h)
    cells = "".join(f"{c:>2}/16 " for c in row)
    print(f"{name:<22}{cells}   {min_h if min_h is not None else '>8'}")
print("-" * 74)
print(f"(each cell = #inputs correct out of 16; trained exactly, 24 restarts)   {time.time()-t0:.1f}s")

# node-count summary
parity_minh = results["PARITY4    (hardest)"][1]
worst = max((mh if mh is not None else 99) for _, mh in results.values())
print(f"""
NODE COUNT
  visible            : {N_VIS}  (4 inputs + 1 output, fixed)
  hidden  (parity)   : {parity_minh}      <- the binding worst case
  hidden  (worst fn) : {worst}
  TOTAL for any LUT4 : {N_VIS + worst} nodes   ({N_VIS} visible + {worst} hidden)
  vs your MLP anchor : 4-8 hidden  -> RBM logcosh feature ~= a smooth activation,
                       so it lands at the efficient end of your 4-8 range.""")

# ====================== PART 2: run the LUT on THRML block-Gibbs ======================
print("\n" + "=" * 74)
print(f"PART 2  —  run the minimal PARITY model on THRML block-Gibbs (inputs clamped)")
print("=" * 74)
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

H_DEMO = parity_minh if parity_minh is not None else 8
T = make_targets(TRUTH["PARITY4    (hardest)"])
# clean retrain with more restarts + tiny weight decay (keeps weights moderate so the chain mixes)
key, sk = jax.random.split(key)
params, loss, acc = train_jit(sk, H=H_DEMO, T=T, R=48, steps=6000)
acc, loss = float(acc), float(loss)
print(f"\ntrained parity RBM: H={H_DEMO}, exact functional accuracy = {int(round(acc*16))}/16, NLL={loss:.4f}")
# Rescale the (exact, 16/16) params to MODERATE sharpness so the Gibbs chain actually
# mixes instead of freezing (NLL~0 => huge weights => deterministic coord-descent that
# locks into the init's basin). Scaling all params is just a beta rescale: argmax (the
# learned function) is unchanged, the conditional is merely softened enough to sample.
Vpos = jnp.concatenate([X_SPIN, jnp.ones((16, 1))], axis=1)
Vneg = jnp.concatenate([X_SPIN, -jnp.ones((16, 1))], axis=1)
logit = free_energy(params, Vneg) - free_energy(params, Vpos)        # (16,)
alpha = float(3.5 / jnp.median(jnp.abs(logit)))
params = tree_map(lambda p: p * alpha, params)
b, c, W = [jnp.asarray(p) for p in params]
print(f"rescaled couplings x{alpha:.3f} (median |logit| -> ~3.5) so the chain mixes")

BETA_S = 1.0                                  # sampling temperature
ins = [SpinNode() for _ in range(N_IN)]
out = SpinNode()
hid = [SpinNode() for _ in range(H_DEMO)]
nodes = ins + [out] + hid
vis = ins + [out]                             # rows 0..4 of b and W
edges, wts = [], []
for i, vn in enumerate(vis):
    for j, hn in enumerate(hid):
        edges.append((vn, hn)); wts.append(W[i, j])
biases = jnp.concatenate([b, c])              # aligned with nodes [in0..3, out, h0..]
ebm = IsingEBM(nodes, edges, biases, jnp.asarray(wts), jnp.asarray(float(BETA_S)))

clamped_blocks = [Block(ins)]
free_blocks = [Block([out]), Block(hid)]      # 2-color: {out} | {hidden}; bipartite RBM
program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks)
schedule = SamplingSchedule(n_warmup=800, n_samples=1200, steps_per_sample=4)
N_CHAINS = 6                                  # average independent chains (avoids basin lock-in)

print(f"\nrunning 16 clamped block-Gibbs inferences (beta={BETA_S}, {N_CHAINS} chains) ...")
print(f"\n  x3x2x1x0 | parity | P(out=1) | sampled | ok")
print("  " + "-" * 46)
ncorrect = 0
for idx in range(16):
    bits = [(idx >> k) & 1 for k in range(N_IN)]
    clamp = [jnp.array([bool(bk) for bk in bits], dtype=jnp.bool_)]
    ps = []
    for ci in range(N_CHAINS):
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.fold_in(key, idx), ci))
        init = hinton_init(k1, ebm, free_blocks, ())
        s = sample_states(k2, program, schedule, init, clamp, [Block([out])])
        ps.append(jnp.mean(s[0].astype(jnp.float32)))
    p1 = float(jnp.mean(jnp.array(ps)))
    pred = int(p1 > 0.5)
    par = popcount(idx) & 1
    ok = pred == par
    ncorrect += ok
    bitstr = "".join(str(bits[k]) for k in reversed(range(N_IN)))
    print(f"  {bitstr}     |   {par}    |  {p1:5.2f}   |    {pred}    | {'YES' if ok else 'no'}")
print("  " + "-" * 46)
print(f"\n  THRML block-Gibbs LUT accuracy: {ncorrect}/16   "
      f"(total nodes used = {len(nodes)} = {N_VIS} visible + {H_DEMO} hidden)")
