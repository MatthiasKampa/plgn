"""
Push further on "how many nodes does a LUT4 need?":

(a) DISTRIBUTION of minimal hidden count over many 4-input functions (not just
    parity + 3 examples). Is parity really the ceiling, or does some function need >4?

(b) SKIP edges: compare the restricted Boltzmann machine (output depends on input
    only through hidden) vs a general BM with direct input<->output couplings.
    Prediction: skip edges = a built-in linear separator, so linearly-separable
    functions collapse to H=0, while parity (not linear) keeps its hidden requirement.

1-layer RBM, exact closed-form conditional P(y|x) (logcosh free energy), trained
for hundreds of functions x many restarts in parallel via vmap.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
N_IN = 4
def popcount(i): return bin(i).count("1")
IDX = jnp.arange(16)
XBITS = jnp.stack([(IDX >> k) & 1 for k in range(N_IN)], axis=1)
X_SPIN = 2.0 * XBITS.astype(jnp.float32) - 1.0           # (16,4)

def logcosh(x):
    a = jnp.abs(x)
    return a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)

# ---------- function set: a few structured-hard ones + many random ----------
def bits(i): return [(i >> k) & 1 for k in range(N_IN)]
STRUCTURED = {
    "PARITY (x0^x1^x2^x3)":        [popcount(i) & 1 for i in range(16)],
    "EXACTLY-2 (symmetric)":       [1 if popcount(i) == 2 else 0 for i in range(16)],
    "(x0&x1)^(x2&x3)":             [((bits(i)[0] & bits(i)[1]) ^ (bits(i)[2] & bits(i)[3])) for i in range(16)],
    "x0 ^ (x1&x2&x3)":             [(bits(i)[0] ^ (bits(i)[1] & bits(i)[2] & bits(i)[3])) for i in range(16)],
}
F_RAND = 400
key = jax.random.key(20)
key, sk = jax.random.split(key)
rand_tt = jax.random.bernoulli(sk, 0.5, (F_RAND, 16)).astype(jnp.float32)
struct_tt = jnp.array(list(STRUCTURED.values()), jnp.float32)
TT = jnp.concatenate([struct_tt, rand_tt], axis=0)        # (Ffun,16)
Tfun = 2.0 * TT - 1.0
Ffun = Tfun.shape[0]
N_STRUCT = len(STRUCTURED)

# ---------- closed-form conditional logit (single model) ----------
def logit_fn(skip):
    def logit_one(p):                                     # -> (16,)
        b_out, c, Win, Wout, skipw = p                    # scalars/(H,)/(H,4)/(H,)/(4,)
        def negF(y):
            net = c[None, :] + X_SPIN @ Win.T + (Wout * y)[None, :]   # (16,H)
            val = b_out * y + jnp.sum(logcosh(net), axis=1)          # (16,)
            if skip:
                val = val + (X_SPIN @ skipw) * y
            return val
        return negF(1.0) - negF(-1.0)
    return logit_one

def tmap(f, *t): return jax.tree_util.tree_map(f, *t)

def best_acc_per_fun(H, skip, key, R=12, steps=1500, lr=0.05):
    """Train all functions x R restarts at hidden count H; return best acc/16 per function."""
    M = Ffun * R
    Trep = jnp.repeat(Tfun, R, axis=0)                    # (M,16)
    ks = jax.random.split(key, 5)
    params = (
        0.1 * jax.random.normal(ks[0], (M,)),             # b_out
        0.6 * jax.random.normal(ks[1], (M, H)),           # c
        0.6 * jax.random.normal(ks[2], (M, H, N_IN)),     # Win
        0.6 * jax.random.normal(ks[3], (M, H)),           # Wout
        jnp.zeros((M, N_IN)),                             # skipw (0; trained only if skip)
    )
    logit_one = logit_fn(skip)

    def loss_acc(p, T):
        lg = logit_one(p)
        return jnp.mean(jax.nn.softplus(-lg * T)), jnp.mean((jnp.sign(lg) == T).astype(jnp.float32))

    @jax.jit
    def run(params):
        m = tmap(jnp.zeros_like, params); v = tmap(jnp.zeros_like, params)
        def batched(p):
            l, a = jax.vmap(loss_acc)(p, Trep)
            return jnp.sum(l), a
        def step(carry, t):
            params, m, v = carry
            (_, accs), g = jax.value_and_grad(batched, has_aux=True)(params)
            if not skip:                                  # freeze skip weights for the RBM family
                g = (g[0], g[1], g[2], g[3], jnp.zeros_like(g[4]))
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = tmap(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
            v = tmap(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
            params = tmap(lambda pp, mm, vv: pp - lr * (mm / (1 - b1 ** (t + 1))) / (jnp.sqrt(vv / (1 - b2 ** (t + 1))) + eps),
                          params, m, v)
            return (params, m, v), accs
        (_, _, _), ah = jax.lax.scan(step, (params, m, v), jnp.arange(steps))
        return ah[-1]                                     # (M,)
    accs = run(params)
    return accs.reshape(Ffun, R).max(axis=1)              # (Ffun,)

# ---------- sweep H, record minimal H to reach 16/16, per family ----------
H_SWEEP = [0, 1, 2, 3, 4, 5, 6]
BIG = 99
results = {}
for skip in (False, True):
    fam = "RBM+skip" if skip else "RBM"
    min_h = jnp.full((Ffun,), BIG)
    t0 = time.time()
    for H in H_SWEEP:
        key, sk = jax.random.split(key)
        acc = best_acc_per_fun(H, skip, sk)
        solved = acc >= (16 - 0.5) / 16                   # 16/16
        min_h = jnp.where(solved & (min_h == BIG), H, min_h)
    results[fam] = min_h
    print(f"[{fam}] swept H={H_SWEEP} over {Ffun} functions x12 restarts   {time.time()-t0:.1f}s")

# ---------- report ----------
print("\n" + "=" * 70)
print(f"MINIMAL HIDDEN-NODE COUNT — distribution over {Ffun} LUT4 functions")
print(f"({N_STRUCT} structured-hard + {F_RAND} random)")
print("=" * 70)
print(f"\n{'min hidden H':<14}" + "".join(f"{('RBM' if i==0 else 'RBM+skip'):>12}" for i in range(2)))
print(f"{'(=> nodes)':<14}{'(5+H)':>12}{'(5+H)':>12}")
print("-" * 70)
rbm, skp = results["RBM"], results["RBM+skip"]
for H in H_SWEEP:
    cr = int(jnp.sum(rbm == H)); cs = int(jnp.sum(skp == H))
    print(f"H={H} ({5+H:>2} nodes){'':<2}{cr:>10}  {100*cr/Ffun:4.0f}%{cs:>6}  {100*cs/Ffun:4.0f}%")
unsolved_r = int(jnp.sum(rbm == BIG)); unsolved_s = int(jnp.sum(skp == BIG))
print("-" * 70)
print(f"{'unsolved<=6':<14}{unsolved_r:>10}        {unsolved_s:>6}")
print(f"{'MAX min-H':<14}{int(jnp.max(jnp.where(rbm==BIG,0,rbm))):>10}        {int(jnp.max(jnp.where(skp==BIG,0,skp))):>6}")

print("\nStructured functions (min hidden H):")
print(f"  {'function':<26}{'RBM':>6}{'RBM+skip':>10}")
for i, name in enumerate(STRUCTURED):
    r = int(rbm[i]); s = int(skp[i])
    print(f"  {name:<26}{(r if r<BIG else '>6'):>6}{(s if s<BIG else '>6'):>10}")

# which random function(s) are hardest under RBM?
rand_minh = rbm[N_STRUCT:]
hardest = int(jnp.max(jnp.where(rand_minh == BIG, 0, rand_minh)))
print(f"\nHardest random function needs H={hardest} (RBM).  "
      f"Parity needs H={int(rbm[0])}.  -> parity is{'' if hardest<=int(rbm[0]) else ' NOT'} the ceiling.")
