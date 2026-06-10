"""
LUT4 architecture search: which hidden-node circuit is "best"?

Fixed: 4 input nodes + 1 output node. Between them we try different hidden
arrangements (1 or 2 layers, various sizes), wired like an MLP realized as an
undirected energy-based model:

  1-layer (H1):        inputs(4) -- h1(H1) -- output(1)        (both ends touch h1)
  2-layer (H1,H2):     inputs(4) -- h1(H1) -- h2(H2) -- output(1)

We train each on the BINDING worst case (4-input PARITY) and score it on the
metrics that matter for a real circuit:
  - nodes        total node count            (5 + hidden)
  - edges        couplings to wire           (hardware cost)
  - maxdeg       largest node degree         (Z-1 native budget = 12 couplings)
  - acc          best functional accuracy /16 over random inits (capacity)
  - robust       fraction of inits reaching 16/16 (trainability / reliability)

Exact conditional P(y|x) via enumeration of all 2^Htot hidden states -> a clean,
topology-independent capacity measurement (no MCMC noise).
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
N_IN = 4
def popcount(i): return bin(i).count("1")
IDX = jnp.arange(16)
XBITS = jnp.stack([(IDX >> k) & 1 for k in range(N_IN)], axis=1)
X_SPIN = 2.0 * XBITS.astype(jnp.float32) - 1.0
def targets(tt): return 2.0 * jnp.asarray(tt, jnp.float32) - 1.0
FUNCS = {
    "AND4":    [1 if popcount(i) == 4 else 0 for i in range(16)],
    "THR>=2":  [1 if popcount(i) >= 2 else 0 for i in range(16)],
    "PARITY":  [popcount(i) & 1 for i in range(16)],
    "RANDOM":  [int(b) for b in jax.random.bernoulli(jax.random.key(7), 0.5, (16,))],
}

# ---------- build a topology: returns Htot, edge index list (nodes: 0-3 in, 4 out, 5.. hidden) ----------
def build(spec):
    edges = []
    if spec[0] == "1L":
        (H1,) = spec[1:]; Htot = H1
        h1 = range(5, 5 + H1)
        for i in range(4):
            for h in h1: edges.append((i, h))
        for h in h1: edges.append((4, h))
    else:  # "2L"
        H1, H2 = spec[1:]; Htot = H1 + H2
        h1 = range(5, 5 + H1); h2 = range(5 + H1, 5 + H1 + H2)
        for i in range(4):
            for h in h1: edges.append((i, h))
        for a in h1:
            for b in h2: edges.append((a, b))
        for b in h2: edges.append((b, 4))
    return Htot, edges

def max_degree(Htot, edges):
    deg = [0] * (5 + Htot)
    for i, j in edges: deg[i] += 1; deg[j] += 1
    return max(deg)

# ---------- exact conditional logit via hidden enumeration ----------
def make_lossfn(spec):
    Htot, edges = build(spec)
    Ntot = 5 + Htot
    ei = jnp.array([e[0] for e in edges]); ej = jnp.array([e[1] for e in edges])
    n_edges = len(edges)
    Hcfg = (((jnp.arange(2 ** Htot)[:, None] >> jnp.arange(Htot)[None, :]) & 1) * 2 - 1).astype(jnp.float32)
    nH = Hcfg.shape[0]

    def logit(params, x):                         # scalar: log P(y=+1|x) - log P(y=-1|x)
        b_rest, W = params                         # b_rest:(1+Htot,) for [out, hidden..]; W:(n_edges,)
        bias = jnp.zeros(Ntot).at[4:].set(b_rest)
        def negE_for_y(y):
            S = jnp.zeros((nH, Ntot)).at[:, 0:4].set(x).at[:, 4].set(y).at[:, 5:].set(Hcfg)
            return (S @ bias) + jnp.sum(W * (S[:, ei] * S[:, ej]), axis=1)   # -E
        return jax.scipy.special.logsumexp(negE_for_y(1.0)) - jax.scipy.special.logsumexp(negE_for_y(-1.0))

    def loss_acc(params, T):
        lg = jax.vmap(lambda x: logit(params, x))(X_SPIN)
        return jnp.mean(jax.nn.softplus(-lg * T)), jnp.mean((jnp.sign(lg) == T).astype(jnp.float32))
    return Htot, n_edges, loss_acc

def tmap(f, *t): return jax.tree_util.tree_map(f, *t)

def train_config(spec, T, key, R=16, steps=3000, lr=0.05):
    Htot, n_edges, loss_acc = make_lossfn(spec)
    kb, kw = jax.random.split(key)
    params = (0.6 * jax.random.normal(kb, (R, 1 + Htot)), 0.6 * jax.random.normal(kw, (R, n_edges)))

    @jax.jit
    def run(params):
        m = tmap(jnp.zeros_like, params); v = tmap(jnp.zeros_like, params)
        def batched(p):
            l, a = jax.vmap(lambda pp: loss_acc(pp, T))(p)
            return jnp.sum(l), (l, a)
        def step(carry, t):
            params, m, v = carry
            (_, (ls, accs)), g = jax.value_and_grad(batched, has_aux=True)(params)
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = tmap(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
            v = tmap(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
            params = tmap(lambda pp, mm, vv: pp - lr * (mm / (1 - b1 ** (t + 1))) / (jnp.sqrt(vv / (1 - b2 ** (t + 1))) + eps),
                          params, m, v)
            return (params, m, v), accs
        (_, _, _), ah = jax.lax.scan(step, (params, m, v), jnp.arange(steps))
        final = ah[-1]
        return jnp.max(final), jnp.mean(final == 1.0)
    best_acc, robust = run(params)
    return Htot, n_edges, max_degree(Htot, build(spec)[1]), float(best_acc), float(robust)

# ====================== sweep ======================
CONFIGS = [
    ("1L", 2), ("1L", 3), ("1L", 4), ("1L", 5), ("1L", 6),
    ("2L", 2, 1), ("2L", 2, 2), ("2L", 3, 2), ("2L", 2, 3),
    ("2L", 3, 3), ("2L", 4, 2), ("2L", 2, 4), ("2L", 4, 4),
]
print("=" * 78)
print("LUT4 CIRCUIT SEARCH  —  target = PARITY (worst-case 4-input function)")
print("=" * 78)
print(f"\n{'config':<14}{'nodes':>6}{'hidden':>7}{'edges':>7}{'maxdeg':>8}{'hw<=12':>8}{'acc/16':>8}{'robust':>8}")
print("-" * 78)
key = jax.random.key(1)
T = targets(FUNCS["PARITY"])
rows = []
t0 = time.time()
for spec in CONFIGS:
    key, sk = jax.random.split(key)
    Htot, n_edges, maxdeg, acc, robust = train_config(spec, T, sk)
    name = f"1L({spec[1]})" if spec[0] == "1L" else f"2L({spec[1]},{spec[2]})"
    nodes = 5 + Htot
    rows.append((name, spec, nodes, Htot, n_edges, maxdeg, acc, robust))
    print(f"{name:<14}{nodes:>6}{Htot:>7}{n_edges:>7}{maxdeg:>8}{'yes' if maxdeg<=12 else 'NO':>8}"
          f"{int(round(acc*16)):>6}/16{robust:>8.2f}")
print("-" * 78)
print(f"trained {len(CONFIGS)} configs x 16 restarts, exact (hidden-enumerated)   {time.time()-t0:.1f}s")

# ---------- pick winner: solves parity (16/16), then min nodes, then min edges, then max robustness ----------
solved = [r for r in rows if round(r[6] * 16) == 16]
solved.sort(key=lambda r: (r[2], r[4], -r[7]))            # nodes, edges, -robust
print("\nRANKING among configs that LEARN parity (16/16), best first:")
print(f"  {'config':<12}{'nodes':>6}{'edges':>7}{'robust':>8}")
for r in solved[:6]:
    print(f"  {r[0]:<12}{r[2]:>6}{r[4]:>7}{r[7]:>8.2f}")

if solved:
    w = solved[0]
    print(f"\nBEST CIRCUIT: {w[0]}  ->  {w[2]} nodes ({w[3]} hidden), {w[4]} edges, "
          f"max degree {w[5]} (<=12 OK), trains {w[7]*100:.0f}% of the time")
    # verify the winner across all four functions
    print(f"\nVerifying {w[0]} on all functions:")
    key, sk = jax.random.split(key)
    for fname, tt in FUNCS.items():
        key, sk = jax.random.split(key)
        _, _, _, acc, robust = train_config(w[1], targets(tt), sk)
        print(f"  {fname:<8}: {int(round(acc*16))}/16   (trains {robust*100:.0f}% of inits)")
