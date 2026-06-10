"""
Resolve the confound: is parity's "needs 4 hidden" a REPRESENTATIONAL floor, or
just a TRAINABILITY artifact?

Logic: in the +skip model the skip term is purely LINEAR in the output logit, and
parity is uncorrelated with every linear function -> skip can't help represent parity.
Yet +skip learned parity at H=2. So pure RBM must also REPRESENT parity at H=2;
it just may rarely FIND it. Probe with many restarts to find the true floor.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
IDX = jnp.arange(16)
X_SPIN = 2.0 * jnp.stack([(IDX >> k) & 1 for k in range(4)], 1).astype(jnp.float32) - 1.0
T = 2.0 * jnp.array([bin(i).count("1") & 1 for i in range(16)], jnp.float32) - 1.0   # parity, +/-1

def logcosh(x):
    a = jnp.abs(x); return a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)

def tmap(f, *t): return jax.tree_util.tree_map(f, *t)

def logit_one(p, use_skip):
    b_out, c, Win, Wout, skipw = p
    def negF(y):
        net = c[None, :] + X_SPIN @ Win.T + (Wout * y)[None, :]
        v = b_out * y + jnp.sum(logcosh(net), axis=1)
        return v + ((X_SPIN @ skipw) * y if use_skip else 0.0)
    return negF(1.0) - negF(-1.0)

def probe(H, R, steps, key, use_skip=False, lr=0.05):
    ks = jax.random.split(key, 5)
    params = (0.1 * jax.random.normal(ks[0], (R,)),
              0.6 * jax.random.normal(ks[1], (R, H)),
              0.6 * jax.random.normal(ks[2], (R, H, 4)),
              0.6 * jax.random.normal(ks[3], (R, H)),
              jnp.zeros((R, 4)))
    def loss_acc(p):
        lg = logit_one(p, use_skip)
        return jnp.mean(jax.nn.softplus(-lg * T)), jnp.mean((jnp.sign(lg) == T).astype(jnp.float32))
    @jax.jit
    def run(params):
        m = tmap(jnp.zeros_like, params); v = tmap(jnp.zeros_like, params)
        def batched(p):
            l, a = jax.vmap(loss_acc)(p); return jnp.sum(l), a
        def step(carry, t):
            params, m, v = carry
            (_, accs), g = jax.value_and_grad(batched, has_aux=True)(params)
            if not use_skip: g = (*g[:4], jnp.zeros_like(g[4]))
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = tmap(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
            v = tmap(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
            params = tmap(lambda pp, mm, vv: pp - lr * (mm / (1 - b1 ** (t + 1))) / (jnp.sqrt(vv / (1 - b2 ** (t + 1))) + eps), params, m, v)
            return (params, m, v), accs
        (params, _, _), ah = jax.lax.scan(step, (params, m, v), jnp.arange(steps))
        return ah[-1], params
    accs, params = run(params)
    return accs, params

print("=" * 64)
print("PARITY floor probe — pure RBM, 512 restarts, 3000 steps each")
print("=" * 64)
print(f"\n{'H':>3}{'nodes':>7}{'best acc':>10}{'# solved (16/16)':>20}{'solve rate':>12}")
print("-" * 64)
key = jax.random.key(0)
R = 512
for H in [1, 2, 3, 4]:
    key, sk = jax.random.split(key)
    t0 = time.time()
    accs, _ = probe(H, R, 3000, sk)
    n_solved = int(jnp.sum(accs >= 15.5 / 16))
    print(f"{H:>3}{5+H:>7}{int(round(float(jnp.max(accs))*16)):>8}/16{n_solved:>16}/{R}{n_solved/R:>11.1%}   ({time.time()-t0:.1f}s)")
print("-" * 64)

# does +skip actually USE the skip weights to learn parity at H=2?  (it shouldn't)
key, sk = jax.random.split(key)
accs, params = probe(2, 256, 3000, sk, use_skip=True)
best = int(jnp.argmax(accs))
skipw_best = params[4][best]
print(f"\n+skip @ H=2: best acc {int(round(float(jnp.max(accs))*16))}/16; "
      f"learned |skip weights| of best solver = {[round(float(w),3) for w in skipw_best]}")
print(f"  -> mean |skipw| = {float(jnp.mean(jnp.abs(skipw_best))):.3f}  "
      f"(near 0 confirms skip is unused for parity; it only eased optimization)")
