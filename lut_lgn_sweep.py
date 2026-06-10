"""
Does MORE hidden capacity stabilize whole-network CD on the deep LGN?

Same task/network as lut_lgn.py (f = MAJ3(x0,x1,x2) XOR MAJ3(x3,x4,x5)), same plain
contrastive divergence (both phases sampled on THRML, negative phase re-init each step).
We sweep the per-gate hidden width and report, with BEST-CHECKPOINT tracking:
  best  = peak accuracy ever (what an FPGA early-stop would keep)
  final = end-of-training accuracy (stability: does it hold near best or regress?)
  peak@ = step of the best
  oAdeg = degree of the interior gate node = HA + HC  (Z-1 budget is <=12)
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

jax.config.update("jax_platform_name", "cpu")
BETA = 1.0; P = 64

bits = ((jnp.arange(P)[:, None] >> jnp.arange(6)) & 1)
X = 2.0 * bits.astype(jnp.float32) - 1.0
maj = lambda a, b, c: ((a + b + c) >= 2).astype(jnp.int32)
Y = (maj(bits[:, 0], bits[:, 1], bits[:, 2]) ^ maj(bits[:, 3], bits[:, 4], bits[:, 5]))
T = 2.0 * Y.astype(jnp.float32) - 1.0
POS_CB = (jnp.concatenate([X, T[:, None]], 1) > 0); NEG_CB = (X > 0)
SCHED = SamplingSchedule(n_warmup=120, n_samples=160, steps_per_sample=2)   # matched to lut_lgn.py baseline
SCHED_EVAL = SamplingSchedule(n_warmup=200, n_samples=300, steps_per_sample=2)

@eqx.filter_jit
def sample_phase(program, sched, keys, init, clamp, obs):
    def one(k, i, c): return sample_states(k, program, sched, i, [c], [obs])[0]
    return jax.vmap(one)(keys, init, clamp)

def build(HA, HB, HC):
    I_IN = list(range(6)); I_OA, I_OB, I_OC = 6, 7, 8
    I_HA = list(range(9, 9 + HA)); I_HB = list(range(9 + HA, 9 + HA + HB)); I_HC = list(range(9 + HA + HB, 9 + HA + HB + HC))
    N = 9 + HA + HB + HC; ALLH = I_HA + I_HB + I_HC
    edges = []
    for h in I_HA:
        for u in [0, 1, 2, I_OA]: edges.append((u, h))
    for h in I_HB:
        for u in [3, 4, 5, I_OB]: edges.append((u, h))
    for h in I_HC:
        for u in [I_OA, I_OB, I_OC]: edges.append((u, h))
    deg = [0] * N
    for a, b in edges: deg[a] += 1; deg[b] += 1
    NODES = [SpinNode() for _ in range(N)]; EDGES = [(NODES[a], NODES[b]) for a, b in edges]
    k = jax.random.key(0); k, kb, kw = jax.random.split(k, 3)
    b0 = jnp.concatenate([jnp.zeros(9), 0.1 * jax.random.normal(kb, (HA + HB + HC,))])
    w0 = 0.3 * jax.random.normal(kw, (len(edges),))
    BASE = IsingEBM(NODES, EDGES, b0, w0, jnp.array(BETA))
    blk = lambda idxs: Block([NODES[i] for i in idxs])
    return dict(N=N, nE=len(edges), EI=jnp.array([a for a, b in edges]), EJ=jnp.array([b for a, b in edges]),
                maxdeg=max(deg), oadeg=deg[I_OA], BASE=BASE, b0=b0, w0=w0, I_IN=I_IN, I_OC=I_OC,
                NEG_CLAMP=[blk(I_IN)], NEG_FREE=[blk([I_OA, I_OB, I_OC]), blk(ALLH)], NEG_OBS=blk([I_OA, I_OB, I_OC] + ALLH), NEG_OBS_IDX=[I_OA, I_OB, I_OC] + ALLH,
                POS_CLAMP=[blk(I_IN + [I_OC])], POS_FREE=[blk([I_OA, I_OB]), blk(ALLH)], POS_OBS=blk([I_OA, I_OB] + ALLH), POS_OBS_IDX=[I_OA, I_OB] + ALLH)

def rebuild(net, b, w): return eqx.tree_at(lambda e: (e.biases, e.weights), net['BASE'], (b, w))
def assemble(net, sampled, obs_idx, clamp_idx, clamp_vals):
    S = sampled.shape[1]; full = jnp.zeros((P, S, net['N']))
    full = full.at[:, :, jnp.array(clamp_idx)].set(jnp.broadcast_to(clamp_vals[:, None, :], (P, S, len(clamp_idx))))
    full = full.at[:, :, jnp.array(obs_idx)].set(2.0 * sampled.astype(jnp.float32) - 1.0)
    return full
def moments(net, full): return full.mean(1).mean(0), (full[:, :, net['EI']] * full[:, :, net['EJ']]).mean(1).mean(0)

def pos_phase(net, b, w, key):
    ebm = rebuild(net, b, w); prog = IsingSamplingProgram(ebm, net['POS_FREE'], net['POS_CLAMP'])
    s = sample_phase(prog, SCHED, jax.random.split(key, P), hinton_init(key, ebm, net['POS_FREE'], (P,)), POS_CB, net['POS_OBS'])
    return moments(net, assemble(net, s, net['POS_OBS_IDX'], net['I_IN'] + [net['I_OC']], jnp.concatenate([X, T[:, None]], 1)))
def neg_phase(net, b, w, key):
    ebm = rebuild(net, b, w); prog = IsingSamplingProgram(ebm, net['NEG_FREE'], net['NEG_CLAMP'])
    s = sample_phase(prog, SCHED, jax.random.split(key, P), hinton_init(key, ebm, net['NEG_FREE'], (P,)), NEG_CB, net['NEG_OBS'])
    return moments(net, assemble(net, s, net['NEG_OBS_IDX'], net['I_IN'], X))
def eval_acc(net, b, w, key, sharpen=2.0):
    ebm = rebuild(net, b * sharpen, w * sharpen); prog = IsingSamplingProgram(ebm, net['NEG_FREE'], net['NEG_CLAMP'])
    s = sample_phase(prog, SCHED_EVAL, jax.random.split(key, P), hinton_init(key, ebm, net['NEG_FREE'], (P,)), NEG_CB, net['NEG_OBS'])
    oc = s[:, :, 2].astype(jnp.float32).mean(1)
    return float(jnp.mean(((oc > 0.5).astype(jnp.int32) == Y).astype(jnp.float32)))

def train_cd(net, key, STEPS=220, WD=2e-3):
    b, w = net['b0'], net['w0']
    mb = jnp.zeros(net['N']); vb = jnp.zeros(net['N']); mw = jnp.zeros(net['nE']); vw = jnp.zeros(net['nE'])
    best, peak, traj = -1.0, 0, []
    for step in range(1, STEPS + 1):
        key, kp, kn = jax.random.split(key, 3)
        sbp, eep = pos_phase(net, b, w, kp); sbn, een = neg_phase(net, b, w, kn)
        gb = -BETA * (sbp - sbn) + WD * b; gw = -BETA * (eep - een) + WD * w
        lr = 0.01 + 0.04 * (1.0 - step / STEPS)
        b1, b2, eps, t = 0.9, 0.999, 1e-8, step
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
        b = b - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
        w = w - lr * (mw / (1 - b1 ** t)) / (jnp.sqrt(vw / (1 - b2 ** t)) + eps)
        if step % 12 == 0 or step == 1:
            key, ke = jax.random.split(key); acc = eval_acc(net, b, w, ke)
            traj.append(int(round(acc * P)))
            if acc > best: best, peak = acc, step
    key, ke = jax.random.split(key); final = eval_acc(net, b, w, ke, sharpen=2.5)
    return int(round(best * P)), int(round(final * P)), peak, traj

CONFIGS = [(4, 4, 3), (6, 6, 4), (8, 8, 4), (12, 12, 6)]
print("does more hidden capacity stabilize whole-network CD?  (task solved = 64/64, chance = 32)\n")
print(f"{'(HA,HB,HC)':<13}{'nodes':>6}{'oAdeg':>7}{'<=12':>6}{'best':>8}{'final':>8}{'peak@':>7}   trajectory(/64)")
print("-" * 100)
for (HA, HB, HC) in CONFIGS:
    net = build(HA, HB, HC); t0 = time.time()
    best, final, peak, traj = train_cd(net, jax.random.key(1))
    fit = 'Y' if net['oadeg'] <= 12 else 'N'
    print(f"({HA},{HB},{HC})      {net['N']:>5}{net['oadeg']:>7}{fit:>6}{best:>6}/64{final:>6}/64{peak:>7}   {traj}  ({time.time()-t0:.0f}s)", flush=True)
print("-" * 100)
print("read: 'best' rising with width => more capacity helps; 'final' near 'best' => stable (no peak-then-regress).")
