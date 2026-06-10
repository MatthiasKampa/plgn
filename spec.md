# Extropic Z-1 — Next-Generation Thermodynamic Chip: Specification Dossier

**Document date:** 2026-06-10
**Subject:** Extropic's next-generation Thermodynamic Sampling Unit (TSU), publicly named **Z-1 / Z1**.
**Revision (2026-06-10):** *Connectivity* corrected to the paper's exact connection-rule graph (G₁₂); see **Architecture & connectivity**. Earlier drafts described the degree only vaguely ("~12 neighbors").

## Executive summary

Extropic is a probabilistic-computing startup (founders Guillaume Verdon / "Beff Jezos" and Trevor McCourt) building **Thermodynamic Sampling Units (TSUs)** — chips built entirely from standard CMOS transistors that sample directly from programmable probability distributions using transistor thermal/shot noise as their entropy source. After two early devices — the **X0** test chip (dozens of probabilistic circuits) and the **XTR-0** development platform — the company has announced **Z-1 (Z1)**, its first production-scale TSU, targeting "hundreds of thousands of probabilistic circuits per chip, and millions per card," with early access stated for **2026**. The headline efficiency claim is "~10,000× less energy than GPUs," but it is critical to note this is a **simulation-based, system-level estimate on a small image benchmark (Fashion-MNIST)**, not a measured result from production Z-1 silicon.

---

## Status & confidence — labeling scheme

Each spec is tagged with one of the following:

- **[confirmed]** — Stated by Extropic itself (extropic.ai pages, their arXiv paper, their GitHub) or directly verifiable.
- **[Extropic-stated target]** — A forward-looking number Extropic gives for Z-1 specifically (design target, not yet measured silicon).
- **[paper/prototype value]** — A concrete figure from Extropic's arXiv paper or X0 prototype, which describes the *architecture and simulations*, not necessarily the final Z-1 part.
- **[user-estimate]** — A figure supplied in the task brief that I could **not** corroborate in any source. Flagged explicitly.
- **[secondary]** — Reported by third-party tech coverage only; treat as corroboration, not authority.
- **[not publicly disclosed]** — No reliable public value exists. Not invented here.

> **Single most important caveat up front:** The task brief's target of **"~512 × 512 nodes ≈ 262,144 pbits"** is **NOT substantiated by any source I found.** Every Extropic statement and all secondary coverage describe Z-1 as **"hundreds of thousands of pbits"** or, more specifically, **"~250,000 pbits per chip."** 512×512 = 262,144 is *numerically close* to 250,000, so it appears to be a **back-calculated estimate (the user's own), not an Extropic-published grid dimension.** Extropic has not publicly disclosed Z-1's exact grid geometry. See "Open questions."

---

## Specification table

| Field | Value | Status | Source |
|---|---|---|---|
| Chip name / codename | **Z-1** (also written **Z1**) | [confirmed] | [extropic.ai/hardware](https://extropic.ai/hardware) |
| Roadmap position | 3rd public device: **X0** (test chip) → **XTR-0** (dev platform) → **Z-1** (first production-scale TSU) | [confirmed] | [extropic.ai/hardware](https://extropic.ai/hardware) |
| Product class / tagline | "Accelerated Intelligence" production chip | [confirmed] | [extropic.ai/hardware](https://extropic.ai/hardware) |
| Scale — pbits per chip | **"Hundreds of thousands of probabilistic circuits per chip"**; reported as **~250,000 pbits** | [Extropic-stated target] / [secondary] | [extropic.ai/hardware](https://extropic.ai/hardware); [vktr.com](https://www.vktr.com/ai-news/extropic-claims-10000x-energy-savings-with-new-probabilistic-ai-chip/) |
| Scale — per card | **"Millions per card"** (multiple TSUs chained) | [Extropic-stated target] | [extropic.ai/hardware](https://extropic.ai/hardware) |
| Grid dimensions | **Not publicly disclosed.** ~512×512 (≈262,144) is a *user estimate*, unconfirmed; Extropic says "hundreds of thousands" / "~250k" | [user-estimate] / [not publicly disclosed] | — (no source states 512×512) |
| Connectivity — degree per node | **12** (default graph **G₁₂**). Paper defines a family **G₈/G₁₂/G₁₆/G₂₀/G₂₄** (degrees 8–24); 12 "used in most cases" | [paper/prototype value] | [arXiv 2510.23972v2 §III + Table 2](https://arxiv.org/html/2510.23972v2) |
| Connection rules (exact graph) | Edges from **"connection rules"** `(a,b)`, each a C4 (90°-rotation) orbit ⇒ **4 edges per rule**. **G₁₂ = {(0,1),(4,1),(9,10)}** = orthogonal nearest-neighbor + skew "skip" connections at length **√17 ≈ 4.12** and **√181 ≈ 13.45** | [paper/prototype value] | [arXiv 2510.23972v2 Table 2 (App. D.2)](https://arxiv.org/html/2510.23972v2) |
| Self-bias ("the +1") | **Per-node bias term** — *not* a self-loop. In the reverse/denoising EBM each visible node additionally has **one clamped coupling** to the corresponding node of the previous denoising step | [paper/prototype value] | [arXiv 2510.23972v2 §III, Fig 9](https://arxiv.org/html/2510.23972v2) |
| Topology / graph family | Single **L×L** square grid (L≈70), **bipartite via the ordinary `(x+y)` checkerboard** — every connection rule has `a+b` **odd**, so all 12 neighbors are opposite-color. Toroidal/periodic (THRML) vs open-boundary (paper) | [paper/prototype value] | [arXiv 2510.23972v2 §III](https://arxiv.org/html/2510.23972v2) |
| Update scheme | **Block Gibbs** over the two checkerboard colors of the single grid: sample color-1 conditioned on color-2, then vice-versa — a whole color updates in parallel (exact, since each color is conditionally independent given the other) | [paper/prototype value] | [arXiv 2510.23972v2 §III](https://arxiv.org/html/2510.23972v2) |
| Sampling algorithm | Block (blocked) Gibbs sampling; ~K≈1000 Gibbs iterations per denoising step in experiments | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Device technology | **All-transistor, standard CMOS** (no superconductors, no nanomagnets, no cryogenics) | [confirmed] | [extropic.ai/writing/inside-x0-and-xtr-0](https://extropic.ai/writing/inside-x0-and-xtr-0) |
| Operating temperature | **Room temperature** | [confirmed] (X0); implied for Z-1 | [secondary deep-dive](https://www.vastkind.com/extropic-thermodynamic-computing-tsu-deep-dive/) |
| Fabrication process node | **Not publicly disclosed** (paper says "standard CMOS"; studied via PDK at multiple process corners; specific node — e.g. nm — not stated) | [not publicly disclosed] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Entropy / noise source | **True device noise — shot-noise dynamics of subthreshold transistors** (natural thermal/shot noise) | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Energy per sampling cell (E_cell) | **≈ 2 fJ** per cell update (used in the energy model E = T·K_mix·L²·E_cell) | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Autocorrelation / RNG time constant | **τ₀ ≈ 100 ns** (approx. exponential autocorrelation decay of the RNG) | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Cell density estimate | Paper states **"~10⁶ sampling cells could be fit into a 6×6 µm chip"** — ⚠️ see note: **almost certainly a typo for 6×6 mm**; 6×6 µm (=36 µm²) for a million cells is physically impossible | [paper/prototype value] (with error flag) | [arXiv 2510.23972v2 §VI](https://arxiv.org/html/2510.23972v2) |
| Largest model demonstrated (simulation) | **~50,000 cells/nodes** (largest DTM shown); individual EBMs = 70×70 grids | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Probabilistic primitives | **PBIT** (Bernoulli), **PDIT** (categorical), **PMODE** (Gaussian), **PMOG** (mixture of Gaussians) | [confirmed] | [extropic.ai/hardware](https://extropic.ai/hardware); [secondary](https://www.vastkind.com/extropic-thermodynamic-computing-tsu-deep-dive/) |
| Energy-efficiency headline | **"~10,000× more energy efficient than modern algorithms running on GPUs"** | [Extropic-stated, simulation] | [extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one) |
| What the 10,000× is measured against | **Energy per generated sample**, achieving **parity in quality** (FID) with GPU baselines (single-step VAE/GAN; DDPM at varying steps) on **Fashion-MNIST**, via **simulation/system-level analysis** | [paper/prototype value] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Precision / bit-depth of weights & biases | **Not publicly disclosed** as a final Z-1 spec (pbit state is binary {0,1}; bias/weight resolution for the production part not stated) | [not publicly disclosed] | — |
| Sampling throughput (samples/s) | **Not publicly disclosed** as a Z-1 figure (derivable in principle from τ₀≈100 ns + parallelism, but Extropic has not published a headline number) | [not publicly disclosed] | — |
| Clock / update rate | **Not publicly disclosed** for Z-1 (RNG τ₀≈100 ns is the relevant prototype timescale) | [not publicly disclosed] | [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Host interface | **XTR-0 dev platform: CPU + FPGA + two sockets for TSU daughterboards**, low-latency host↔TSU link. Z-1 form factor described as "server racks or desktop expansion slots" (PCIe-class implied, not explicitly stated) | [confirmed] (XTR-0) / [secondary] (Z-1 form factor) | [extropic.ai/writing/inside-x0-and-xtr-0](https://extropic.ai/writing/inside-x0-and-xtr-0); [secondary](https://www.vastkind.com/extropic-thermodynamic-computing-tsu-deep-dive/) |
| Memory / I/O | **Not publicly disclosed** (in-memory/in-place compute is implied by the architecture; no DRAM/HBM spec published) | [not publicly disclosed] | — |
| Software stack | **THRML** — open-source JAX library; v0.1.3; Apache-2.0; released 2025-10-29 | [confirmed] | [github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml) |
| Supported model classes | Energy-based models (EBMs), Ising / Boltzmann machines, sparse & heterogeneous probabilistic graphical models; **Denoising Thermodynamic Models (DTMs)** as the flagship generative algorithm | [confirmed] | [github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml); [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1) |
| Target applications | Generative AI (diffusion-like image generation), probabilistic inference/sampling, energy-based ML; early kits cited going to "AI labs and weather companies" | [confirmed] / [secondary] | [extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one); [therundown.ai](https://www.therundown.ai/p/extropics-10-000x-ai-energy-breakthrough) |
| Availability / timeline | **Z-1 early access 2026.** (X0: Q1 2025; XTR-0: Q3 2025, in beta with partners) | [confirmed] | [extropic.ai/hardware](https://extropic.ai/hardware) |
| Status | Announced; **in design/build** ("moving from breakthrough to buildout"). Not yet shipping production silicon | [confirmed] | [extropic.ai/writing/thermodynamic-computing-from-zero-to-one](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one) |

---

## Roadmap & positioning

Extropic's public hardware roadmap has three steps, all stated on their hardware page:

1. **X0** — "Silicon Prototype," availability Q1 2025. A first test chip of **"dozens of probabilistic circuits"** that runs at **room temperature** and "validated our science and proved that we can make all-transistor Thermodynamic Sampling Units (TSUs)." It demonstrated that the probabilistic primitives (pbit/pdit/pmode/pmog) can be built from ordinary transistors — no exotic nanomagnets, no cryogenics.
2. **XTR-0** — "Experimental Testing & Research Platform," availability Q3 2025. A development board consisting of **a CPU, an FPGA, and two sockets that receive daughterboards hosting TSUs** (it houses X0-class chips). Its purpose is low-latency communication between the Extropic chip and a conventional processor so partners can prototype algorithms. Reported to be in beta with early partners ("AI labs and weather companies").
3. **Z-1 (Z1)** — "Accelerated Intelligence" production chip, **early access 2026.** First production-scale TSU: **"hundreds of thousands of probabilistic circuits per chip, and millions per card,"** "mass-manufacturable using standard CMOS processes." This is the "next-generation chip" the brief asks about.

So Z-1 is the **first chip intended for real workloads at scale**, succeeding the X0 proof-of-concept and the XTR-0 prototyping platform.

## Architecture & connectivity

The architecture is an **Ising / energy-based model realized in silicon**. Each node (pbit) is a probabilistic bit whose state probability is set by a **per-node bias plus the weighted sum of its connected neighbors** — the conditional form of a Boltzmann/Ising model. Connectivity is deliberately **sparse and local** (analogous in spirit to D-Wave's fixed Pegasus/Zephyr lattices — a hardwired interaction graph, not all-to-all), and the paper specifies it **precisely** via *connection rules*. Local connectivity keeps the per-cell circuit small and the chip mass-manufacturable; all-to-all wiring would not scale physically.

**Connection rules (the exact graph).** A rule `(a,b)` connects node `(x,y)` to the four sites `(x+a, y+b)`, `(x−b, y+a)`, `(x−a, y−b)`, `(x+b, y−a)` — the offset `(a,b)` plus its three 90° rotations (a C4 orbit), so **each rule contributes 4 edges**. The default **degree-12 graph "G₁₂" uses three rules — `(0,1)`, `(4,1)`, `(9,10)`** — giving these 12 offsets:

- `(0,1)` → `(0,±1), (±1,0)` — the 4 **orthogonal nearest neighbors** (von Neumann);
- `(4,1)` → `(+4,+1), (−1,+4), (−4,−1), (+1,−4)` — 4 **long-range skew "skip" connections** (length √17 ≈ 4.12 cells);
- `(9,10)` → `(+9,+10), (−10,+9), (−9,−10), (+10,−9)` — 4 longer skew skips (length √181 ≈ 13.45 cells).

So it is **not** three short "distance shells" — it is one nearest-neighbor bond plus **two long-range skip-connections**. The paper's **Table 2** defines a whole family **G₈/G₁₂/G₁₆/G₂₀/G₂₄** (degrees 8–24), with **degree-12 "used in most cases"** — i.e. degree/geometry is a tunable design knob. The "**12 + 1**" is **12 grid couplings + a per-node bias** (there is no true self-loop); in the reverse/denoising model each visible node also carries **one clamped coupling** to the corresponding node of the previous denoising step.

**Why it is bipartite.** Every connection rule has **`a+b` odd**, so all 12 neighbors land on cells of the opposite **`(x+y) mod 2`** parity. The two color classes are therefore just the **ordinary checkerboard of the single L×L grid** — no special bipartite construction is needed, and the parallel two-color block-Gibbs update is *exact*. (The paper's "deep Boltzmann machine / multiple hidden layers" wording does **not** mean two stacked grids: it is *one* grid whose nodes are randomly partitioned into visible/latent.) This is the key property enabling the massively parallel update scheme (below).

**Prototype-vs-product gap.** The **G₁₂ rule set, L≈70, and boundary handling** are from the *paper's simulated* models. Extropic has **not** published the exact interaction graph of the **physical Z-1 part**; the G₈–G₂₄ family shows the rule set is a parameter the shipping chip may differ on.

**Implementation cross-check.** Extropic's own **THRML** library encodes a *simplified* variant in its MNIST example (`tests/test_train_mnist.py` → `get_double_grid` with **axial** jumps `{1,4,15}`, degree 12+1, enforcing bipartiteness with a literal **two-layer split**) — a convenience encoding, **not** the paper's skew-rule geometry. Its example notebooks use only degree-4 von Neumann (toy) or **D-Wave Pegasus** (`examples/02_spin_models.ipynb`, also not the Z-1 graph). The authoritative geometry is **Table 2** of the paper.

## Sampling / update model

Updates run as **block (blocked) Gibbs sampling** over the two color classes of the bipartite graph: "a single iteration of Gibbs sampling corresponds to sampling the first color block conditioned on the second and then vice versa." Because every node within a color class is conditionally independent given the other class, **an entire color-block updates in parallel** — i.e. checkerboard/graph-coloring block Gibbs sampling implemented directly in hardware. The two colors are simply the **`(x+y)` checkerboard of the single grid** — bipartiteness is guaranteed by the odd-parity connection rules (see *Architecture & connectivity*), so no special construction is required and the parallel update is exact. In the paper's experiments roughly **K ≈ 1000** Gibbs iterations are used per denoising step.

The flagship generative algorithm is the **Denoising Thermodynamic Model (DTM)** — a diffusion-like model that **chains together multiple energy-based models**, each of which progressively denoises the data. It is purpose-built so each denoising step maps onto a TSU's native block-Gibbs sampling, which is where the energy advantage is claimed to come from.

## Energy-efficiency claims (read skeptically)

Extropic's headline is: **"Running DTMs on our TSUs could be 10,000× more energy efficient than modern algorithms running on GPUs, as shown by our simulations."** Precisely what that means, from the primary paper:

- **Metric:** energy *per generated sample*.
- **Quality bar:** parity in **Fréchet Inception Distance (FID)** with GPU baselines (single-step VAE and GAN; DDPM at varying diffusion-step counts).
- **Workload:** **Fashion-MNIST** — a small 28×28 grayscale image benchmark.
- **Method:** **simulation / system-level analysis** using an energy model **E = T·K_mix·L²·E_cell** with **E_cell ≈ 2 fJ** per cell update — *not* a measurement from a fabricated Z-1.

Skeptical caveats (some voiced by Extropic, some by independent commentators):
- It is **simulated**, on a **toy benchmark**; "end-to-end proof at meaningful scale is still missing" and third-party measurement has not occurred (per secondary deep-dives).
- Fashion-MNIST does not represent large, diverse production AI workloads; a 10,000× figure on a tiny benchmark may not survive scaling.
- The comparison is generative-sampling-specific (diffusion-like image gen). It is **not** a general claim that TSUs are 10,000× more efficient at all AI tasks.
- The figure bundles algorithm (DTM) + hardware (TSU) advantages together against a conventional algorithm-on-GPU stack; it is not an apples-to-apples same-algorithm comparison.

**Bottom line:** treat "10,000×" as a *projected, simulation-based, narrow-benchmark* figure, not a delivered Z-1 measurement.

## Software stack (THRML)

**THRML** (github.com/extropic-ai/thrml) is Extropic's open-source **JAX** library — described as "a JAX library for building and sampling probabilistic graphical models, with a focus on efficient block Gibbs sampling and energy-based models." It supports sparse, heterogeneous graphical models and arbitrary PyTree node states, provides discrete EBM utilities (Ising-style examples), and is GPU-accelerated. It **doubles as a TSU simulator** and as the place to prototype thermodynamic algorithms today, before Z-1 hardware ships. Release: **v0.1.3, 2025-10-29, Apache-2.0, Python ≥3.10.** The accompanying research paper is *"An efficient probabilistic hardware architecture for diffusion-like models"* (arXiv:2510.23972; authors include Extropic's Guillaume Verdon and Trevor McCourt, with MIT's Isaac Chuang and others). A community DTM replication repo also exists (`pschilliOrange/dtm-replication`).

## Applications

Stated/intended workloads: **generative AI** (diffusion-like image generation via DTMs), **probabilistic inference and sampling**, and **energy-based machine learning** generally. Extropic frames the chips as an answer to AI's energy/data-center crunch. Early development kits (XTR-0) are reported going to **AI labs and weather/climate modeling companies** — a natural fit, since weather is heavily probabilistic/Monte-Carlo.

## Availability / timeline

- **X0:** Q1 2025 (test chip, done).
- **XTR-0:** Q3 2025 (dev platform, in beta with partners).
- **Z-1:** **early access 2026.** Currently in design/build; Extropic describes the moment as "moving from breakthrough to buildout." No production Z-1 silicon has shipped or been independently benchmarked as of this document's date (2026-06-10).

---

## Open questions / unconfirmed

These are genuinely **not publicly known** (or are unverified) as of 2026-06-10 — they are intentionally left blank rather than guessed:

1. **Exact Z-1 grid geometry.** No source gives "512×512" or any explicit grid. The ~512×512 ≈ 262,144 figure in the brief is **a user estimate** that merely lands near Extropic's "~250,000 / hundreds of thousands" language. Extropic has not disclosed rows×columns.
2. **Exact Z-1 pbit count.** "Hundreds of thousands" (Extropic) / "~250,000" (secondary) — no precise official number.
3. **CMOS process node** (e.g. which nm node).
4. **Physical die area / transistors per pbit.** The paper's "6×6 µm for 10⁶ cells" is **almost certainly a typo** (physically impossible at 36 µm²; likely 6×6 mm) — do not rely on it.
5. **Physical Z-1 connectivity.** The exact graph is now precisely known *for the paper's simulated architecture* (**G₁₂ = rules (0,1),(4,1),(9,10)**, Table 2). But Extropic has **not** confirmed the shipping Z-1 uses this specific rule set, degree, L, or boundary handling — the **G₈–G₂₄** family shows it is a tunable knob.
6. **Sampling throughput (samples/s), clock/update rate, latency** for the actual Z-1 part.
7. **Bit-depth/precision** of biases and weights in production silicon.
8. **Host interface details** (PCIe generation/lanes), on-card memory, power envelope (TDP), and per-card TSU count.
9. **Independent, third-party energy measurements** — none exist; all efficiency numbers are Extropic's own simulations on a small benchmark.

---

## Sources

**Primary (Extropic and its authors):**
- [Extropic — Hardware page](https://extropic.ai/hardware) — X0/XTR-0/Z-1 roadmap, "hundreds of thousands per chip, millions per card," early access 2026, primitives.
- [Extropic — Thermodynamic Computing: From Zero to One](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one) — announcement (2025-10-29), 10,000× claim wording, "breakthrough to buildout."
- [Extropic — Inside X0 and XTR-0](https://extropic.ai/writing/inside-x0-and-xtr-0) — XTR-0 = CPU + FPGA + two TSU daughterboard sockets; all-transistor TSUs.
- [Extropic — TSU 101](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware) — conceptual intro (page rendered as template on fetch; listed for completeness).
- [Extropic — Software page](https://extropic.ai/software) — THRML/software stack.
- [arXiv:2510.23972 — "An efficient probabilistic hardware architecture for diffusion-like models"](https://arxiv.org/abs/2510.23972) — Jelinčič, Lockwood, Garlapati, Schillinger, Chuang, Verdon, McCourt. Sparse Boltzmann L×L grids, degree-12, bipartite block Gibbs, E_cell≈2 fJ, τ₀≈100 ns, Fashion-MNIST/FID, "~10⁶ cells in 6×6 µm" (likely typo). **Connectivity specifics:** §III (degree family, bipartite-by-construction); **Appendix D.2 + Table 2** (the connection-rule definition and the exact G₈/G₁₂/G₁₆/G₂₀/G₂₄ offset sets); Fig 9 caption (visible/latent partition + clamped inter-step coupling); Eq. 88. [HTML v1](https://arxiv.org/html/2510.23972v1), [HTML v2](https://arxiv.org/html/2510.23972v2).
- [GitHub — extropic-ai/thrml](https://github.com/extropic-ai/thrml) — JAX library, block Gibbs, EBMs, v0.1.3, Apache-2.0, 2025-10-29. Connectivity cross-check: `tests/test_train_mnist.py` (`get_double_grid`, axial jumps {1,4,15} — a simplified two-layer encoding, *not* the paper's skew-rule geometry); `thrml/models/ising.py`; `examples/02_spin_models.ipynb` (demonstrates D-Wave Pegasus, also not the Z-1 graph).

**Secondary (third-party coverage — corroboration only):**
- [Vastkind — Extropic TSU deep dive](https://www.vastkind.com/extropic-thermodynamic-computing-tsu-deep-dive/) — primitives, roadmap, skeptical framing ("end-to-end proof at meaningful scale is still missing").
- [VKTR — Extropic claims 10,000× energy savings](https://www.vktr.com/ai-news/extropic-claims-10000x-energy-savings-with-new-probabilistic-ai-chip/) — Z1 ~250,000 pbits.
- [The Rundown AI — Extropic's 10,000× breakthrough](https://www.therundown.ai/p/extropics-10-000x-ai-energy-breakthrough) — "Z-1 chip next year," kits to AI labs and weather companies.
- [Geeky Gadgets — Thermodynamic computing / p-bits](https://www.geeky-gadgets.com/thermodynamic-computing-p-bits/) — 10,000× claim, simulation/small-scale caveats.
- [Medium (Cogni Down Under) — Extropic TSU review](https://medium.com/@cognidownunder/extropic-tsu-review-physics-beats-math-and-this-startup-just-proved-it-with-a-chip-that-thinks-in-9082868de469) — overview/analysis.
- [Medium (P. Lourenço) — The 10,000× energy revolution](https://medium.com/@pedromlourenco/the-10-000x-energy-revolution-how-thermodynamic-computing-could-rewrite-the-rules-of-ai-4dc17e413347) — overview.
- [GitHub — pschilliOrange/dtm-replication](https://github.com/pschilliOrange/dtm-replication) — community DTM replication.
