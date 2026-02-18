# Assignment 1 — Plan (Sampler Synthesis)

This plan assumes you start from `assignment-01-starter.ipynb` and keep it as the **main runnable artifact**. The notebook already contains:
- Target log densities for **Rosenbrock** and **Neal’s Funnel**
- Baseline **RWMH** and **HMC**
- Diagnostics helpers using **ArviZ**

Goal: add **one novel sampler** + clear baselines + diagnostics + 3-page report.

---

## 0) What we will build

### Primary novel method (recommended)
**GG-NUTS: Geometry-Gated NUTS**
- Two NUTS kernels that differ only in `max_depth`:
  - **Stable**: shallow trees (`D_s`)
  - **Explore**: deeper trees (`D_l`)
- Choose which one to run at each iteration based on a *geometry score* at current state:
  \[
  s(q)=\sqrt{\nabla U(q)^\top M\,\nabla U(q)}
  \]
  \[
  w(q)=\sigma(a(s(q)-b))
  \]
- With prob `w(q)` run Stable-NUTS else Explore-NUTS.

**Why this is safe:** you are not modifying NUTS internals; you are composing two already-correct kernels.

### Optional secondary method (only if time remains)
**EE-MS-HMC: Energy-Error–Adaptive Multi-Scale HMC**
- Use a short probe to estimate local energy error; pick between (ε,L) pairs.

---

## 1) Repository structure (keep simple)

### Keep the notebook as the main deliverable
✅ Recommended for DS595: submit the notebook as the executable artifact with plots + diagnostics.

### Add a small `src/` folder for clean code
This keeps your notebook readable, while still showing “real” engineering.

Create:
```
src/
  __init__.py
  gg_nuts.py        # GG-NUTS wrapper + helpers
  diagnostics.py    # (optional) ESS per grad-eval helper
```

**Why not only notebooks?**
- Notebooks are fine, but small python modules help readability and reduce copy/paste bugs.
- Your report will also be easier to write if your method is in a clean function.

---

## 2) Implementation checklist (in order)

### 2.1 Confirm baselines run (already in starter)
In `assignment-01-starter.ipynb`:
- Run Rosenbrock baselines (RWMH, HMC)
- Run Funnel baselines (RWMH, HMC) — there is a TODO cell for this

Deliverable baseline outputs:
- Sample scatter over contour plot
- Trace plots
- Autocorrelation plot
- ESS (ArviZ)

### 2.2 Add BlackJAX NUTS baseline
Add one cell that runs **vanilla NUTS** (single `max_depth`) on:
- Rosenbrock
- Funnel

Record:
- `acceptance_rate` (or equivalent stat)
- number of gradient evaluations (if available) or proxy via steps/tree depth

### 2.3 Implement GG-NUTS (novel sampler)

#### Math you will implement
- Let `log_prob_fn(q)` be your target log density.
- Define `U(q) = -log_prob_fn(q)` and `g(q)=∇U(q) = -∇log_prob_fn(q)`.

Geometry score (Option B):
- Choose a fixed mass matrix `M` (start with identity; optionally use diagonal from warmup).
- \[
  s(q)=\sqrt{g(q)^\top M\,g(q)}
  \]
For diagonal `M=diag(m)`:
- `s(q) = sqrt(sum(m * g**2))`

Gate:
- `w(q)=sigmoid(a*(s(q)-b))`
- Sample `bernoulli(w)` using JAX RNG
- If 1 → run NUTS with `max_depth=D_s`
- Else → run NUTS with `max_depth=D_l`

#### How to choose `b` quickly (no overfitting)
- Do a short pilot run (200–500 iters) with vanilla NUTS
- Compute `s(q)` along the chain
- Set `b = median(s(q))`

Hyperparameters to start:
- `D_s=6`, `D_l=9`
- `a=2` (or 5 if you want sharper switching)

### 2.4 Add ablation (makes it “paper-like”)
Run a **fixed-mixture** version:
- Choose Stable vs Explore with constant probability `p` (no geometry)
Compare:
- fixed-mixture vs GG-NUTS
This isolates the effect of the geometry gate.

---

## 3) Experiments you must produce (minimal but strong)

For each benchmark (Rosenbrock and Funnel):
1) **RWMH baseline**
2) **HMC baseline**
3) **Vanilla NUTS baseline**
4) **Fixed-mixture NUTS** (ablation)
5) **GG-NUTS** (your method)

Diagnostics (per assignment):
- Acceptance rate
- ESS (ArviZ)
- Autocorrelation
- Visuals of samples vs target
- Traces

**Add one “compute fairness” metric (recommended):**
- ESS per gradient evaluation (or ESS per leapfrog step)
This is important because NUTS depth changes computation.

---

## 4) Notebook workflow (where to put what)

In `assignment-01-starter.ipynb`, add sections:

1) **Run Funnel baselines** (fill the TODO)
2) **NUTS baseline** (new section)
3) **GG-NUTS implementation** (import from `src/gg_nuts.py`)
4) **GG-NUTS results: Rosenbrock**
5) **GG-NUTS results: Funnel**
6) **Ablation: fixed-mixture vs gated-mixture**
7) **Hyperparameter sweep (small)**
   - try `a ∈ {2,5}`
   - try `(D_s,D_l) ∈ {(5,8),(6,9)}`
8) **Summary table**
   - acceptance
   - ESS
   - ESS/grad-eval (if you compute it)

Keep results reproducible by fixing PRNG seeds.

---

## 5) What to write in the report (3 pages max)

### Introduction (~0.5 page)
- Problem: heterogeneous geometry (banana valley, funnel neck)
- Why fixed-budget sampling can waste compute / struggle
- Key idea: adapt exploration budget by local stiffness score

### Method (~0.5 page)
- Define target π(q), potential U(q)
- Define two NUTS kernels K_s, K_l
- Define score s(q) and gate w(q)
- State final transition kernel: mixture of K_s and K_l
- Brief invariance argument: each kernel targets π; mixture preserves π

### Experiments (~1 page)
- Show both benchmarks
- Include plots + diagnostics
- Compare against MH, HMC, NUTS
- Include ablation

### Discussion (~0.5 page)
- Where it helps (funnel neck stability / compute efficiency)
- Where it hurts (unnecessary switching, sensitivity to a/b, etc.)
- Next steps (use energy error instead of gradient norm; learn threshold; use warmup mass matrix)

### AI Collaboration (~0.5 page)
- What you used AI for (design, correctness check, implementation help)
- What you corrected
- Failure modes you considered (e.g., breaking detailed balance if you changed ε inside trajectories)
- What you learned

---

## 6) Concrete deliverables checklist (what you submit)

- ✅ `assignment-01-starter.ipynb` updated and runnable end-to-end
- ✅ `src/gg_nuts.py` (and optional helpers)
- ✅ `report/` filled with your 3-page NeurIPS-style PDF
- ✅ (Optional) `plan.md` (this file) for clarity

---

## 7) Time budget (15 hours)

- 1–2h: get starter notebook running + fill Funnel baseline TODO
- 1h: add vanilla NUTS baseline
- 1h: implement GG-NUTS wrapper + pilot threshold `b`
- 3–5h: run experiments + tune (small sweeps)
- 3–5h: plots cleanup + write report + AI-collaboration section

---

## 8) “Don’t mess this up” rules

- Do **not** modify NUTS internals.
- Do **not** change step size *inside* a leapfrog trajectory (for HMC variants).
- Always compute diagnostics on **post burn-in** samples.
- Always compare with a compute-aware metric for NUTS depth changes.

