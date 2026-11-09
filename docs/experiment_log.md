# Experiment log

One entry per question. Each entry has a hypothesis, the command run, the result and a conclusion.

---

## 2026-11-08 · E001 · Which viscosities does a 256² grid resolve?

**Hypothesis.** The proposal's planned training range (ν from 1e-4 to 1e-3 at a 256² simulation
resolution) is resolved.

**Method.** Decaying turbulence, random initial field with E = 0.5, IF-RK4, CFL 0.5, run to t = 4.
Metric: `resolution_ratio`, the maximum over time of E(k) in the last 10% of resolved shells
divided by peak E(k).

| n | k_peak | ν = 1e-4 | ν = 3e-4 | ν = 1e-3 | ν = 2e-3 |
| --- | --- | --- | --- | --- | --- |
| 256 | 10 | 1.5e-3 | 8.4e-4 | 2.1e-4 | 4.1e-5 |
| 256 | 6 | 2.3e-4 | 9.5e-5 | 1.8e-5 | 4.7e-6 |
| 128 | 4 | 1.2e-3 | 6.4e-4 | 2.1e-4 | 9.0e-5 |

**Result.** The hypothesis is rejected. ν = 1e-4 at 256² shows visible grid-scale noise in the
vorticity field. ν = 1e-3 with k_peak = 6 gives smooth fields and a spectrum that falls about
7 decades before the cutoff.

**Conclusion.** Narrow the M2 dataset to roughly ν ∈ [1e-3, 5e-3] at 256² for training, or
simulate at 512² if the higher-Re end is needed. Re-run this check for the forced (Kolmogorov) setup
in M2, since forcing at k_f = 4 places energy at larger scales than k_peak = 6. Adopt
resolution ratio < 1e-4 as the acceptance threshold for every generated trajectory.

---

## 2026-11-08 · E002 · How much faster is the batched PyTorch solver?

**Hypothesis.** Batching trajectories in PyTorch beats the NumPy solver on CPU, and by much more on a GPU.

**Method.** `python scripts/benchmark_solver.py`, IF-RK4, 20 timed steps, median of 5 repeats after warm-up.
Metric: trajectory-steps per second (batch size × steps / wall time). CPU run on a 2-core cloud VM,
PyTorch 2.15, NumPy 2.5.

| n | NumPy (traj-steps/s) | Torch float64, best batch | Torch float32, best batch |
| --- | --- | --- | --- |
| 64 | 780 | 3,123 (4.0×, batch 16) | 5,975 (7.7×, batch 16) |
| 128 | 339 | 666 (2.0×, batch 16) | 1,468 (4.3×, batch 16) |
| 256 | 88 | 143 (1.6×, batch 16) | 303 (3.5×, batch 16) |

**Result.** Confirmed on CPU: 3–8× in float32, with gains shrinking as n grows (FFT-bound, only 2 cores).
Batch 64 is slightly slower per trajectory than batch 16 on this machine (cache pressure).

**Conclusion.** Use float32 and batches of about 16 on CPU. GPU numbers are still to measure on the
home PC (`python scripts/benchmark_solver.py --out figures/bench_gpu.json`); they decide the M2 data
budget.

---

## 2026-11-08 · E003 · Is Kolmogorov-forced turbulence resolved at 128²?

**Method.** `python scripts/demo_forced.py --n 128 --t-final 40`; k_f = 4, drag 0.1, three viscosities in
one batch, statistics over t ∈ [20, 40].

| ν | Stationary energy (mean ± std) | Max resolution ratio | Visual check |
| --- | --- | --- | --- |
| 1e-3 | 0.93 ± 0.05 | 5.6e-5 | faint grid-scale ripples |
| 2e-3 | 1.05 ± 0.17 | 5.4e-5 | smooth |
| 5e-3 | 0.85 ± 0.06 | 6.9e-6 | smooth |

**Result.** All runs reach a statistically stationary state within t ≈ 5–10. ν = 1e-3 passes the
1e-4 resolution threshold from E001 but still shows ripples, so that threshold is too loose.

**Conclusion.** In M2, simulate at 256² and tighten the acceptance threshold (provisionally 2e-5) after
a 256² scan on the GPU. Use a spin-up of at least t = 10 before recording, and longer windows than
t = 20 for statistics, since the energy fluctuates by ±15% over periods of about 10 time units.

---

## 2026-11-08 · E004 · Grid and time step for the forced dataset; first dataset run

**Question.** Which simulation grid and time step does `forced_v1` need, and does the pipeline's
quality control work in practice?

**Measurements at 256² (Kolmogorov k_f = 4, drag 0.1, dt = 2.5e-3, float32, t ∈ [10, 25]).**

| ν | Max resolution ratio | Max CFL |
| --- | --- | --- |
| 1.5e-3 | 5.0e-8 | 0.54 |
| 1.0e-3 | 1.5e-7 | 0.52 |
| 5.0e-4 | 8.6e-7 | 0.50 |

Forced turbulence at these Re is much better resolved than the decaying case in E001 (which started
from an energetic k_peak = 6 field). 256² resolves the whole planned range, including the high-Re test
set, so no 512² simulations are needed (about 8× cheaper than planned). dt = 2.5e-3 keeps the CFL
number near 0.5.

**Dev dataset (`configs/datasets/forced_dev.yaml`, 128² → 64², 140 trajectories, 2-core CPU).**
About 1 minute per batch of 16 trajectories (~10 minutes in total); 205 MB on disk. All random-IC
splits pass the 2e-5 resolution threshold (max 1.4e-5). The quality check rejected 3 of 12 structured
initial conditions, all "perturbed Taylor–Green". The cause was the initial condition itself: its
random perturbation peaked at k = 8, which leaves a spectral tail near the 128² cutoff. Lowering the
peak to k = 5 fixed it (12/12 accepted).

**Other observations.**
- Frame-to-frame increments have about half the standard deviation of the field itself (2.17 vs
  4.15 at save_dt = 0.25). Each prediction step is a substantial update, not a small correction;
  this matters for residual-prediction models in M3.
- The shear-layer initial conditions roll up into Kelvin–Helmholtz vortices within the recorded
  window, as expected.

**Expected cost of `forced_v1`.** About 2.0e7 trajectory-steps at 256² in total. That would be
about 18 hours at this CPU's ~300 trajectory-steps/s. On the RTX 4090 it should be far less;
the GPU benchmark will give the real figure. About 4.3 GB on disk (train 1.6 GB; test_id 2.2 GB
because it also stores 128² and 256² fields).

---

## 2026-11-09 · E005 · First learned surrogates vs baselines (dev scale, CPU)

**Question.** Do the training and evaluation stack work end to end, and where do a small CNN and
U-Net stand against persistence and the coarse solver?

**Setup.** `forced_dev` (64 training trajectories, 64², save interval 0.25). Small models trained
for 12 epochs on a 2-core CPU, one-step regime, checkpoint chosen by 20-step validation rollout
error: CNN (width 32, depth 4, 75k parameters, ~14 min) and U-Net (widths 16/32/64, 449k
parameters, ~16 min). Evaluated on `test_id` (16 trajectories) over 40 steps (10 time units).
All models and baselines in float32.
Figure: `figures/compare_dev_test_id.png`; table: `figures/compare_dev_test_id.md`.

| Model | 1-step rel. L2 | Mean rel. L2 (40 steps) | High-corr. time (ρ > 0.8) | Small-scale log-spectrum error | ms/step/traj (CPU) |
| --- | --- | --- | --- | --- | --- |
| persistence | 0.506 | 1.28 | 0.23 | 0.02 | 0.2 |
| coarse solver (64²) | **0.077** | **0.40** | **7.50** (12% censored) | 0.05 | **3.7** |
| CNN | 0.453 | 1.11 | 0.28 | 3.96 | 3.8 |
| U-Net | 0.084 | 0.92 | 2.80 | 0.06 | 5.8 |

**Findings.**
1. **Rollout and one-step error disagree.** The CNN's one-step validation error fell from
   0.46 to 0.33 over training, while its 20-step rollout error got *worse* after epoch 1, so
   rollout-based model selection kept the epoch-1 checkpoint. This is the problem the training-regime
   study (M5) targets.
2. **The CNN behaves like a learned diffusion.** Its spectrum collapses at small scales and energy
   accumulates in the largest scale (1.9× the true energy by t = 10). Its receptive field (21 px)
   cannot represent the non-local Poisson coupling.
3. **The U-Net is accurate per step but compounds errors.** Its one-step error is close to the
   coarse solver's and its spectrum is excellent, but it decorrelates 2.7× sooner.
4. **The coarse solver is a strong, fair baseline.** 0.055 of its 0.077 one-step error is the 2/3
   dealiasing floor (it cannot represent stored modes with |k| ≥ 64/3), so beating it on the first
   step partly reflects resolution, not dynamics. That must be stated whenever a model beats it.
5. **Timing has to be fair.** In float64 the solver took 42 ms/step; in float32 it takes 3.7 ms,
   faster than the U-Net on this CPU. Comparing against the float64 number would have produced a
   false 7× speed-up claim.

**Conclusion.** The pipeline works. At dev scale on CPU, neither network beats the coarse solver on
accuracy or speed. Next: full-size models on `forced_v1` on the GPU (`configs/train/*_v1.yaml`),
then the FNO (M4) and the training-regime ablation (M5).

---

## 2026-11-09 · E006 · From-scratch FNO: correctness against `neuraloperator`, and resolution transfer

**Questions.** (1) Is the from-scratch FNO correct? The M4 exit criterion was test error within
about 10% of the `neuraloperator` library on the same data. (2) Does zero-shot resolution transfer
hold up?

**Setup.** As E005: `forced_dev`, 12 CPU epochs, one-step regime, 1 seed, `test_id` (16
trajectories), 40-step rollouts. All FNOs: width 32, 4 layers, about 1.2M parameters. Ours uses
modes = 12 (|ky| ≤ 12, kx < 12); the library uses n_modes = 24, about the same set because its
convention counts total modes per axis. Timings measured on an idle 2-core CPU.

**Unit-level correctness (tests/test_fno.py).** The spectral layer reproduces the analytic Fourier
multiplier for single modes in both weight blocks, including the kx = 0 column where both blocks
contribute. It drops modes above the cutoff, and gives identical output functions for band-limited
inputs on 32² and 64² grids.

**Results at the training resolution (64²).**

| Model | 1-step rel. L2 | Mean rel. L2 (40 steps) | High-corr. time | ms/step/traj |
| --- | --- | --- | --- | --- |
| coarse solver | **0.077** | **0.40** | **7.50** | 3.7 |
| U-Net (E005) | 0.084 | 0.92 | 2.80 | 5.8 |
| FNO, ours (paper architecture) | 0.141 | 1.01 | 1.97 | 3.2 |
| FNO, ours + per-block channel MLP | 0.128 | 0.98 | 2.36 | 4.2 |
| FNO, `neuraloperator` 2.0 | 0.132 | 0.94 | 2.58 | 4.4 |

**Finding 1: correctness.** The paper-architecture FNO was 7% behind the library on one-step error
and 24% on high-correlation time. Inspecting the library's defaults showed three architectural
differences rather than a bug: a per-block channel MLP, a two-layer lifting, and raw (non-periodic)
coordinate channels. Adding the channel MLP to ours (`model.channel_mlp=true`) closed the one-step
gap (0.128 vs 0.132, now 3% better). High-correlation time is within 9% (2.36 vs 2.58), from a
single seed each. Exit criterion met. The remaining rollout difference needs multi-seed runs (M5)
before reading anything into it.

**Finding 2: zero-shot resolution transfer (trained at 64², evaluated on the 128² copy of
`test_id`).**

| Model | 1-step rel. L2 at 64² → 128² | High-corr. time at 64² → 128² | Energy at 128² |
| --- | --- | --- | --- |
| FNO, ours | 0.141 → 0.143 | 1.97 → 1.97 | stable |
| FNO + channel MLP | 0.128 → 0.130 | 2.36 → 2.36 | stable |
| `neuraloperator` FNO | 0.132 → 0.135 | 2.58 → 2.56 | stable |
| U-Net | 0.084 → 0.401 | 2.80 → 0.33 | loses ~75% |
| CNN | 0.453 → 0.483 | 0.28 → 0.27 | grows |

The FNOs transfer without loss of accuracy; the convolutional models do not, because their kernels
are tied to the pixel size. But the FNOs also gain nothing from the finer grid: their spectra fall
below the truth at the new small scales (log-spectrum error ~0.9 for 12 ≤ k < 42), because their
retained modes cannot create them. That matches the caveat in the proposal (section 6.2). "Zero-shot
super-resolution" here means "consistent at a finer grid", not "more accurate". The 128² coarse
solver is exact here only because `forced_dev` was simulated at 128². On `forced_v1` (simulated at
256²) it becomes a real baseline.

**Finding 3: at dev scale the U-Net still leads the learned models** on one-step error and
correlation time, and the coarse solver leads everything, as in E005.

**Timing note.** Two earlier measurements were taken while a training job was using the CPU and
were up to 4× too slow (15 ms instead of 3.2 ms for the FNO). All timings above were re-measured on an
idle machine. In future, benchmark with nothing else running.

**Next.** Full-size models on `forced_v1` on the GPU (`configs/train/{cnn,unet,fno}_v1.yaml`), then
the M5 study with multiple seeds.

---

## 2026-11-09 · E007 · Dry run of the M5 study pipeline (dev data, tiny budgets)

**Purpose.** Prove the study tooling end to end before using GPU time. This is a pipeline
check, not a result. Models trained for 2 epochs of 25 batches and are barely better than
persistence.

**Run.** `configs/study/m5_dev.yaml` on the regenerated `forced_dev` (which now includes a
240-frame `test_long` split): 36 training runs (3 models × 4 regimes × 2 seeds, a 4-point
data-scaling sweep for two models, few-shot fine-tuning with 2 and 4 trajectories), 172
evaluations, both baselines, and the Pareto benchmark. 32 minutes on the 2-core CPU,
0 failures. A second invocation trained and evaluated nothing (resume works).
`python -m nfds.study.aggregate` produced the report, main table, paired regime-effect table and
all six figures (`results/m5_dev/summary/REPORT.md`).

**Problems the dry run caught (fixed).**
1. The long-run spectra included trajectories that had blown up, while the long-run energy ratio
   excluded them, so the CNN's spectrum peaked at 1e8 next to a modest energy bar. All long-run
   statistics are now computed over the same non-failed trajectories, and the figure labels each
   model with its failure rate.
2. The held-out viscosity band in the Reynolds plot was drawn in white, which looked like a gap
   in the data and had an invisible legend entry. It is now hatched.

**What the diagnostics already show (on undertrained models).** The long-run test separates
failure modes clearly: the U-Net piles energy into the smallest scales and 44% of its 60-time-unit
rollouts blow up, while the FNO and CNN decay towards a near-zero-energy state. The real study
will measure how training regimes change this.

---

## 2026-11-09 · E008 · Interactive app: what end-to-end testing caught

**Setup.** FastAPI server (128² reference solver, CPU) with three dev models; the built client
driven by headless Chrome (`scripts/e2e_app.py`): load, play, paint a vortex, push the Reynolds
number to the top of the slider, switch to the coarse-solver baseline, resize to a phone width.

**Problems found and fixed.**
1. Every settings change or brush stroke sends a frame at the *same* simulation time. The client
   treated "time did not increase" as "a new flow started" and wiped the chart history after each
   interaction. Now only time going backwards resets; a same-time frame replaces the last point.
2. At Re = 10,000 the reference solver's own field showed grid-scale ripples: a 128² grid cannot
   resolve that flow (the same limit measured in E001). The app was presenting an unreliable
   "ground truth". The slider now stops at Re = 5,000 (where the 128² reference stays resolved in
   testing), and the server reports the reference's resolution ratio every frame; above 1e-4 the
   reference panel warns that its fine detail is unreliable (relevant with `--ref-n 64`).
3. Smaller fixes: spectrum axis range and tick density, a favicon request that logged a 404, and
   a colour clash (the low-correlation readout used the ochre reserved for "outside the training
   range").

**Performance (2-core CPU).** 6–11 frames per second with a 128² reference and a 64² U-Net; the
reference solver dominates. Each frame is about 16 KB of float16 data.
