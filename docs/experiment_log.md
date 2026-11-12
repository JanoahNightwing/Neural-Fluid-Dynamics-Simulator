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

---

## 2026-11-09 · E009 · Hybrid solver-in-the-loop model (dev scale, CPU)

**Question.** Does putting the reference solver inside the model (coarse 64² solver step plus a
learned CNN correction, trained end to end through the differentiable solver) beat both the pure
networks and the coarse solver? And does it beat the solver at equal cost?

**Setup.** `configs/train/hybrid_dev.yaml`: 25 solver substeps per save interval (dt = 0.01,
CFL about 0.55) and a CNN corrector identical to `cnn_dev_small` (75k parameters), so the
comparison isolates the effect of the solver. The corrector's output layer starts at zero, so the
untrained hybrid is exactly the coarse solver (unit-tested). One-step training for 12 epochs
(19 min), and 4-step unrolled training for 6 epochs (40 min). Evaluated on `test_id` (16
trajectories, 40 steps), timings on an idle CPU.

| Model | 1-step rel. L2 | Mean rel. L2 (40 steps) | High-corr. time | Small-scale log-spectrum error | ms/step/traj |
| --- | --- | --- | --- | --- | --- |
| coarse solver 64² | 0.077 | 0.404 | 7.50 (12% censored) | 0.05 | 4.1 |
| CNN alone (E005) | 0.453 | 1.107 | 0.28 | 3.96 | 3.8 |
| U-Net (E005) | 0.084 | 0.922 | 2.80 | 0.06 | 5.7 |
| **hybrid, one-step** | **0.032** | 0.309 | 8.31 (25% censored) | 0.01 | 8.6 |
| **hybrid, unroll 4** | 0.037 | **0.242** | **≥ 8.86** (38% censored) | 0.01 | 8.9 |

**Findings.**
1. **The solver in the loop transforms the same small CNN.** Alone it is barely better than
   persistence; wrapped around the solver it beats the coarse solver on every accuracy metric.
   Its one-step error (0.032) is below the coarse solver's dealiasing floor (0.055, E005): the
   correction restores the fine scales a 2/3-dealiased 64² solver cannot represent.
2. **Unrolled training trades one-step for rollout accuracy**, as the M5 hypothesis predicts:
   one-step error 0.032 → 0.037, mean 40-step error 0.309 → 0.242, and more trajectories
   that never decorrelate within the horizon.
3. **At equal cost, the hybrid is on the solver's Pareto curve, not below it**
   (`figures/pareto_dev_hybrid.png`). At t = 2.5: the solver at 80² gives 0.064 for 27.5 ms per
   time unit and at 96² gives 0.031 for 41.3 ms; the unrolled hybrid gives 0.053 for 33.9 ms,
   about what the solver would achieve at that cost (≈0.049, interpolated). The pure networks
   are far above the curve. So on this CPU the hybrid is the best *64²* model, but not a better
   use of compute than a finer grid.

**Why, and what to try on the GPU.** The corrector costs more than the solver step it corrects
(~4.6 ms vs ~3.6 ms per step here). The levers are (a) a cheaper corrector, (b) fewer solver
substeps (the CFL number allows larger steps), and (c), the approach of Kochkov et al., running
the hybrid on a *coarser* grid (32² or 48²) and asking the correction to recover 64²-level
accuracy, where the solver alone is cheap but poor. On a GPU the relative costs of convolutions
and FFTs also change. The `hybrid` experiment in `configs/study/m5_main.yaml` (3 seeds, one-step
and unrolled) measures the first two at full scale. The third needs datasets stored on coarser
grids and is the natural next experiment.

## E010 — GPU solver benchmark (RTX 4060, 8 GB) and dataset batch size

**Setup.** `python scripts/benchmark_solver.py --out figures/bench_gpu.json` on Windows 11,
torch 2.15 + CUDA, idle machine. IF-RK4, ν = 1e-3, 20 timed steps after warm-up, median of repeats.
Units: trajectory-steps per second (batch × steps / s); speed-up relative to NumPy at the same grid.

| Grid | NumPy | torch CPU f32 b16 | CUDA f64 b1 | CUDA f64 b16 | CUDA f64 b64 | CUDA f32 b1 | CUDA f32 b16 | CUDA f32 b64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64² | 1,284 | 2,294 | 255 | 10,527 | 27,407 | 751 | 10,726 | 44,804 |
| 128² | 465 | 3,273 | 761 | 5,432 | 5,361 | 715 | 10,991 | 31,763 |
| 256² | 111 | 574 | 631 | 1,313 | 1,037 | 727 | 7,695 | 4,244 |

**Findings.**
1. **Single trajectories are launch-bound**: ~1.4 ms per step at every grid size, so the GPU only
   pays off when batched. This also bounds the app's real-time reference solver (~700 steps/s).
2. **float64 is 2–6× slower** than float32 on this consumer card at 128²–256². Generation stays in
   float32, which matched float64 trajectories to < 1e-5 (M1 tests).
3. **At 256² throughput peaks at batch 16** (7,695 vs 4,244 at batch 64): the 64-trajectory batch
   no longer fits in cache-friendly working sets on an 8 GB card. `forced_v1` now generates with
   `batch_size: 16`; batch size does not change the data (each trajectory is independent).

**Estimate for `forced_v1`.** 4,000 spin-up + 9,900 recorded steps per trajectory; 1,000 train
trajectories ≈ 13.9 M trajectory-steps ≈ 30 min, all splits ≈ 50 min, ~4 GB on disk.

**Measured (generation run).** 2,138 s for train, 2,884 s (48 min) for all splits, matching the
estimate. All 1,472 trajectories accepted. Worst resolution ratio per split: train 3.2e-7,
test_re_high 2.9e-6 (ν = 3e-4, the hardest case, still 7× below the 2e-5 threshold), test_re_low
1.1e-11. Worst CFL 0.93 in test_ic (structured ICs recorded from t = 0, no spin-up), 0.60–0.65
elsewhere. The resolution ratio falls as a clean power law in ν across all splits
(`figures/forced_v1_summary.png`). Note: stored at 64², the test_re_high fields carry filaments at
the grid scale that the training data rarely has; this is part of what that split tests.

## E011 — Training cost on the RTX 4060, and the GPU memory spill

**Setup.** `python scripts/time_training.py` with the v1 configs (batch 64, bf16 autocast) on the
full `forced_v1` data: 40 timed batches per (model, regime) after a warm-up epoch.

| model | regime | ms/batch | peak GB | min/epoch | h/run (50 epochs) |
| --- | --- | ---: | ---: | ---: | ---: |
| cnn | onestep | 428 | 2.34 | 11.3 | 9.4 |
| cnn | unroll4 | 11,308 | **8.96** | 287 | 239 |
| cnn | pushforward4 | 733 | 2.34 | 18.6 | 15.5 |
| unet | onestep | 468 | 3.86 | 12.3 | 10.2 |
| unet | unroll4 | 14,927 | **14.41** | 373 | 311 |
| unet | pushforward4 | 785 | 3.86 | 19.8 | 16.5 |
| fno | onestep | 107 | 1.45 | 2.8 | 2.4 |
| fno | unroll4 | 402 | 4.79 | 10.1 | 8.5 |
| fno | pushforward4 | 218 | 1.46 | 5.5 | 4.6 |
| hybrid | onestep | 249 | 1.27 | 6.6 | 5.5 |
| hybrid | unroll4 | 1,108 | 6.45 | 27.9 | 23.3 |

**Findings.**
1. **CNN and U-Net unroll4 exceeded the 8.6 GB card and did not fail.** On Windows the NVIDIA
   driver by default spills over-allocations into system RAM across PCIe ("CUDA sysmem
   fallback"), so instead of an out-of-memory error training ran 26–32× slower than one-step
   (expected ≈ 4×). FNO unroll4 fits and costs 3.8× its one-step time, as expected.
2. **Fix.** Training now caps PyTorch's allocation at 95% of the card
   (`gpu_memory_fraction`), so an oversize run fails fast instead of spilling, and supports
   gradient accumulation (`optim.micro_batch_size`): the batch of 64 is processed in micro-batches
   whose gradients are summed before the optimiser step. Because the loss is a per-sample mean
   weighted by micro-batch share, the update is identical to the full batch (unit test: gradients
   agree to 1e-9), so the regime ablation keeps one batch size. cuDNN autotuning is now on.
3. **The convolutional models are compute-heavy at this size.** The v1 CNN (20 convolutions of
   96 channels at 64²) is ≈ 13.6 GFLOP per sample forward, ≈ 2.6 TFLOP per training batch, so
   428 ms is ≈ 6 TFLOP/s sustained. The FNO is 4× cheaper per step than either.
4. **The plan as written does not fit this GPU.** Even with the spill fixed, 50 epochs × 56 runs
   is several hundred GPU-hours; the study has to be resized (fewer steps per run and/or smaller
   convolutional models), see the next entry.

**Follow-up measurement (unroll4 with the fix).** CNN 1,543 ms/batch (micro-batch 32, 4.64 GB),
U-Net 2,586 ms (micro-batch 32, 7.31 GB), FNO 401 ms (4.92 GB), hybrid 1,170 ms (6.72 GB). Unrolled
training now costs 3.6–5.5× one-step, as expected, instead of 26–32×.

## E012 — Sizing the M5 study for one RTX 4060 (~48 GPU-hours)

**Budget.** About 48 GPU-hours, run in sessions (resumable). Electricity is negligible (~200 W at
the wall, ~10 kWh); the constraint is wall-clock time.

**Decisions.**
1. **Fixed training length in steps, not passes over the data:** every run trains 5,000 steps of
   batch 64 (10 "epochs" of 500 steps; `optim.steps_per_epoch`). Consecutive frames are strongly
   correlated, so a full pass (≈1,550 batches) is mostly redundant; 5,000 steps is ≈3.3 passes.
   Fixing steps also makes the data-scaling experiment fair: a 50-trajectory run used to get 77
   steps per epoch and a 1,000-trajectory run 1,546; now all get the same number of updates.
   The dev runs were still improving at the end (1.4k steps of batch 32), so this is ≈7× more
   training samples than any dev result.
2. **The CNN is resized to width 64, depth 8**, the same size as the hybrid model's corrector.
   It was the most expensive model per sample after the U-Net and the weakest in E005; at the
   corrector's size, "CNN" vs "hybrid" becomes a controlled comparison of the same network with
   and without the solver in the loop. Its cost is estimated from the hybrid's (≈0.4× the v1 CNN)
   and will be checked against the first run's epoch time.
3. **U-Net and FNO keep their v1 sizes.** The U-Net unroll4 run is the single most expensive item
   (≈3.6 h); it stays, because the regime ablation needs every regime for every model.
4. **Three seeds for the main comparison**, one for scaling and few-shot, as planned.
5. **Staged execution**, so that every stage ends with a usable result and a bug costs at most
   one stage: stage 1 = all 14 seed-0 runs of the regime and hybrid experiments plus baselines
   (the complete main table, ≈12 h); stage 2 = seeds 1–2 (≈24 h); stage 3 = scaling, few-shot and
   the Pareto benchmark (≈5 h).

**Estimate (seconds per batch × 5,000 steps, plus validation and evaluation).** Per seed, the 14
runs cost ≈7.8 s per step in total (U-Net 4.3, hybrid 1.4, CNN ≈1.25, FNO 0.83), i.e. ≈10.8 h of
training, ≈0.4 h of validation and ≈1 h of evaluation: ≈12 h per seed, ≈41 h in all with stage 3.
