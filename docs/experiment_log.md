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
