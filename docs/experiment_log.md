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
