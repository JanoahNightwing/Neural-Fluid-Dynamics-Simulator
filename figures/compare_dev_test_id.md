| Model | 1-step rel. L2 | Mean rel. L2 (horizon) | High-corr. time (ρ>0.8) | Failures | log-spectrum err. large (k<5) | log-spectrum err. inertial (5<=k<12) | log-spectrum err. small (12<=k<21) | ms/step/traj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse_solver | 0.0767 | 0.404 | 7.50 (12% censored) | 0% | 0.03 | 0.02 | 0.05 | 4.14 |
| persistence | 0.5062 | 1.282 | 0.23 | 0% | 0.05 | 0.02 | 0.02 | 0.20 |
| cnn_dev_small | 0.4527 | 1.107 | 0.28 | 0% | 0.32 | 2.03 | 3.96 | 3.76 |
| fno_dev_small | 0.1407 | 1.014 | 1.97 | 0% | 0.03 | 0.02 | 0.37 | 3.24 |
| fno_mlp_dev_small | 0.1281 | 0.983 | 2.36 | 0% | 0.06 | 0.07 | 0.35 | 4.18 |
| hybrid_dev_onestep | 0.0324 | 0.309 | 8.31 (25% censored) | 0% | 0.02 | 0.01 | 0.01 | 8.61 |
| hybrid_dev_unroll4 | 0.0370 | 0.242 | 8.86 (38% censored) | 0% | 0.01 | 0.01 | 0.01 | 8.86 |
| neuralop_fno_dev_small | 0.1317 | 0.935 | 2.58 | 0% | 0.03 | 0.02 | 0.33 | 4.43 |
| unet_dev_small | 0.0843 | 0.922 | 2.80 | 0% | 0.03 | 0.03 | 0.06 | 5.68 |
