| Model | 1-step rel. L2 | Mean rel. L2 (horizon) | High-corr. time (ρ>0.8) | Failures | log-spectrum err. large (k<5) | log-spectrum err. inertial (5<=k<12) | log-spectrum err. small (12<=k<21) | ms/step/traj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse_solver | 0.0767 | 0.404 | 7.50 (12% censored) | 0% | 0.03 | 0.02 | 0.05 | 3.71 |
| persistence | 0.5062 | 1.282 | 0.23 | 0% | 0.05 | 0.02 | 0.02 | 0.20 |
| cnn_dev_small | 0.4527 | 1.107 | 0.28 | 0% | 0.32 | 2.03 | 3.96 | 3.76 |
| unet_dev_small | 0.0843 | 0.922 | 2.80 | 0% | 0.03 | 0.03 | 0.06 | 5.83 |
