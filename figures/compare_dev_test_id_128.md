| Model | 1-step rel. L2 | Mean rel. L2 (horizon) | High-corr. time (ρ>0.8) | Failures | log-spectrum err. large (k<5) | log-spectrum err. inertial (5<=k<12) | log-spectrum err. small (12<=k<42) | ms/step/traj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse_solver | 0.0000 | 0.001 | 10.00 (100% censored) | 0% | 0.00 | 0.00 | 0.00 | 26.15 |
| persistence | 0.5072 | 1.282 | 0.23 | 0% | 0.05 | 0.02 | 0.03 | 1.58 |
| cnn_dev_small | 0.4826 | 1.358 | 0.27 | 0% | 0.41 | 0.23 | 2.08 | 18.22 |
| fno_dev_small | 0.1427 | 1.014 | 1.97 | 0% | 0.03 | 0.02 | 0.96 | 16.69 |
| fno_mlp_dev_small | 0.1302 | 0.982 | 2.36 | 0% | 0.06 | 0.08 | 0.89 | 21.05 |
| neuralop_fno_dev_small | 0.1347 | 0.936 | 2.56 | 0% | 0.03 | 0.02 | 0.94 | 21.62 |
| unet_dev_small | 0.4006 | 1.279 | 0.33 | 0% | 0.97 | 0.43 | 0.78 | 20.30 |
