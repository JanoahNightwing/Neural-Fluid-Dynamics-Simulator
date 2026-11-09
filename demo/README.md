# Demo model

`unet_dev_small/best.pt` is a small U-Net (0.45M parameters) trained for 12 CPU epochs on the
development dataset (see `docs/experiment_log.md`, E005). It is here so the interactive app works
straight after cloning:

```bash
pip install -e ".[app]"
python -m nfds.app          # serves trained runs from runs/ plus this demo model
```

It is deliberately weak: it tracks the solver for about 3 time units before decorrelating. Models
trained on the full dataset (`runs/m5_main/...`) replace it in the app's model list automatically.
