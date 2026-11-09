# GPU runbook (Windows + NVIDIA GPU)

Everything needed to produce the project's real results on a machine with an NVIDIA GPU
(written for an RTX 4090 on Windows; Linux commands are the same with `/` instead of `\`
and `source .venv/bin/activate`). Run each step in PowerShell from the repository folder
with the virtual environment active. Every long step can be interrupted and resumed.

## 0. One-time setup

1. Install the latest NVIDIA driver (NVIDIA app), Python 3.12 from python.org (tick
   "Add Python to PATH") and Git for Windows. Check the GPU: `nvidia-smi`.
2. Get the code and create the environment:

   ```powershell
   cd $HOME
   git clone https://github.com/JanoahNightwing/nfds.git
   cd nfds
   py -m venv .venv
   .venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

3. Install PyTorch **with CUDA**. On Windows, plain `pip install torch` gives a CPU-only
   build. Use the command from https://pytorch.org/get-started/locally (Stable, Windows,
   Pip, Python, newest CUDA), then:

   ```powershell
   pip install -e ".[dev,data]"
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   pytest -q
   ```

   Expect `True <your GPU>` and all tests passing (more than on a laptop: the GPU tests run).

## 1. Benchmark the solver (2 minutes)

```powershell
python scripts\benchmark_solver.py --out figures\bench_gpu.json
```

Keep the printed table; it decides how long the next steps take.

## 2. Generate the full dataset (~4.3 GB)

```powershell
python -m nfds.data.generate configs\datasets\forced_v1.yaml --out data\forced_v1
python -m nfds.data.inspect data\forced_v1
```

The generator prints progress with an estimate of the remaining time. Afterwards check the
summary line per split: every split should show all (or nearly all) trajectories accepted.
Open `data\forced_v1\summary.png` and `data\forced_v1\DATASHEET.md`.

## 3. A quick check of the study (15-30 minutes)

Run one seed of the one-step regime first, to see real timings before committing to the
full study:

```powershell
python -m nfds.study.run configs\study\m5_main.yaml --only "regimes-*-onestep-*-s0"
python -m nfds.study.run configs\study\m5_main.yaml --list
```

Look at `runs\m5_main\regimes-unet-onestep-nall-s0\metrics.jsonl`: the training loss should
fall, and the validation rollout error should be well below 1.

## 4. The full study (overnight)

```powershell
python -m nfds.study.run configs\study\m5_main.yaml
```

This trains 50 runs and evaluates every run and both baselines on every test set. It is safe
to stop (Ctrl+C, reboot) and run the same command again: finished work is skipped. If a run
fails, it is recorded in `results\m5_main\failures.json` and the study moves on.

To stop Windows from sleeping during the run: Settings > System > Power > Screen and sleep >
"When plugged in, put my device to sleep after: Never" (restore it afterwards).

## 5. Aggregate

```powershell
python -m nfds.study.aggregate configs\study\m5_main.yaml
```

Produces `results\m5_main\summary\REPORT.md` with the main table, the regime-effect table and
all figures.

## 6. Share the results

Commit the small result files (JSON, tables, figures; checkpoints and data stay local):

```powershell
git add results figures
git commit -m "Add M5 study results"
git push
```

Git for Windows will open a browser to sign in to GitHub the first time.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is False | You installed the CPU build: reinstall with the CUDA command from pytorch.org |
| CUDA out of memory | Add a lower batch size to every run: `python -m nfds.study.run configs\study\m5_main.yaml optim.batch_size=32` |
| A run fails repeatedly | See `results\m5_main\failures.json`; the rest of the study still runs |
| Very slow training | Check `nvidia-smi` shows Python using the GPU, and that nothing else heavy is running |
