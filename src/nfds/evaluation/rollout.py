"""Autoregressive rollouts."""

from __future__ import annotations

import torch

from nfds.evaluation.steppers import Stepper


@torch.no_grad()
def rollout(stepper: Stepper, trajectories: torch.Tensor, nu: torch.Tensor, horizon: int,
            batch_size: int = 64, start: int = 0) -> torch.Tensor:
    """Roll a stepper forward from true initial frames.

    Parameters
    ----------
    trajectories : (N, T, H, W) physical vorticity (only the first n_in frames
        after ``start`` are fed to the stepper)
    nu : (N,) viscosities
    horizon : number of predicted frames

    Returns
    -------
    (N, horizon, H, W) predictions of frames start + n_in ... start + n_in + horizon - 1.
    Once a prediction becomes non-finite it stays non-finite (NaNs propagate), which
    the failure metric relies on.
    """
    n_in = stepper.n_in
    out = []
    for i in range(0, trajectories.shape[0], batch_size):
        window = trajectories[i : i + batch_size, start : start + n_in].clone()
        nu_b = nu[i : i + batch_size]
        preds = []
        for _ in range(horizon):
            nxt = stepper(window, nu_b)
            preds.append(nxt)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)
        out.append(torch.stack(preds, dim=1))
    return torch.cat(out, dim=0)
