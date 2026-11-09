"""Wrapper around the ``neuraloperator`` library's FNO, used only as a correctness reference
for our own implementation. Same interface as our models: residual on the last frame, the
normalised log-viscosity as an extra constant input channel."""

from __future__ import annotations

import torch
from torch import nn


class NeuralopFNO(nn.Module):
    def __init__(self, n_in: int = 1, width: int = 32, n_modes: int = 24, n_layers: int = 4,
                 **kwargs) -> None:
        super().__init__()
        try:
            from neuralop.models import FNO as LibFNO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install neuraloperator to use the reference FNO") from e
        self.n_in = n_in
        self.net = LibFNO(n_modes=(n_modes, n_modes), hidden_channels=width,
                          in_channels=n_in + 1, out_channels=1, n_layers=n_layers, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # neuraloperator models add a "_metadata" entry to their state dict that strict
        # loading of the wrapper would reject.
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("_metadata")}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, nu_feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        nu = nu_feat.view(-1, 1, 1, 1).to(x.dtype).expand(b, 1, h, w)
        return x[:, -1:] + self.net(torch.cat([x, nu], dim=1))
