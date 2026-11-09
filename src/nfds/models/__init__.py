"""Neural surrogates. Every model maps (frames, nu_feature) -> next frame, in normalised units."""

from nfds.models.cnn import ResidualCNN
from nfds.models.fno import FNO
from nfds.models.unet import UNet

MODELS = {"cnn": ResidualCNN, "unet": UNet, "fno": FNO}


def build_model(name: str, n_in: int = 1, **kwargs):
    if name == "neuralop_fno":  # optional reference implementation, imported lazily
        from nfds.models.neuralop_adapter import NeuralopFNO

        return NeuralopFNO(n_in=n_in, **kwargs)
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)} + neuralop_fno")
    return MODELS[name](n_in=n_in, **kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["FNO", "MODELS", "ResidualCNN", "UNet", "build_model", "count_parameters"]
