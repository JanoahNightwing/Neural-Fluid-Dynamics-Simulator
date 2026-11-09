"""Neural surrogates. Every model maps (frames, nu_feature) -> next frame, in normalised units."""

from nfds.models.cnn import ResidualCNN
from nfds.models.unet import UNet

MODELS = {"cnn": ResidualCNN, "unet": UNet}


def build_model(name: str, n_in: int = 1, **kwargs):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](n_in=n_in, **kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["MODELS", "ResidualCNN", "UNet", "build_model", "count_parameters"]
