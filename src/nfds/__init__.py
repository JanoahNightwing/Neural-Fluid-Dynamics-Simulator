"""Neural Fluid Dynamics Simulator."""

from nfds.grid import SpectralGrid
from nfds.solver.spectral_numpy import SpectralVorticitySolver

__all__ = ["SpectralGrid", "SpectralVorticitySolver"]
__version__ = "0.5.0"
