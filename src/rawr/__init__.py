"""Right-Answer Wrong-Reason (RAWR) — top-level package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rawr")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "benchmark",
    "prompts",
    "generate",
    "label",
    "activations",
    "analysis",
    "sae_features",
    "report",
]
