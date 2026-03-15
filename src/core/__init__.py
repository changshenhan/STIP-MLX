# Feature-space permutation
from src.core.permutation import Permutation

__all__ = ["Permutation"]

try:
    from src.core.permutation_mlx import PermutationMLX
    __all__.append("PermutationMLX")
except ImportError:
    PermutationMLX = None  # type: ignore[misc, assignment]
