"""
Base abstractions for sign optimization.

Provides the SignOptimizer abstract base class and OptimizationResult dataclass
that all optimizer implementations must use. This ensures a uniform interface
for benchmarking and comparing different approaches.

The canonical ``compute_gradient_energy`` lives in ``energy.py`` and is
re-exported here for convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.director import DirectorField
from .energy import compute_gradient_energy


@dataclass
class OptimizationResult:
    """Result container returned by every SignOptimizer.

    Attributes:
        director: Optimized director field with consistent signs.
        initial_energy: Gradient energy before optimization.
        final_energy: Gradient energy after optimization.
        energy_by_layer: Per-layer energy snapshots (if applicable).
        flips_by_layer: Number of sign flips per layer (if applicable).
        total_flips: Total number of voxels whose sign was changed.
        method: Short identifier for the optimizer that produced this result.
        metadata: Extensible dict for optimizer-specific information
            (e.g. ``converged``, ``iterations``, ``history``).
    """

    director: DirectorField
    initial_energy: float
    final_energy: float
    energy_by_layer: List[float] = field(default_factory=list)
    flips_by_layer: List[int] = field(default_factory=list)
    total_flips: int = 0
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def energy_reduction(self) -> float:
        """Absolute energy reduction."""
        return self.initial_energy - self.final_energy

    @property
    def energy_reduction_pct(self) -> float:
        """Percentage energy reduction (0-100)."""
        if self.initial_energy == 0:
            return 0.0
        return 100.0 * self.energy_reduction / self.initial_energy


class SignOptimizer(ABC):
    """Abstract base class for all sign-optimization strategies.

    Every optimizer must implement :meth:`optimize` which accepts a
    ``DirectorField`` with arbitrary signs and returns an
    ``OptimizationResult`` containing the sign-consistent field.

    Example::

        class MyOptimizer(SignOptimizer):
            def optimize(self, director, verbose=False):
                ...
                return OptimizationResult(director=..., ...)
    """

    @abstractmethod
    def optimize(
        self,
        director: DirectorField,
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run the sign-optimization algorithm.

        Args:
            director: Input director field with potentially inconsistent signs.
            verbose: If ``True``, print progress to stdout.

        Returns:
            An ``OptimizationResult`` with the optimized field and statistics.
        """
        ...


__all__ = [
    "SignOptimizer",
    "OptimizationResult",
    "compute_gradient_energy",
]
