from typing import Any

import equinox as eqx


class MarketModel(eqx.Module):
    """Abstract base class for all Market Models.

    Every concrete subclass must implement:
      _drift(self, t, state) -> PyTree
      _diffusion(self, t, state) -> PyTree

    These are called exclusively by the Solver Layer (AD-4); they are
    not part of the public API surface.
    """

    def __check_init__(self) -> None:
        if type(self) is MarketModel:
            raise TypeError(
                "MarketModel is abstract and cannot be instantiated directly"
            )
        for name in ("_drift", "_diffusion"):
            if getattr(type(self), name) is getattr(MarketModel, name):
                raise TypeError(
                    f"Can't instantiate abstract class '{type(self).__name__}': "
                    f"'{name}' is not implemented"
                )

    def _drift(self, t: Any, state: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _drift(self, t, state)"
        )

    def _diffusion(self, t: Any, state: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _diffusion(self, t, state)"
        )

    def initial_state(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement initial_state() "
            "or pass y0 explicitly to simulate()"
        )
