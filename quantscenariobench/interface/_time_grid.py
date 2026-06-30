import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class TimeGrid(eqx.Module):
    """An explicit, ordered sequence of simulation time points.

    Carries a 1-D, strictly monotonically increasing array of floats.
    Non-uniform spacing is supported (FR-3, AD-12).  No Market Model or
    Solver Layer may accept a (start, stop, steps) spec in place of a
    TimeGrid instance.
    """

    t: Float[Array, " T"]

    def __init__(self, t: Float[Array, " T"]) -> None:
        t = jnp.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError(
                f"TimeGrid requires a 1-D array of time points; got shape {t.shape}"
            )
        if t.shape[0] < 2:
            raise ValueError(
                "TimeGrid requires at least 2 time points"
            )
        if not bool(jnp.all(jnp.diff(t) > 0)):
            raise ValueError(
                "TimeGrid time points must be strictly monotonically increasing"
            )
        object.__setattr__(self, "t", t)

    def __len__(self) -> int:
        return int(self.t.shape[0])
