from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class BasketMetadata:
    """Basket-level provenance record attached to a correlated basket (FR-47, AD-36).

    Additive alongside the N unchanged per-asset Scenario/Metadata records
    simulate_correlated_basket() returns — this carries the cross-asset
    information no single Scenario's own Metadata can express: the
    correlation matrix used, the shared basket seed, and each
    constituent's identity (mirroring Metadata's model_name/model_version
    fields rather than inventing a new identifier scheme).
    """

    rho: Any  # Float[Array, "N N"] — the validated correlation matrix
    basket_seed: int
    constituents: list  # list[dict] — one {"model_name", "model_version"} per asset
