# QuantScenarioBench

A JAX-native Python framework for generating reproducible stochastic market scenarios and publishing them as Hugging Face benchmark datasets.

```python
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import Heston
import jax.numpy as jnp

model = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
tg    = TimeGrid(jnp.linspace(0.0, 1.0, 253))   # 252 daily steps, 1 year

scenario = simulate(model, tg, n_paths=10_000, seed=42)

print(scenario.observation.shape)    # (10000, 253)  — asset price paths
print(scenario.latent_state.shape)   # (10000, 253)  — variance paths
```

---

## Why QuantScenarioBench?

Quantitative finance research routinely benchmarks models against each other, but reproducible, openly published path datasets are rare. QuantScenarioBench makes it straightforward to:

- **Generate** large batches of price paths from established stochastic volatility models with a single call
- **Compare** models on a shared schema — every `Scenario` has the same fields regardless of the model used
- **Export** results to Parquet or publish directly to the Hugging Face Hub
- **Reproduce** any result exactly — the same `(model, time_grid, n_paths, seed)` always produces bit-identical paths on the same computational backend

---

## Installation

```bash
pip install quantscenariobench
```

**Requirements:** Python ≥ 3.11, JAX ≥ 0.4.38.

For optional Hugging Face publishing:

```bash
pip install "quantscenariobench[dev]"   # includes datasets, pandas
pip install huggingface_hub             # for publish_to_hub
```

---

## Quick Start

### 1. Generate a scenario

```python
import quantscenariobench   # enables JAX float64 globally
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston, RoughBergomi
import jax.numpy as jnp

tg = TimeGrid(jnp.linspace(0.0, 1.0, 253))   # daily steps over 1 year

# Black-Scholes (no latent state)
bs = BlackScholes(mu=0.0, sigma=0.2, S0=100.0)
s  = simulate(bs, tg, n_paths=50_000, seed=0)
print(s.observation.shape)    # (50000, 253)
print(s.latent_state.shape)   # (50000, 0)   — empty for GBM

# Heston stochastic volatility
heston = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
s      = simulate(heston, tg, n_paths=50_000, seed=0)
print(s.observation.shape)    # (50000, 253)  — asset price paths
print(s.latent_state.shape)   # (50000, 253)  — variance paths

# Rough Bergomi (non-Markovian, Hurst H=0.1)
rb = RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0)
s  = simulate(rb, tg, n_paths=50_000, seed=0)
```

### 2. Export to Parquet

```python
from quantscenariobench.export import export_parquet

export_parquet([s], "my_dataset.parquet")
```

Every exported file follows the same 12-column schema regardless of model:

| Column | Type | Description |
|--------|------|-------------|
| `observation` | `list<float64>` | Asset price path (one row per path) |
| `latent_state` | `list<float64>` | Latent state path (empty for Black-Scholes) |
| `seed` | `int64` | Integer PRNG seed |
| `prng_key_info` | `string` | JAX PRNGKey derivation description |
| `model_name` | `string` | Market model class name |
| `model_version` | `string` | Model specification version |
| `parameters` | `string` | JSON-encoded model parameters |
| `time_grid` | `string` | JSON-encoded time points |
| `n_paths` | `int64` | Number of simulation paths |
| `library_version` | `string` | `quantscenariobench` version |
| `dataset_version` | `string` | Dataset version identifier |
| `generated_at` | `string` | UTC ISO-8601 generation timestamp |

### 3. Publish to Hugging Face Hub

```python
from quantscenariobench.export import publish_to_hub

url = publish_to_hub([s], "my-org/my-dataset", token="hf_...")
print(url)   # https://huggingface.co/datasets/my-org/my-dataset
```

`publish_to_hub` writes the Parquet file and a dataset card (README.md) in a single call, then uploads both to the Hub.

### 4. Replay paths deterministically

```python
scenario, dW = simulate(heston, tg, n_paths=1000, seed=7, return_randomness=True)

# Later, reproduce the exact same paths from the saved increments
replayed = simulate(heston, tg, n_paths=1000, seed=7, randomness=dW)
```

---

## Available Models

### Black-Scholes (`BlackScholes`)

Geometric Brownian Motion:

$$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t$$

| Parameter | Description |
|-----------|-------------|
| `mu` | Drift (use `0.0` for risk-neutral) |
| `sigma` | Constant volatility |
| `S0` | Initial asset price |

### Heston (`Heston`)

Stochastic volatility with mean-reverting variance:

$$dS_t = \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\,\sqrt{v_t}\,dW^v_t, \quad \text{Corr}(dW^S, dW^v) = \rho$$

| Parameter | Description |
|-----------|-------------|
| `mu` | Asset drift |
| `kappa` | Variance mean-reversion speed |
| `theta` | Long-run variance |
| `xi` | Vol-of-vol |
| `rho` | Asset–variance correlation (leverage effect) |
| `v0` | Initial variance |
| `S0` | Initial asset price |

The Feller condition `2κθ ≥ ξ²` ensures the variance process stays positive. Violation emits a `QuantScenarioBenchValidationWarning`.

### Rough Bergomi (`RoughBergomi`)

Non-Markovian stochastic volatility driven by fractional Brownian motion:

$$V_t = \xi_0 \exp\!\left(\eta\,W^H_t - \tfrac{1}{2}\eta^2 t^{2H}\right), \quad dS_t = \mu\,S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t$$

where $W^H_t$ is a Riemann–Liouville fractional Brownian motion with Hurst exponent $H$, discretised via the Volterra representation.

| Parameter | Description |
|-----------|-------------|
| `H` | Hurst exponent; `H < 0.5` for rough volatility (empirically `H ≈ 0.1`) |
| `eta` | Vol-of-vol amplitude |
| `rho` | Correlation between asset BM and fBM driver |
| `xi0` | Initial variance level |
| `S0` | Initial asset price |
| `mu` | Asset drift |

---

## Reproducibility

Identical `(model, time_grid, n_paths, seed)` inputs produce **bit-identical paths** on the same computational backend (CPU / GPU / TPU). Cross-backend bit-identity is not guaranteed due to floating-point differences across JAX backends. The `seed`, `prng_key_info`, and `library_version` metadata fields document full provenance for every batch.

All simulations run in **float64** (enabled automatically on `import quantscenariobench`).

---

## Pre-built Benchmark Datasets

Three v1 benchmark datasets are published on Hugging Face for immediate use:

| Model | Dataset | Paths | Steps |
|-------|---------|-------|-------|
| Black-Scholes | [tim-nish/qsb-black-scholes](https://huggingface.co/datasets/tim-nish/qsb-black-scholes) | 10,000 | 253 (daily, 1 yr) |
| Heston | [tim-nish/qsb-heston](https://huggingface.co/datasets/tim-nish/qsb-heston) | 10,000 | 253 (daily, 1 yr) |
| Rough Bergomi | [tim-nish/qsb-rough-bergomi](https://huggingface.co/datasets/tim-nish/qsb-rough-bergomi) | 10,000 | 253 (daily, 1 yr) |

All three use the same time grid (`linspace(0, 1, 253)`), seed (`42`), and initial spot (`S0=100`, `mu=0`) for direct cross-model comparison.

Load any dataset with:

```python
from datasets import load_dataset

ds = load_dataset("tim-nish/qsb-heston", split="train")
print(ds.column_names)
# ['observation', 'latent_state', 'seed', 'prng_key_info', 'model_name',
#  'model_version', 'parameters', 'time_grid', 'n_paths',
#  'library_version', 'dataset_version', 'generated_at']
```

These datasets are representative samples. To generate a dataset at any scale or with custom parameters, use `simulate()` and `export_parquet()` or `publish_to_hub()` directly.

---

## Architecture

```
quantscenariobench/
├── api/           simulate() — single public entry point
├── interface/     MarketModel ABC, Scenario, TimeGrid, Metadata
├── models/        BlackScholes, Heston, RoughBergomi
├── solver/        Euler-Maruyama SDE solver (diffrax / lineax)
├── export/        export_parquet(), generate_dataset_card(), publish_to_hub()
└── testing/       Conformance suite for custom model authors
```

**Dependency rule:** `models` and `export` import only from `interface`. The solver and API layers compose these. This keeps any custom model implementation minimal.

### Implementing a custom model

Subclass `MarketModel` (an `equinox.Module`) and implement three methods:

```python
import equinox as eqx
import jax.numpy as jnp
from quantscenariobench.interface import MarketModel

class MyModel(MarketModel):
    sigma: float
    S0: float

    def _drift(self, t, state):
        return jnp.zeros_like(state)   # zero drift

    def _diffusion(self, t, state):
        return self.sigma * state      # scalar diffusion

    def initial_state(self):
        return jnp.array(self.S0, dtype=float)
```

Pass the model directly to `simulate()` — no other integration required.

---

## Development

```bash
git clone https://github.com/tim-nish/QuantScenarioBench
cd QuantScenarioBench
pip install -e ".[dev]"
pytest
```

The test suite covers closed-form price validation (Gil-Pélaez inversion for Heston; Black-Scholes formula for GBM), statistical properties (skew monotonicity in H for rBergomi), Parquet round-trips, and dataset card conformance.

---

## License

MIT
