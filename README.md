# Heterodyne

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

CPU-optimized JAX package for heterodyne X-ray Photon Correlation Spectroscopy (XPCS) analysis
under nonequilibrium conditions, implementing the theoretical framework from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) and
[He et al. PNAS 2025](https://doi.org/10.1073/pnas.2514216122) for characterizing
transport properties in flowing soft matter systems.

## Heterodyne Model

The package implements a two-component scattering model (PNAS 2025 SI Eqs. S-77–S-98)
where scattered light from a moving **sample** interferes with that from a static
**reference**, producing oscillations in the cross-correlation whose frequency encodes
the sample velocity.

### Two-Time Correlation (Eq. S-95)

$$c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f^2} \left[ C_{\text{ref}} + C_{\text{sample}} + C_{\text{cross}} \right]$$

with three physical contributions:

$$C_{\text{ref}} = [x_r(t_1) \cdot x_r(t_2)]^2 \exp\left(-q^2 \int_{t_1}^{t_2} J_r(t') dt'\right)$$

$$C_{\text{sample}} = [x_s(t_1) \cdot x_s(t_2)]^2 \exp\left(-q^2 \int_{t_1}^{t_2} J_s(t') dt'\right)$$

$$C_{\text{cross}} = 2 x_r(t_1) x_r(t_2) x_s(t_1) x_s(t_2) \exp\left(-\frac{1}{2} q^2 \int_{t_1}^{t_2} [J_s(t') + J_r(t')] dt'\right) \cos\left[q \cos(\varphi) \int_{t_1}^{t_2} \mathbb{E}[v] dt'\right]$$

$$f^2 = [x_s(t_1)^2 + x_r(t_1)^2][x_s(t_2)^2 + x_r(t_2)^2]$$

where $x_s(t)$ is the sample fraction, $x_r(t) = 1 - x_s(t)$ the reference fraction,
$\beta$ the optical contrast, $\varphi$ the angle between velocity and $\vec{q}$, and
$f^2$ a normalization ensuring $c_2(t, t) = 1 + \beta$ on the diagonal.

### One-Time Equilibrium Form (Eq. S-98)

When all parameters are time-independent and $J_n(t) = 6D_n$ (Wiener process), the
two-time correlation reduces to a function of lag time $\tau = t_2 - t_1$ only:

$$g_2(\vec{q}, \tau) = 1 + \beta \left[ (1 - x)^2 e^{-6q^2 D_r \tau} + x^2 e^{-6q^2 D_s \tau} + 2x(1 - x) e^{-3q^2(D_r + D_s)\tau} \cos[q \cos(\varphi) \mathbb{E}[v] \tau] \right]$$

where $x = x_s^2 / (x_s^2 + x_r^2)$ is the equilibrium sample intensity fraction.

### Fitting Model

The implementation wraps the correlation with per-angle scaling parameters:

$$c_2^{\text{model}} = \text{offset} + \text{contrast} \times \frac{C_{\text{ref}} + C_{\text{sample}} + C_{\text{cross}}}{f^2}$$

### Rate Functions

Each component has its own power-law transport coefficient, and the sample has an
additional velocity rate:

$$J(t) = D_0 t^\alpha + D_{\text{offset}} \qquad v(t) = v_0 t^\beta + v_{\text{offset}}$$

All time integrals are evaluated **numerically** via cumulative trapezoid — no analytical
antiderivatives are ever used, ensuring correctness for the general power-law form.

### Parameters

The model has 14 physics parameters organized into five groups, plus 2 per-angle scaling
parameters:

**Reference transport** — $J_r(t) = D_{0,r} t^{\alpha_r} + D_{\text{offset},r}$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `D0_ref` | Reference diffusion prefactor | 1e4 | Å²/s<sup>α+1</sup> |
| `alpha_ref` | Reference transport exponent | 0.0 | — |
| `D_offset_ref` | Reference transport rate offset | 0.0 | Å²/s |

**Sample transport** — $J_s(t) = D_{0,s} t^{\alpha_s} + D_{\text{offset},s}$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `D0_sample` | Sample diffusion prefactor | 1e4 | Å²/s<sup>α+1</sup> |
| `alpha_sample` | Sample transport exponent | 0.0 | — |
| `D_offset_sample` | Sample transport rate offset | 0.0 | Å²/s |

**Velocity** — $v(t) = v_0 t^\beta + v_{\text{offset}}$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `v0` | Velocity prefactor | 1e3 | Å/s<sup>β+1</sup> |
| `beta` | Velocity exponent (0 = constant velocity) | 0.0 | — |
| `v_offset` | Velocity offset (negative for reversal) | 0.0 | Å/s |

**Sample fraction** — $f_s(t) = f_0 \exp(f_1 (t - f_2)) + f_3$, where $f_s(t) \in [0, 1]$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `f0` | Fraction amplitude | 0.5 | — |
| `f1` | Exponential rate (0 = constant fraction) | 0.0 | s⁻¹ |
| `f2` | Time shift | 0.0 | s |
| `f3` | Baseline offset | 0.0 | — |

**Flow angle**

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `phi0` | Flow angle offset relative to q-vector | 0.0 | degrees |

**Per-angle scaling** (2 parameters per detector angle)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `contrast` | Optical contrast β (speckle contrast) | 0.5 |
| `offset` | Baseline offset | 1.0 |

Total: **14 physics + 2 scaling parameters per angle**.

## Installation

```bash
pip install heterodyne
```

For development:

```bash
git clone https://github.com/imewei/heterodyne.git
cd heterodyne
make dev    # or: uv sync
```

**Requirements:** Python 3.12+, CPU-only (no GPU). Runs on Linux, macOS, and Windows.

## Quick Start

### CLI

```bash
# Generate a config template
heterodyne-config --output config.yaml

# Run NLSQ optimization
heterodyne --method nlsq --config config.yaml

# Run Consensus Monte Carlo for uncertainty quantification
heterodyne --method cmc --config config.yaml
```

### Python API

```python
from heterodyne.data import load_xpcs_data
from heterodyne.config import ConfigManager
from heterodyne.optimization import fit_nlsq_jax
from heterodyne.optimization.cmc import fit_cmc_jax

# Load data and config
data = load_xpcs_data("experiment.hdf5")
config = ConfigManager("config.yaml")

# NLSQ trust-region optimization
nlsq_result = fit_nlsq_jax(data, config)

# CMC with NLSQ warm-start for Bayesian uncertainty
cmc_result = fit_cmc_jax(data, config, nlsq_result=nlsq_result)
```

### Data Flow

```
YAML config --> XPCSDataLoader(HDF5) --> HeterodyneModel --> NLSQ or CMC --> Results (JSON + NPZ)
```

## Optimization Methods

**NLSQ** (primary) -- JAX-native trust-region Levenberg-Marquardt with automatic
anti-degeneracy defense, CMA-ES global search for multi-scale problems, and memory-aware
routing for large datasets.

**CMC** (secondary) -- Consensus Monte Carlo using NumPyro NUTS sampling with automatic
sharding, NLSQ warm-start priors, and multiprocessing across CPU cores. Produces
publication-quality posterior distributions with ArviZ diagnostics.

## Configuration

Heterodyne uses YAML configuration files. Generate a template:

```bash
heterodyne-config --output config.yaml
```

Key sections:

```yaml
experimental_data:
  file_path: "data.h5"
optimization:
  method: "nlsq"
  nlsq:
    anti_degeneracy:
      per_angle_mode: "auto"   # auto, constant, individual, fourier
  cmc:
    sharding:
      max_points_per_shard: "auto"
```

## CLI Commands

| Command | Short alias | Purpose |
|---------|-------------|---------|
| `heterodyne` | `ht` | Run XPCS analysis (NLSQ/CMC) |
| `heterodyne-config` | `ht-config` | Generate and validate config files |
| `heterodyne-config-xla` | `ht-config-xla` | Configure XLA device settings |
| `heterodyne-validate` | `ht-validate` | System validation |
| `heterodyne-post-install` | `ht-post-install` | Install shell completion (bash/zsh/fish) |
| `heterodyne-cleanup` | `ht-cleanup` | Remove shell completion files |
| `hexp` | — | Plot experimental data (skip optimization) |
| `hsim` | — | Plot simulated C2 heatmaps from config |

### Plotting Shortcuts

```bash
# Inspect experimental data before fitting
hexp --config config.yaml

# Preview simulated C2 heatmaps from current parameters
hsim --config config.yaml --contrast 0.3 --offset-sim 1.0
```

Shell completion and additional aliases (e.g. `het`, `het-nlsq`, `het-cmc`) are
available after running `heterodyne-post-install --interactive`.

## Development

```bash
make test       # All tests
make test-fast  # Exclude slow tests
make quality    # Format + lint + type-check + shellcheck
make verify     # Full local CI verification before pushing
```

## Documentation

- [Changelog](CHANGELOG.md)

## Citation

If you use Heterodyne in your research, please cite:

```bibtex
@article{He2024,
  author  = {He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Transport coefficient approach for characterizing nonequilibrium
             dynamics in soft matter},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {31},
  year    = {2024},
  doi     = {10.1073/pnas.2401162121}
}
```

```bibtex
@article{He2025,
  author  = {He, Hongrui and Liang, Heyi and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Bridging microscopic dynamics and rheology in the yielding
             of charged colloidal suspensions},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {122},
  number  = {42},
  year    = {2025},
  doi     = {10.1073/pnas.2514216122}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Authors

- Wei Chen (weichen@anl.gov) -- Argonne National Laboratory
- Hongrui He (hhe@anl.gov) -- Argonne National Laboratory
