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

$$c_1(\phi, t_1, t_2) = \text{offset} + \text{contrast} \times \exp\left(-\tfrac{1}{2} q^2 \int_{t_1}^{t_2} J(t')\ dt'\right) \times \cos\left(q \cos(\phi) \int_{t_1}^{t_2} v(t')\ dt'\right)$$

$$J(t) = D_{0} \cdot t^{\alpha} + D_{\text{offset}} \qquad v(t) = v_0 \cdot \sigma(t; t_0) + v_{\text{offset}}$$

The two-component model tracks reference and sample independently, each with their own
diffusion coefficient, anomalous exponent, onset time, and width parameter. All time integrals
are evaluated numerically via cumulative trapezoid on the discrete time grid.

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| D0_ref, D0_sample | Diffusion coefficients | 1e4 | Å²/s^α |
| alpha_ref, alpha_sample | Anomalous exponents | 0.0 | — |
| D_offset_ref, D_offset_sample | Diffusion offsets | 0.0 | Å² |
| v0 | Velocity amplitude | 1e3 | Å/s |
| v_offset | Velocity offset | 0.0 | Å/s |
| t0_ref, t0_sample | Onset times | varies | s |
| sigma_ref, sigma_sample | Width parameters | varies | s |
| q_power_ref, q_power_sample | q-dependence exponents | 2.0 | — |

Per-angle contrast and offset (2 scaling parameters) are added automatically based on the
number of azimuthal angles. Total: 14 physics + 2 scaling parameters per angle.

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
from heterodyne.optimization import fit_nlsq_jax
from heterodyne.optimization.cmc import fit_cmc_jax
from heterodyne.data import load_xpcs_data
from heterodyne.config import ConfigManager

data = load_xpcs_data("config.yaml")
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

| Command | Purpose |
|---------|---------|
| `heterodyne` | Run XPCS analysis (NLSQ/CMC) |
| `heterodyne-config` | Generate and validate config files |
| `heterodyne-config-xla` | Configure XLA device settings |
| `heterodyne-validate` | System validation |
| `heterodyne-post-install` | Install shell completion (bash/zsh/fish) |
| `heterodyne-cleanup` | Remove shell completion files |

Shell completion and aliases are available after running `heterodyne-post-install --interactive`.

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
