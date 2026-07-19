# Tanaka certificates

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue?style=flat-square)
![uv](https://img.shields.io/badge/managed%20with-uv-2b0231?style=flat-square)
![Tests](https://img.shields.io/badge/tests-pytest-informational?style=flat-square)
<!-- ![Coverage](https://img.shields.io/badge/coverage-local%20only-lightgrey?style=flat-square) -->

This repository explores extending the framework of Neural Continuous-Time Supermartingale Certificates [1] to the case of *piecewise* twice continously-differentiable certificates, rather than twice continously-differentiable certificates. This requires using the Itô-Tanaka-Meyer formula, which accounts for *kinks* and the infinitesimal time spent at these kinks.

It is supervised by Grigory Neustroev, and is a part of Filip Morawiec's 2026 Summer Research Programme at the University of Birmingham in the topic of AI Safety, within the lab of Prof. Mirco Giacobbe.

You can find the research log and weekly reports in the [`docs/research/pdf/`](docs/research/pdf/) directory.

Development documentation can be accessed through the [`docs/dev/index.md`](docs/dev/index.md) file.

## Project structure

```
.
├── .github/
│   └── workflows/             # GitHub Actions configuration
├── docs/
│   ├── dev/                    # Development notes and verifier documentation
│   │   ├── img/                # Images used by development docs
│   │   └── verifier_pwq_2d/    # Piecewise-quadratic verifier notes
│   └── research/
│       ├── log/                # Research-log TeX sources
│       │   ├── figures/        # Figures used by the research log
│       │   └── sections/       # Research-log section TeX files
│       ├── pdf/                # Rendered research log and weekly reports
│       └── weekly-reports/     # Weekly-report TeX sources and images
├── output/                     # Generated experiment artifacts
├── scripts/
│   ├── research/               # Research-log plot generation scripts
│   └── plot_*.py               # Plotting and visualization scripts
├── tanaka_certificates/
│   ├── nn/                     # Neural-certificate training utilities
│   ├── sde/                    # Stochastic differential equation models
│   ├── verifier/               # Multidimensional PWQ certificate verifiers
│   ├── artifacts.py            # Experiment artifact path helpers
│   ├── cell_discovery.py       # Cell discovery for piecewise networks
│   ├── certificate.py          # Certificate abstractions
│   ├── generator_supremum.py   # Generator supremum optimization helpers
│   ├── problems.py             # Problem definitions
│   ├── ra.py                   # Reach-avoid specification helpers
│   └── regions.py              # Region and geometry utilities
├── tests/
│   ├── nn/                     # Neural-certificate tests
│   ├── verifier/               # Verifier tests
│   └── test_*.py               # Unit and integration tests
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Development

Install the Python environment and run the test suite with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
uv run pytest
```

Plotting and experiment scripts write results to named directories under
`output/`, including the timestamp and Git revision.

### Compiling TeX files

Install a TeX distribution that includes `latexmk` and `listings`. On Debian
or Ubuntu, these are provided by:

```bash
sudo apt install latexmk texlive-latex-recommended
```

Compile the research log, poster, and every weekly report from the repository
root with the shared script:

```bash
./scripts/compile_latex.sh
```

To compile only selected document groups, pass one or more of `log`, `poster`,
or `weekly`, for example:

```bash
./scripts/compile_latex.sh log poster
```

The research log and weekly reports are written under `docs/research/pdf/`;
the poster is written to `docs/research/poster/poster.pdf`.

## References

[1] - Neustroev, Grigory, et al. “Neural Continuous-Time Supermartingale Certificates.” Proceedings of the AAAI Conference on Artificial Intelligence, edited by , vol. 39, no. 26, Apr. 2025, pp. 27538–46. Crossref, https://doi.org/10.1609/aaai.v39i26.34966.
