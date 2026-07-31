# Itô–Tanaka certificates

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue?style=flat-square)
![uv](https://img.shields.io/badge/managed%20with-uv-2b0231?style=flat-square)
[![Tests](https://github.com/gournge/tanaka-certificates/actions/workflows/tests.yml/badge.svg)](https://github.com/gournge/tanaka-certificates/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/gournge/tanaka-certificates?style=flat-square)](https://codecov.io/gh/gournge/tanaka-certificates)

This repository explores extending the framework of Neural Continuous-Time Supermartingale Certificates [1] to the case of piecewise twice continously-differentiable certificates, rather than twice continously-differentiable certificates. This requires using the Itô-Tanaka-Meyer formula, which accounts for kinks and the infinitesimal time spent at these kinks.

It is supervised by Grigory Neustroev, and is a part of Filip Morawiec's 2026 Summer Research Programme at the University of Birmingham in the topic of AI Safety, within the lab of Prof. Mirco Giacobbe.


## Results

<!-- plot it with centering as a medium size (it's a square) -->
<!-- docs/research/poster/verifier-guided-poisson-teacher.png -->
<div style="text-align:center">
  <img src="docs/research/poster/verifier-guided-poisson-teacher.png" width="400px" />
</div>

The main experiment fits a piecewise-quadratic certificate for a
two-dimensional Ornstein–Uhlenbeck reach–avoid problem, as shown above. 
That problem's setup is $\alpha=1.97$, $\beta=2$, and $\epsilon=0.1$, so it gives a 
pretty loose (but valid) bound:

$\mathbb P(\text{unsafe set or domain exit before target}) \leq 0.985.$

The repository also contains:

- a multidimensional Itô–Tanaka certificate theorem;
- piecewise-quadratic cell discovery and verification;
- a residual ICNN architecture whose local-time sign is safe by construction;
- training experiments, counterexamples, research notes, and tests.

> **Note:** Although I used AI tools for programming, I wrote the key tests and went 
> through the math and code myself.


## Project links

- [Project overview and write-up](https://filipmorawiec.com/tanaka)
- [A0 poster](docs/research/poster/poster.pdf)
- [Research log](docs/research/pdf/research-log.pdf)
- [Latest weekly report](docs/research/pdf/weekly-reports/2026-07-20.pdf)
- [All rendered research documents](docs/research/pdf/README.md)
- [Development notes](docs/dev/index.md)

## Quick start

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked development environment:

```bash
git clone https://github.com/gournge/tanaka-certificates.git
cd tanaka-certificates
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

## Reproduce the poster experiment

Run the verifier-guided Poisson-teacher experiment with its default fixed
configuration:

```bash
uv run python scripts/train_icnn_poisson_teacher.py --method verifier-lp
```

The command fits the certificate, verifies it, and writes a checkpoint,
diagnostic figure, and `verification.log` to a revision-stamped directory under
`output/`. The log records the verification status, number of cells, certificate
parameters, and numerical diagnostics.

Other useful entry points include:

```bash
# Controlled construction-safe OU benchmark
uv run python scripts/train_verified_radial_ou_certificate.py

# Ideal martingale/trajectory illustration used in the poster
uv run python scripts/plot_intro_martingale_committor.py
```

## Repository guide

- `tanaka_certificates/verifier/` — exact regional and numerical PWQ verifiers;
- `tanaka_certificates/cell_discovery.py` — conservative activation-cell discovery;
- `tanaka_certificates/nn/` — certificate architectures and training utilities;
- `scripts/` — reproducible experiments and plotting entry points;
- `tests/` — unit, integration, and verifier regression tests;
- `docs/research/` — poster, research log, and weekly reports;
- `docs/dev/` — implementation and verifier notes.

Experiment scripts write their outputs to ignored, revision-stamped directories
under `output/`.

## Compiling the research documents

Install a TeX distribution containing `latexmk` and `listings`. On Debian or
Ubuntu:

```bash
sudo apt install latexmk texlive-latex-recommended
```

Compile the poster, research log, and weekly reports from the repository root:

```bash
./scripts/compile_latex.sh
```

To compile selected groups only, pass `log`, `poster`, or `weekly`:

```bash
./scripts/compile_latex.sh log poster
```

## Reference

G. Neustroev et al., “Neural Continuous-Time Supermartingale Certificates,”
*Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39,
no. 26, pp. 27538–27546, 2025.
[doi:10.1609/aaai.v39i26.34966](https://doi.org/10.1609/aaai.v39i26.34966)
