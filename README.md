# Tanaka certificates

This repository explores extending the framework of Neural Continuous-Time Supermartingale Certificates [1] to the case of *piecewise* twice continously-differentiable certificates, rather than twice continously-differentiable certificates. This requires using the Itô-Tanaka-Meyer formula, which accounts for *kinks* and the infinitesimal time spent at these kinks.

It is supervised by Grigory Neustroev, and is a part of Filip Morawiec's 2026 Summer Research Programme at the University of Birmingham in the topic of AI Safety, within the lab of Prof. Mirco Giacobbe.

You can find the research log and weekly reports in the [`docs/research/pdf/`](docs/research/pdf/) directory.

Development documentation can be accessed through the [`docs/dev/index.md`](docs/dev/index.md) file.

## Project structure

```
.
├── docs
│   ├── dev
│   │   └── index.md
│   └── research
│       ├── log
│       │   ├── figures
│       │   └── sections
│       ├── pdf
│       │   ├── research-log.pdf
│       │   └── weekly-reports
│       └── weekly-reports
├── pyproject.toml
├── README.md
├── scripts
│   ├── __init__.py
│   └── research/
├── tanaka_certificates
│   ├── artifacts.py
│   ├── certificate.py
│   ├── facet.py
│   ├── __init__.py
│   ├── nn
│   │   └── __init__.py
│   ├── ra.py
│   ├── regions.py
│   ├── sde
│   │   ├── base.py
│   │   ├── constant.py
│   │   ├── __init__.py
│   │   └── ornstein_uhlenbeck.py
│   └── verifier.py
├── tests
│   ├── test_artifacts.py
│   ├── test_nn.py
│   ├── test_sde.py
│   ├── test_verifier_ornstein_uhlenbeck.py
│   └── test_verifier.py
└── uv.lock
```

## Development

Install the Python environment and generate the one-dimensional research plots
with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
uv run python -m scripts.research.generate_one_dim_plots
```

The plotting command writes each result to a named directory containing its
timestamp and Git revision, such as
`output/a1b2c3d_2026-07-01_14-30-00_ou_generator_three_cases/`. Pass
`--output PATH` to use another artifact root. Run the test suite with
`uv run pytest`.

### Compiling TeX files

Install a TeX distribution that includes `latexmk`, then run the following
commands from the repository root.

To compile the research log:

```bash
mkdir -p docs/research/pdf
latexmk -pdf -cd -outdir=../pdf -jobname=research-log docs/research/log/main.tex
```

To compile a weekly report:

```bash
mkdir -p docs/research/pdf/weekly-reports
latexmk -pdf -cd -outdir=../pdf/weekly-reports docs/research/weekly-reports/2026-07-01.tex
```

These commands place rendered documents in `docs/research/pdf/`, keeping
generated LaTeX build files out of the source directories.

## References

[1] - Neustroev, Grigory, et al. “Neural Continuous-Time Supermartingale Certificates.” Proceedings of the AAAI Conference on Artificial Intelligence, edited by , vol. 39, no. 26, Apr. 2025, pp. 27538–46. Crossref, https://doi.org/10.1609/aaai.v39i26.34966.
