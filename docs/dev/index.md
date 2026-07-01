# Developer documentation

The package provides small building blocks for experiments with one-dimensional
stochastic differential equations and piecewise-linear Tanaka certificates.

- [Usage](usage.md) shows how to simulate an SDE, build a certificate, and run
  the verifier.
- [Key abstractions](abstractions.md) explains how the main objects fit
  together and records the verifier's current support boundary.

From the repository root, install the development environment and run the test
suite with:

```bash
uv sync --dev
uv run pytest
```
