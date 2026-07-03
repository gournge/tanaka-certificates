# Usage

Run these examples from the repository root after `uv sync --dev`.

## Simulate an SDE

The built-in Euler--Maruyama solver accepts either one scalar initial state or
a NumPy-compatible batch of scalar states. A seed makes an experiment
reproducible.

```python
from tanaka_certificates.sde import EulerMaruyama, OrnsteinUhlenbeck1D

sde = OrnsteinUhlenbeck1D(
    mean_reversion=1.0,
    volatility=0.5,
    long_term_mean=0.0,
)
times, states = EulerMaruyama().simulate(
    sde,
    x0=[-1.0, 0.0, 1.0],
    T=2.0,
    n_steps=1_000,
    seed=42,
)

assert times.shape == (1_001,)
assert states.shape == (1_001, 3)
```

Custom models implement `SDE.drift(t, x)` and `SDE.diffusion(t, x)`. These
methods should use NumPy-style arithmetic so they work for both scalar and
batched states.

## Construct a piecewise-linear certificate

`create_1d_certificate_given_breakpoints` constructs a frozen PyTorch
`Certificate` that interpolates the supplied points. The two slope arguments
control linear extrapolation beyond the outermost breakpoints.

```python
import numpy as np
import torch

from tanaka_certificates.facet import Breakpoint
from tanaka_certificates.nn import create_1d_certificate_given_breakpoints

certificate = create_1d_certificate_given_breakpoints(
    breakpoints=[
        Breakpoint(np.array([-0.5]), np.array([0.5])),
        Breakpoint(np.array([0.0]), np.array([0.25])),
        Breakpoint(np.array([0.5]), np.array([0.5])),
    ],
    leftmost_slope=-1.0,
    rightmost_slope=1.0,
)

x = torch.tensor([[-1.0], [0.0], [1.0]])
values = certificate(x)
torch.testing.assert_close(values, torch.tensor([[1.0], [0.25], [1.0]]))
```

Breakpoints may be passed in any order, but their coordinates must be unique.
The factory uses PyTorch's current default dtype, so set it before construction
if an experiment requires a different precision.

## Verify a reach-avoid certificate

The following complete example verifies `V(x) = |x|` outside the target for
the Ornstein--Uhlenbeck process `dX_t = -X_t dt + dW_t`.

```python
import numpy as np

from tanaka_certificates.facet import Breakpoint
from tanaka_certificates.nn import create_1d_certificate_given_breakpoints
from tanaka_certificates.ra import ReachAvoidProblem1D
from tanaka_certificates.regions import Interval, IntervalUnion
from tanaka_certificates.sde import OrnsteinUhlenbeck1D
from tanaka_certificates.verifier import (
    VerificationResult,
    Verifier1DPiecewiseLinear,
)

certificate = create_1d_certificate_given_breakpoints(
    [
        Breakpoint(np.array([-0.5]), np.array([0.5])),
        Breakpoint(np.array([0.0]), np.array([0.25])),
        Breakpoint(np.array([0.5]), np.array([0.5])),
    ],
    leftmost_slope=-1.0,
    rightmost_slope=1.0,
)

problem = ReachAvoidProblem1D(
    domain=IntervalUnion([Interval(-2.0, 2.0)]),
    initial=IntervalUnion([Interval(0.0, 0.0)]),
    unsafe=IntervalUnion([
        Interval(-2.0, -2.0),
        Interval(2.0, 2.0),
    ]),
    target=IntervalUnion([Interval(-1.0, 1.0)]),
    alpha=0.5,
    beta=2.0,
    epsilon=1.0,
)

result = Verifier1DPiecewiseLinear(
    sde=OrnsteinUhlenbeck1D(),
    reach_avoid_problem=problem,
    certificate=certificate,
).verify()

assert result is VerificationResult.VERIFIED
```

Treat `NOT_VERIFIED` as “no proof was established,” not as a counterexample.
Unsupported inputs and malformed certificate networks fail closed to this
result.

## Store experiment outputs

`ResultArtifact` creates a directory whose name includes the current Git
revision and timestamp, and keeps track of paths reserved by a computation.
This is mainly used as a convenient way to track experiments.

```python
from tanaka_certificates import ResultArtifact

artifact = ResultArtifact.create("ou_paths")
figure_path = artifact.path("figures/paths.pdf")
# fig.savefig(figure_path)
```

By default this writes below `output/`. Pass an `output_root` as the second
argument to place artifacts elsewhere.
