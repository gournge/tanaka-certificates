# Key abstractions

The verification path is a composition of four objects:

```text
SDE + ReachAvoidProblem1D + Certificate1D
                    |
                    v
       Verifier1DPiecewiseLinear
                    |
                    v
           VerificationResult
```

## Stochastic dynamics

`SDE` defines the drift and diffusion coefficients of

```text
dX_t = drift(t, X_t) dt + diffusion(t, X_t) dW_t.
```

`BrownianMotion`, `ConstantCoefficients`, and `OrnsteinUhlenbeck` are supplied
models. `EulerMaruyama` is a numerical experiment tool: it simulates one path
per initial state, but does not participate in formal verification.

## Regions and reach-avoid problems

`Interval` is a closed interval `[lower, upper]`; point sets are represented by
equal endpoints. `IntervalUnion` groups intervals and provides membership,
intersection, and difference operations. It does not currently normalize,
sort, or merge overlapping intervals.

`ReachAvoidProblem1D` combines four such regions with three bounds:

- `domain`: the state space on which generator and kink conditions are checked;
- `initial`: where `V <= alpha` must hold;
- `unsafe`: where `V >= beta` must hold;
- `target`: excluded from the generator and kink checks;
- `epsilon`: the required generator decrease, `L V <= -epsilon`.

For the normal reach-avoid path, choose `alpha < beta` and `epsilon >= 0`.

## Certificates and breakpoints

`Certificate` is a semantic subclass of `torch.nn.Sequential`.
`Certificate1D` marks the scalar case. The breakpoint factory returns a frozen
`Certificate` made only from `Linear` and `ReLU` modules. `Breakpoint` stores
one coordinate and its certificate value; exterior slopes are supplied
separately to the factory.

The verifier does not rely on the factory: it discovers the exact affine
pieces of any compatible scalar-input, scalar-output `Linear`/`ReLU` sequential
network and merges adjacent pieces that describe the same affine function.

## Verification contract

`Verifier1DPiecewiseLinear.verify()` checks:

1. the unsafe and initial value bounds;
2. `L V <= -epsilon` on the part of `{V <= beta}` inside the domain and outside
   the target;
3. a non-positive slope jump at every relevant kink (the Tanaka concavity
   condition).

For a piecewise-linear certificate, `V'' = 0` away from kinks, so the smooth
generator reduces to `L V = V' * drift`. Diffusion enters the underlying
stochastic process but not this smooth-piece calculation; kink effects are
handled by the slope-jump condition.

Current verification support is intentionally narrow:

- one-dimensional, scalar-input/scalar-output certificates;
- sequential networks containing only `torch.nn.Linear` and `torch.nn.ReLU`;
- affine drift, detected from its values at interval endpoints and midpoint;
- finite domain intervals for generator checks.

The supplied constant-coefficient and Ornstein--Uhlenbeck SDEs satisfy the
affine-drift requirement. A successful call returns
`VerificationResult.VERIFIED`. A violated condition, unsupported network or
dynamics, non-finite generator interval, or invalid nonnegative `epsilon`
requirement returns `VerificationResult.NOT_VERIFIED`.
