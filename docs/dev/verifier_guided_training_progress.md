# Verifier-guided PWQ training: implementation progress

Status at 2026-07-04. Work was deliberately stopped before completing the
alpha-grid acceptance regression.

## Goal

Train the default two-dimensional Ornstein--Uhlenbeck reach--avoid certificate
against deterministic regional extrema and counterexamples from the exact PWQ
verifier. The intended acceptance criterion is that at least one alpha in
`(0.5, 0.75, 1.0, 1.25, 1.5)` returns `VerificationResult.VERIFIED`.

## Implemented changes

### Training

`tanaka_certificates/nn/train_certificate.py` now includes:

- every corner of every initial, target, and unsafe rectangle in every epoch;
- worst-case hinge losses for upper and lower regional bounds;
- a 50-epoch boundary pretraining phase;
- exact-verifier calls at a configurable interval (default 10 epochs);
- bounded, condition-specific pools for regional and generator counterexamples;
- differentiable generator re-evaluation at retained counterexamples;
- differentiable one-sided normal-gradient losses for invalid faces;
- face-pool refresh after each verifier call because activation faces move;
- early stopping only on exact `VERIFIED`;
- artifact metadata for verifier status, issue counts, final loss components,
  and completed epochs.

The network's terminal ReLU was removed. A `final_linear_has_relu` flag was
added to the certificate/cell-discovery path so the final scalar affine map is
handled correctly.

An optional `enforce_face_concavity` mode (currently enabled by default)
initializes and projects the final scalar weights to be nonpositive. Since the
PWQ top is monotone, this makes every hidden-ReLU normal derivative jump
nonpositive by construction. The differentiable face loss remains as a
guardrail.

### Exact verifier and cell discovery

- `VerificationIssue` now carries the oriented face normal needed by training.
- Exact face verification reports all invalid faces rather than only one
  global worst face.
- Discovered cells retain the scalar affine feature and activation interval.
- The exact verifier uses this metadata to solve `V <= beta` as scalar
  quadratic intervals and clip cell polygons directly. Generic adaptive
  subdivision remains as a fail-closed fallback.

### Shared defaults and diagnostics

- `make_ou_problem(alpha=...)` accepts alpha explicitly.
- `make_default_training_configuration(...)` is the shared configuration
  constructor intended for both plotting and regression tests.
- Plot metrics include training status, issue counts, and final loss terms.
- Numeric logs use three significant digits.

## Measurements so far

Before verifier-guided training, the default alpha `0.5` candidate failed with
approximately:

- target maximum `0.634 > 0.5`;
- unsafe minimum `1.538 < 2.0`;
- generator maximum `1.12 > -0.1`;
- concavity jump `1.79 > 0`.

With counterexample training but the original terminal ReLU, alpha `1.5`
still failed after 400 epochs:

- unsafe minimum `1.832 < 2.0`;
- generator maximum `0.700 > -0.1`;
- concavity jump `1.005 > 0`.

After removing the terminal ReLU and enforcing nonpositive scalar output
weights, exact concavity failures disappeared. At alpha `1.5`, 400 epochs then
ended with:

- initial maximum `1.523 > 1.5`;
- unsafe minimum `1.634 < 2.0`;
- generator maximum `0.695 > -0.1`;
- no concavity issue.

This isolates the remaining problem to regional expressivity/optimization and
the generator condition. The stronger concavity-by-construction architecture
may be too restrictive for the two-obstacle geometry.

## Performance findings

Exact verification every 10 epochs makes a 400-epoch alpha run take multiple
minutes. The scalar-feature sublevel optimization substantially reduces final
verification cost, but rebuilding and checking all faces remains expensive.
A five-alpha regression should not run in the ordinary fast test suite without
further caching or an explicit slow-test marker.

Update after indexed planar face discovery:

- shared-face verification on the measured 69-cell network fell from 2.97 s
  to approximately 0.009 s;
- cell discovery remained approximately 0.32 s at width 8;
- incremental activation-pattern pruning reduced discovery scaling to about
  0.78 s at width 12 and 1.57 s at width 16 on the development machine;
- a default 400-epoch width-8 run now takes approximately 22 s rather than
  multiple minutes.

The verifier now reports one exact generator maximizer per violating cell,
and training retains the best exactly checked iterate. The generator loss also
models the actual disjunction ``V > beta or L V <= -epsilon`` instead of
detaching the current basin predicate.

These changes have not yet produced an accepted candidate. Widths 8, 12, and
16 converge to a similar tradeoff at alpha 1.5: the unsafe minimum remains
roughly 0.4--0.47 below beta while the largest generator violation remains
roughly 0.4--0.58. This makes additional width or verifier frequency a poor
next experiment; the scalar PWQ bottleneck or the optimization formulation is
now the likely limitation.


## Tests and current confidence

Focused tests passed after the terminal-ReLU/cell-discovery changes:

- cell discovery;
- structural exact-verifier checks;
- certificate training smoke/reproducibility checks;
- plotting smoke test.

A full suite was not rerun after the final CEGIS and scalar-feature changes.
The interrupted width-12, boundary-only experiment did not complete and
produced no result.

## Recommended continuation

1. Run the focused tests, then the full suite, to establish the current
   checkpoint.
2. Add unit tests that directly show one optimization step decreases a fixed
   generator counterexample loss and a fixed face-jump loss.
3. Decide whether to retain global nonpositive output-weight projection:
   - first run a boundary-only width-12 experiment at alpha `1.5`;
   - if regional separation remains impossible, disable projection and rely on
     the all-face counterexample loss;
   - compare exact concavity and regional margins after equal training budgets.
4. Increase regional and generator weights only after the architecture
   comparison; current nonzero final regional loss suggests expressivity or
   conflicting objectives, not merely insufficient weighting.
5. Profile `_check_faces` and cache unchanged cell/facet structure between
   verifier refreshes where possible.
6. Run the alpha grid from permissive to strict (`1.5` down to `0.5`) and stop
   after finding the first exactly verified candidate. Record the minimum
   verified alpha in the eventual slow integration regression.
7. If no alpha passes, perform a feasibility study (including reach/unsafe
   Monte Carlo for diagnosis only) before further optimizer tuning. Exact
   verification must remain the acceptance gate.

## Important caveat

The current documented OU test still uses an injected sharp generator fixture;
it is not yet the requested real-training regression. Replace it only after a
deterministic training configuration produces an exactly verified candidate.
