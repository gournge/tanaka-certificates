"""Neural certificates with one-sided singular curvature by construction.

The two residual architectures in this module have the form ``V = S - C``.
``S`` is a shallow C1 piecewise-quadratic network and ``C`` is convex and
piecewise affine.  Consequently only ``-C`` contributes singular Hessian
mass, and that mass is negative semidefinite.  This is precisely the
multidimensional local-time sign condition used by the Tanaka verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tanaka_certificates.certificate import Certificate, PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadraticActivation,
    get_relu_like_piecewise_quadratic_activation,
)


class SmoothHingePWQ(nn.Module):
    r"""A C1 PWQ function made from a quadratic and squared affine hinges.

    ``S(x) = c + p^T x + 1/2 x^T H x
             + sum_k v_k/2 ReLU(a_k^T x + b_k)^2``.

    Squared ReLU is C1, so the branch has no surface-local-time term. Unless
    ``enforce_concavity`` is selected, regular curvature may have either sign.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        *,
        enforce_concavity: bool = False,
        dtype=None,
    ):
        super().__init__()
        if input_dim <= 0 or width < 0:
            raise ValueError("input_dim must be positive and width nonnegative")
        factory = {"dtype": dtype} if dtype is not None else {}
        self.input_dim = input_dim
        self.width = width
        self.enforce_concavity = enforce_concavity
        self.offset = nn.Parameter(torch.zeros((), **factory))
        self.linear = nn.Parameter(torch.zeros(input_dim, **factory))
        self.raw_hessian = nn.Parameter(torch.zeros(input_dim, input_dim, **factory))
        self.hinge = nn.Linear(input_dim, width, **factory)
        self.hinge_coefficients = nn.Parameter(torch.zeros(width, **factory))
        if enforce_concavity:
            with torch.no_grad():
                self.raw_hessian.copy_(
                    0.1 * torch.eye(input_dim, dtype=self.raw_hessian.dtype)
                )
                self.hinge_coefficients.fill_(-2.0)

    @property
    def hessian(self) -> torch.Tensor:
        if self.enforce_concavity:
            return -(self.raw_hessian @ self.raw_hessian.T)
        return 0.5 * (self.raw_hessian + self.raw_hessian.T)

    @property
    def effective_hinge_coefficients(self) -> torch.Tensor:
        if self.enforce_concavity:
            return -F.softplus(self.hinge_coefficients)
        return self.hinge_coefficients

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quadratic = 0.5 * torch.einsum(
            "...i,ij,...j->...", inputs, self.hessian, inputs
        )
        hinges = 0.5 * (
            F.relu(self.hinge(inputs)).square() * self.effective_hinge_coefficients
        ).sum(dim=-1)
        return (self.offset + inputs @ self.linear + quadratic + hinges).unsqueeze(-1)


class DeepReLUICNN(nn.Module):
    """A fully input-convex, scalar CPWL network.

    Every hidden layer receives an unrestricted affine input passthrough.
    Hidden-to-hidden and final hidden-to-output weights use ``softplus`` and
    are therefore nonnegative, following the FICNN construction of Amos,
    Xu, and Kolter (2017).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        hidden_layers: int = 2,
        *,
        dtype=None,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("ICNN dimensions and layer count must be positive")
        factory = {"dtype": dtype} if dtype is not None else {}
        self.input_dim = input_dim
        self.input_layers = nn.ModuleList(
            nn.Linear(input_dim, hidden_width, **factory) for _ in range(hidden_layers)
        )
        self.raw_recurrent_weights = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_width, hidden_width, **factory))
            for _ in range(hidden_layers - 1)
        )
        self.raw_output_weights = nn.Parameter(torch.empty(hidden_width, **factory))
        self.output_input = nn.Linear(input_dim, 1, **factory)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.raw_recurrent_weights:
            nn.init.constant_(parameter, -0.5)
        nn.init.constant_(self.raw_output_weights, -0.5)

    def positive_recurrent_weights(self) -> list[torch.Tensor]:
        return [F.softplus(weight) for weight in self.raw_recurrent_weights]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.input_layers[0](inputs))
        for input_layer, recurrent in zip(
            self.input_layers[1:], self.positive_recurrent_weights()
        ):
            hidden = F.relu(input_layer(inputs) + hidden @ recurrent.T)
        return self.output_input(inputs) + (
            hidden * F.softplus(self.raw_output_weights)
        ).sum(dim=-1, keepdim=True)


class MaxAffineConvex(nn.Module):
    """An explicit convex CPWL function ``max_r (a_r^T x + b_r)``."""

    def __init__(self, input_dim: int, pieces: int, *, dtype=None):
        super().__init__()
        if input_dim <= 0 or pieces <= 0:
            raise ValueError("input_dim and pieces must be positive")
        factory = {"dtype": dtype} if dtype is not None else {}
        self.affine = nn.Linear(input_dim, pieces, **factory)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.affine(inputs).amax(dim=-1, keepdim=True)


@dataclass
class _PWQRegion:
    Q: np.ndarray
    p: np.ndarray
    c: float
    A: np.ndarray
    b: np.ndarray
    affine_matrix: np.ndarray | None = None
    affine_bias: np.ndarray | None = None


class LocalTimeByConstructionCertificate(Certificate):
    """Base class for certificates whose kink sign is structurally guaranteed."""

    smooth: SmoothHingePWQ
    convex_kink: nn.Module

    def __init__(self, *, output_scale: float = 1.0):
        super().__init__()
        if output_scale <= 0.0:
            raise ValueError("output_scale must be positive")
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_scale * (self.smooth(inputs) - self.convex_kink(inputs))

    def local_time_condition_by_construction(self) -> bool:
        return isinstance(self.smooth, SmoothHingePWQ) and isinstance(
            self.convex_kink, (DeepReLUICNN, MaxAffineConvex)
        )

    def discover_cells(self) -> list[Cell]:
        """Compatibility wrapper returning conclusively discovered cells."""
        return self.discover_cells_result().cells

    def discover_cells_result(self) -> CellDiscoveryResult:
        """Return cells and every numerically ambiguous candidate region."""
        from tanaka_certificates.cell_discovery import Cell, CellDiscoveryResult

        regions, unresolved = _smooth_regions(self.smooth)
        if isinstance(self.convex_kink, DeepReLUICNN):
            regions, new_unresolved = _subtract_icnn(regions, self.convex_kink)
        elif isinstance(self.convex_kink, MaxAffineConvex):
            regions, new_unresolved = _subtract_max_affine(
                regions, self.convex_kink
            )
        else:  # Defensive: the verifier also checks the structural predicate.
            raise TypeError("unsupported convex kink branch")
        unresolved.extend(new_unresolved)
        return CellDiscoveryResult(
            [
                Cell(
                    i,
                    float(self.output_scale) * region.Q,
                    float(self.output_scale) * region.p,
                    float(self.output_scale) * region.c,
                    region.A,
                    region.b,
                )
                for i, region in enumerate(regions)
            ],
            unresolved,
        )


class ResidualDeepICNNCertificate(LocalTimeByConstructionCertificate):
    """``C1 PWQ - deep ReLU ICNN`` certificate."""

    def __init__(
        self,
        input_dim: int,
        smooth_width: int = 4,
        icnn_width: int = 4,
        icnn_layers: int = 2,
        *,
        enforce_global_concavity: bool = False,
        output_scale: float = 1.0,
        dtype=None,
    ):
        super().__init__(output_scale=output_scale)
        self.smooth = SmoothHingePWQ(
            input_dim,
            smooth_width,
            enforce_concavity=enforce_global_concavity,
            dtype=dtype,
        )
        self.convex_kink = DeepReLUICNN(input_dim, icnn_width, icnn_layers, dtype=dtype)


class ResidualMaxAffineCertificate(LocalTimeByConstructionCertificate):
    """``C1 PWQ - max(affine cuts)`` certificate."""

    def __init__(
        self,
        input_dim: int,
        smooth_width: int = 4,
        max_affine_pieces: int = 6,
        *,
        enforce_global_concavity: bool = False,
        output_scale: float = 1.0,
        dtype=None,
    ):
        super().__init__(output_scale=output_scale)
        self.smooth = SmoothHingePWQ(
            input_dim,
            smooth_width,
            enforce_concavity=enforce_global_concavity,
            dtype=dtype,
        )
        self.convex_kink = MaxAffineConvex(input_dim, max_affine_pieces, dtype=dtype)


class UnconstrainedPWQCertificate(PiecewiseQuadraticCertificate):
    """The unconstrained deep ReLU/PWQ comparison architecture."""

    def __init__(self, input_dim: int, hidden_width: int = 8, *, dtype=None):
        factory = {"dtype": dtype} if dtype is not None else {}
        super().__init__(
            nn.Linear(input_dim, hidden_width, **factory),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width, **factory),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width, **factory),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width, **factory),
            nn.ReLU(),
            nn.Linear(hidden_width, 1, **factory),
            PiecewiseQuadraticActivation(
                get_relu_like_piecewise_quadratic_activation()
            ),
        )


def _numpy(parameter: torch.Tensor) -> np.ndarray:
    return parameter.detach().cpu().numpy().astype(float, copy=True)


def _candidate_region(region: _PWQRegion, additional_A, additional_b):
    from tanaka_certificates.cell_discovery import classify_full_dimensional_interior

    dimension = len(region.p)
    A = np.vstack((region.A, np.asarray(additional_A).reshape(-1, dimension)))
    b = np.r_[region.b, np.asarray(additional_b, dtype=float)]
    return (
        classify_full_dimensional_interior(A, b, dimension=dimension),
        _PWQRegion(
            region.Q.copy(),
            region.p.copy(),
            region.c,
            A,
            b,
            None if region.affine_matrix is None else region.affine_matrix.copy(),
            None if region.affine_bias is None else region.affine_bias.copy(),
        ),
    )


def _append_candidate(children, unresolved, status, child, stage):
    from tanaka_certificates.cell_discovery import (
        FeasibilityStatus,
        UnresolvedRegion,
    )

    if status is FeasibilityStatus.FEASIBLE:
        children.append(child)
        return True
    if status is FeasibilityStatus.UNKNOWN:
        unresolved.append(UnresolvedRegion(child.A, child.b, stage))
    return False


def _smooth_regions(
    smooth: SmoothHingePWQ,
) -> tuple[list[_PWQRegion], list[UnresolvedRegion]]:
    dimension = smooth.input_dim
    H = _numpy(smooth.hessian)
    regions = [
        _PWQRegion(
            H,
            _numpy(smooth.linear),
            float(smooth.offset.detach()),
            np.empty((0, dimension)),
            np.empty(0),
        )
    ]
    weights = _numpy(smooth.hinge.weight)
    biases = _numpy(smooth.hinge.bias)
    coefficients = _numpy(smooth.effective_hinge_coefficients)
    unresolved: list[UnresolvedRegion] = []
    for hinge_index, (normal, bias, coefficient) in enumerate(
        zip(weights, biases, coefficients)
    ):
        children = []
        for region in regions:
            status, inactive = _candidate_region(region, [normal], [-bias])
            _append_candidate(
                children, unresolved, status, inactive, f"smooth_hinge_{hinge_index}"
            )
            status, active = _candidate_region(region, [-normal], [bias])
            if _append_candidate(
                children, unresolved, status, active, f"smooth_hinge_{hinge_index}"
            ):
                active.Q += coefficient * np.outer(normal, normal)
                active.p += coefficient * bias * normal
                active.c += 0.5 * coefficient * bias**2
        regions = children
    return regions, unresolved


def _split_relu_regions(regions, preactivation, stage):
    children = []
    unresolved = []
    for region in regions:
        matrix, bias = preactivation(region)
        for pattern in product((False, True), repeat=len(bias)):
            signs = np.asarray(pattern, dtype=float)
            additional_A = np.where(signs[:, None] > 0, -matrix, matrix)
            additional_b = np.where(signs > 0, bias, -bias)
            status, child = _candidate_region(region, additional_A, additional_b)
            if _append_candidate(children, unresolved, status, child, stage):
                diagonal = np.diag(signs)
                child.affine_matrix = diagonal @ matrix
                child.affine_bias = diagonal @ bias
    return children, unresolved


def _subtract_icnn(regions, icnn: DeepReLUICNN):
    unresolved = []
    first = icnn.input_layers[0]
    W, bias = _numpy(first.weight), _numpy(first.bias)
    regions, new_unresolved = _split_relu_regions(
        regions, lambda _: (W, bias), "icnn_relu_0"
    )
    unresolved.extend(new_unresolved)
    for layer_index, (layer, recurrent) in enumerate(
        zip(icnn.input_layers[1:], icnn.positive_recurrent_weights()), start=1
    ):
        U, bias, Z = _numpy(layer.weight), _numpy(layer.bias), _numpy(recurrent)

        def preactivation(region, U=U, bias=bias, Z=Z):
            return Z @ region.affine_matrix + U, Z @ region.affine_bias + bias

        regions, new_unresolved = _split_relu_regions(
            regions, preactivation, f"icnn_relu_{layer_index}"
        )
        unresolved.extend(new_unresolved)
    output_weight = _numpy(F.softplus(icnn.raw_output_weights))
    input_weight = _numpy(icnn.output_input.weight[0])
    output_bias = float(icnn.output_input.bias.detach()[0])
    for region in regions:
        region.p -= output_weight @ region.affine_matrix + input_weight
        region.c -= float(output_weight @ region.affine_bias + output_bias)
        region.affine_matrix = region.affine_bias = None
    return regions, unresolved


def _subtract_max_affine(regions, maximum: MaxAffineConvex):
    weights, biases = _numpy(maximum.affine.weight), _numpy(maximum.affine.bias)
    result = []
    unresolved = []
    for region in regions:
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            # This cut is maximal when every other cut is no larger.
            status, child = _candidate_region(
                region,
                weights - weight,
                bias - biases,
            )
            if _append_candidate(
                result, unresolved, status, child, f"max_affine_piece_{index}"
            ):
                child.p -= weight
                child.c -= float(bias)
    return result, unresolved
