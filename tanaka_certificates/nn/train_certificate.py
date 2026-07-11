from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch
from torch import nn

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    PiecewiseQuadraticActivation,
    get_relu_like_piecewise_quadratic_activation,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.sde.base import SDEND


@dataclass
class TrainingCertificateConfiguration:
    epochs: int = 400
    batch_size: int = 256
    hidden_width: int = 8
    learning_rate: float = 3e-3
    boundary_loss_weight: float = 20.0
    generator_loss_weight: float = 5.0
    concavity_loss_weight: float = 5.0
    regularization_weight: float = 1e-6
    constraint_margin: float = 0.1
    generator_margin: float = 0.02
    gradient_clip: float = 10.0
    boundary_pretraining_epochs: int = 50
    boundary_sampling_probability: float = 0.9
    record_network_weights_over_time: bool = True
    torch_seed: int = 2026


@dataclass
class TrainingCertificateArtifact:
    network_over_time: dict[int, PiecewiseQuadraticCertificate]
    final_losses: dict[str, float] = field(default_factory=dict)
    epochs_completed: int = 0


def train_pwq_certificate_baseline(
    sde: SDEND,
    ras: ReachAvoidProblem,
    last_layer_piecewise_quadratic_activation: PiecewiseQuadratic1D | None = None,
    training_configuration: TrainingCertificateConfiguration | None = None,
) -> PiecewiseQuadraticCertificate:
    """Train a deep ReLU network with a scalar PWQ output activation."""

    config = training_configuration or TrainingCertificateConfiguration()
    activation = (
        last_layer_piecewise_quadratic_activation
        or get_relu_like_piecewise_quadratic_activation()
    )
    _validate_inputs(sde, ras, config)
    torch.manual_seed(config.torch_seed)
    dtype = torch.get_default_dtype()
    certificate = PiecewiseQuadraticCertificate(
        nn.Linear(sde.state_dim, config.hidden_width, dtype=dtype),
        nn.ReLU(),
        # nn.Linear(config.hidden_width, config.hidden_width, dtype=dtype),
        # nn.ReLU(),
        # nn.Linear(config.hidden_width, config.hidden_width, dtype=dtype),
        # nn.ReLU(),
        nn.Linear(config.hidden_width, config.hidden_width, dtype=dtype),
        nn.ReLU(),
        nn.Linear(config.hidden_width, config.hidden_width, dtype=dtype),
        nn.ReLU(),
        nn.Linear(config.hidden_width, config.hidden_width, dtype=dtype),
        nn.ReLU(),
        nn.Linear(config.hidden_width, 1, dtype=dtype),
        PiecewiseQuadraticActivation(activation),
    )
    optimizer = torch.optim.Adam(certificate.parameters(), lr=config.learning_rate)
    history: dict[int, PiecewiseQuadraticCertificate] = {}
    final_losses: dict[str, float] = {}

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        initial = _training_region_points(
            ras.initial, config.batch_size, dtype, config.boundary_sampling_probability
        )
        unsafe = _training_region_points(
            ras.unsafe, config.batch_size, dtype, config.boundary_sampling_probability
        )
        target = _training_region_points(
            ras.target, config.batch_size, dtype, config.boundary_sampling_probability
        )

        initial_loss = _upper_level_loss(
            certificate, initial, ras.alpha - config.constraint_margin
        )
        # target_loss = _upper_level_loss(
        #     certificate, target, 0 - config.constraint_margin
        # )
        unsafe_loss = _lower_level_loss(
            certificate, unsafe, ras.beta + config.constraint_margin
        )
        # regional_loss = initial_loss + target_loss + unsafe_loss
        regional_loss = initial_loss + unsafe_loss

        concavity_x, concavity_y = _sample_region_pairs(
            ras.domain,
            config.batch_size,
            dtype,
            config.boundary_sampling_probability,
        )
        concavity_loss = _concavity_loss(certificate, concavity_x, concavity_y)

        zero = sum((parameter.sum() * 0.0 for parameter in certificate.parameters()))
        generator_loss = zero
        if epoch >= config.boundary_pretraining_epochs:
            domain = _sample_region(
                ras.domain,
                config.batch_size,
                dtype,
                config.boundary_sampling_probability,
            )
            domain = domain.requires_grad_(True)
            values, generator = _values_and_generator(certificate, sde, domain)
            outside_target = torch.tensor(
                [not ras.target.contains(x) for x in domain.detach().cpu().numpy()],
                dtype=torch.bool,
            )
            if bool(outside_target.any()):
                generator_violations = torch.relu(
                    generator[outside_target] + ras.epsilon + config.generator_margin
                )
                basin_escape_violations = torch.relu(
                    ras.beta
                    + config.constraint_margin
                    - values[outside_target].squeeze(-1)
                )
                # The verified condition is disjunctive: either the point is
                # outside {V <= beta}, or its generator is sufficiently
                # negative.  Let optimization take whichever repair is
                # currently cheaper instead of detaching the basin predicate
                # and forcing every sampled point through the generator arm.
                violations = torch.minimum(
                    generator_violations, basin_escape_violations
                )
                generator_loss = _worst_case_loss(violations)

        regularization = sum(
            parameter.square().mean() for parameter in certificate.parameters()
        )
        loss = config.boundary_loss_weight * regional_loss
        loss = loss + config.generator_loss_weight * generator_loss
        loss = loss + config.concavity_loss_weight * concavity_loss
        loss = loss + config.regularization_weight * regularization
        loss.backward()
        nn.utils.clip_grad_norm_(certificate.parameters(), config.gradient_clip)
        optimizer.step()
        final_losses = {
            "regional": float(regional_loss.detach()),
            "generator": float(generator_loss.detach()),
            "concavity": float(concavity_loss.detach()),
            "regularization": float(regularization.detach()),
            "total": float(loss.detach()),
        }
        if config.record_network_weights_over_time:
            history[epoch] = deepcopy(certificate).requires_grad_(False)
    certificate.training_artifact = TrainingCertificateArtifact(
        history,
        final_losses,
        config.epochs,
    )
    return certificate


def _region_corners(region) -> torch.Tensor:
    """Return every corner of every rectangle, without random sampling."""
    rectangles = getattr(region, "hyperrectangles", (region,))
    corners = []
    for rectangle in rectangles:
        corners.extend(
            product(*zip(np.asarray(rectangle.lower), np.asarray(rectangle.upper)))
        )
    return torch.as_tensor(np.asarray(corners), dtype=torch.get_default_dtype())


def _training_region_points(region, count, dtype, boundary_probability=0.9):
    return torch.cat(
        (
            _sample_region(region, count, dtype, boundary_probability),
            _region_corners(region).to(dtype),
        )
    )


def _concavity_loss(certificate, x, y):
    """Penalize violations of V((x + y) / 2) >= (V(x) + V(y)) / 2."""
    midpoint_values = certificate((x + y) / 2.0).squeeze(-1)
    endpoint_average = (certificate(x).squeeze(-1) + certificate(y).squeeze(-1)) / 2.0
    return _worst_case_loss(torch.relu(endpoint_average - midpoint_values))


def _upper_level_loss(certificate, points, bound):
    return _worst_case_loss(torch.relu(certificate(points).squeeze(-1) - bound))


def _lower_level_loss(certificate, points, bound):
    return _worst_case_loss(torch.relu(bound - certificate(points).squeeze(-1)))


def _worst_case_loss(violations):
    if violations.numel() == 0:
        return violations.sum()
    # Let every sampled violation contribute a gradient.
    return violations.sum()


def _validate_inputs(sde, ras, config) -> None:
    if not isinstance(sde, SDEND):
        raise TypeError("the PWQ baseline requires a multidimensional SDE")
    positive = (
        config.batch_size,
        config.hidden_width,
    )
    if config.epochs < 0 or any(value <= 0 for value in positive):
        raise ValueError(
            "epochs must be nonnegative and sizes/intervals must be positive"
        )
    if (
        min(
            config.boundary_loss_weight,
            config.generator_loss_weight,
            config.concavity_loss_weight,
            config.regularization_weight,
            config.constraint_margin,
            config.generator_margin,
        )
        < 0
    ):
        raise ValueError("loss weights and margins must be nonnegative")
    if not 0.0 <= config.boundary_sampling_probability <= 1.0:
        raise ValueError("boundary_sampling_probability must be between zero and one")
    for name in ("domain", "initial", "unsafe", "target"):
        if getattr(ras, name).dimension != sde.state_dim:
            raise ValueError(f"{name} dimension does not match the SDE")


def _sample_region(
    region,
    count: int,
    dtype: torch.dtype,
    boundary_probability: float = 0.9,
) -> torch.Tensor:
    rectangles = getattr(region, "hyperrectangles", (region,))
    choices = torch.randint(len(rectangles), (count,))
    points = torch.empty((count, region.dimension), dtype=dtype)
    for index, rectangle in enumerate(rectangles):
        mask = choices == index
        low = torch.tensor(np.asarray(rectangle.lower), dtype=dtype)
        high = torch.tensor(np.asarray(rectangle.upper), dtype=dtype)
        rectangle_count = int(mask.sum())
        rectangle_points = low + torch.rand(
            (rectangle_count, region.dimension), dtype=dtype
        ) * (high - low)
        on_boundary = torch.rand(rectangle_count) < boundary_probability
        boundary_rows = torch.nonzero(on_boundary, as_tuple=False).squeeze(-1)
        if boundary_rows.numel():
            dimensions = torch.randint(region.dimension, (len(boundary_rows),))
            use_upper = torch.randint(2, (len(boundary_rows),)).bool()
            rectangle_points[boundary_rows, dimensions] = torch.where(
                use_upper, high[dimensions], low[dimensions]
            )
        points[mask] = rectangle_points
    return points


def _sample_region_pairs(
    region,
    count: int,
    dtype: torch.dtype,
    boundary_probability: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pairs from the same rectangle, keeping their midpoint in the region."""
    rectangles = getattr(region, "hyperrectangles", (region,))
    choices = torch.randint(len(rectangles), (count,))
    x = torch.empty((count, region.dimension), dtype=dtype)
    y = torch.empty_like(x)
    for index, rectangle in enumerate(rectangles):
        mask = choices == index
        pair_count = int(mask.sum())
        if pair_count == 0:
            continue
        x[mask] = _sample_region(rectangle, pair_count, dtype, boundary_probability)
        y[mask] = _sample_region(rectangle, pair_count, dtype, boundary_probability)
    return x, y


def _values_and_generator(certificate, sde, points):
    values = certificate(points)
    gradient = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
    hessian = torch.stack(
        [
            torch.autograd.grad(
                gradient[:, dimension].sum(),
                points,
                create_graph=True,
                retain_graph=True,
            )[0]
            for dimension in range(sde.state_dim)
        ],
        dim=1,
    )
    numpy_points = points.detach().cpu().numpy()
    drift = torch.as_tensor(
        np.stack([sde.drift(0.0, point) for point in numpy_points]), dtype=points.dtype
    )
    covariance = torch.as_tensor(
        np.stack(
            [
                sde.diffusion(0.0, point) @ sde.diffusion(0.0, point).T
                for point in numpy_points
            ]
        ),
        dtype=points.dtype,
    )
    generator = (gradient * drift).sum(1) + 0.5 * (hessian * covariance).sum(dim=(1, 2))
    return values, generator
