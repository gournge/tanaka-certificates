from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from tanaka_certificates.sde.base import SDEND
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    PiecewiseQuadraticActivation,
    get_relu_like_piecewise_quadratic_activation,
)


@dataclass
class TrainingCertificateConfiguration:
    epochs: int = 100
    batch_size: int = 256
    hidden_width: int = 8
    learning_rate: float = 3e-3
    boundary_loss_weight: float = 10.0
    generator_loss_weight: float = 1.0
    constraint_margin: float = 0.1
    gradient_clip: float = 10.0
    record_network_weights_over_time: bool = True
    torch_seed: int = 2026


@dataclass
class TrainingCertificateArtifact:
    network_over_time: dict[int, PiecewiseQuadraticCertificate]


def train_pwq_certificate_baseline(
    sde: SDEND,
    ras: ReachAvoidProblem,
    last_layer_piecewise_quadratic_activation: PiecewiseQuadratic1D | None = None,
    training_configuration: TrainingCertificateConfiguration | None = None,
) -> PiecewiseQuadraticCertificate:
    """Train a small PWQ certificate using sampled constraint violations.

    Returns a certificate trained to ensure the conditions as stated in
    `tanaka_certificates/verifier/verifier_qwl.py`

    This is deliberately a sampling baseline, not a formal verifier.  It samples
    the small initial/unsafe sets separately so their losses cannot disappear in
    a domain-wide minibatch, and differentiates the network twice to evaluate
    the SDE generator.
    """
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
        nn.Linear(config.hidden_width, 1, dtype=dtype),
        nn.ReLU(),
        PiecewiseQuadraticActivation(activation),
    )
    optimizer = torch.optim.Adam(certificate.parameters(), lr=config.learning_rate)
    history: dict[int, PiecewiseQuadraticCertificate] = {}

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        initial = _sample_region(ras.initial, config.batch_size, dtype)
        unsafe = _sample_region(ras.unsafe, config.batch_size, dtype)
        domain = _sample_region(ras.domain, config.batch_size, dtype).requires_grad_(True)

        initial_violation = torch.relu(
            certificate(initial) - (ras.alpha - config.constraint_margin)
        )
        unsafe_violation = torch.relu(
            ras.beta + config.constraint_margin - certificate(unsafe)
        )
        # A mean loss alone can hide a small violating corner of a region.  The
        # maximum term deliberately steers this sampling baseline toward the
        # verifier's sup/inf conditions.
        initial_loss = initial_violation.square().mean() + initial_violation.max()
        unsafe_loss = unsafe_violation.square().mean() + unsafe_violation.max()

        values, generator = _values_and_generator(certificate, sde, domain)
        outside_target = torch.tensor(
            [not ras.target.contains(x) for x in domain.detach().cpu().numpy()],
            dtype=torch.bool,
        )
        basin = values.detach().squeeze(-1) < ras.beta
        decrease_points = outside_target & basin
        if bool(decrease_points.any()):
            generator_loss = torch.relu(
                generator[decrease_points] + ras.epsilon
            ).mean()
        else:
            generator_loss = generator.sum() * 0.0

        loss = config.boundary_loss_weight * (initial_loss + unsafe_loss)
        loss = loss + config.generator_loss_weight * generator_loss
        loss.backward()
        nn.utils.clip_grad_norm_(certificate.parameters(), config.gradient_clip)
        optimizer.step()
        if config.record_network_weights_over_time:
            history[epoch] = deepcopy(certificate).requires_grad_(False)

    certificate.training_artifact = TrainingCertificateArtifact(history)
    return certificate


def _validate_inputs(sde, ras, config) -> None:
    if not isinstance(sde, SDEND):
        raise TypeError("the PWQ baseline requires a multidimensional SDE")
    if config.epochs < 0 or config.batch_size <= 0 or config.hidden_width <= 0:
        raise ValueError("epochs must be nonnegative and batch/width must be positive")
    if config.boundary_loss_weight < 0 or config.generator_loss_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if config.constraint_margin < 0:
        raise ValueError("constraint_margin must be nonnegative")
    for name in ("domain", "initial", "unsafe", "target"):
        region = getattr(ras, name)
        if region.dimension != sde.state_dim:
            raise ValueError(f"{name} dimension does not match the SDE")


def _sample_region(region, count: int, dtype: torch.dtype) -> torch.Tensor:
    rectangles = getattr(region, "hyperrectangles", (region,))
    choices = torch.randint(len(rectangles), (count,))
    points = torch.empty((count, region.dimension), dtype=dtype)
    for index, rectangle in enumerate(rectangles):
        mask = choices == index
        low = torch.tensor(np.asarray(rectangle.lower), dtype=dtype)
        high = torch.tensor(np.asarray(rectangle.upper), dtype=dtype)
        points[mask] = low + torch.rand((int(mask.sum()), region.dimension), dtype=dtype) * (
            high - low
        )
    return points


def _values_and_generator(certificate, sde, points):
    values = certificate(points)
    gradient = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
    hessian_rows = []
    for dimension in range(sde.state_dim):
        hessian_rows.append(
            torch.autograd.grad(
                gradient[:, dimension].sum(),
                points,
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    hessian = torch.stack(hessian_rows, dim=1)

    numpy_points = points.detach().cpu().numpy()
    drift = torch.as_tensor(
        np.stack([sde.drift(0.0, point) for point in numpy_points]),
        dtype=points.dtype,
    )
    covariance = torch.as_tensor(
        np.stack(
            [sde.diffusion(0.0, point) @ sde.diffusion(0.0, point).T
             for point in numpy_points]
        ),
        dtype=points.dtype,
    )
    generator = (gradient * drift).sum(1) + 0.5 * (
        hessian * covariance
    ).sum(dim=(1, 2))
    return values, generator
