import math
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import torch
from torch import nn

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    PiecewiseQuadraticActivation,
    get_relu_like_piecewise_quadratic_activation,
)
from tanaka_certificates.nn.local_time_architectures import (
    LocalTimeByConstructionCertificate,
    ResidualDeepICNNCertificate,
    ResidualMaxAffineCertificate,
    UnconstrainedPWQCertificate,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.sde.base import SDEND


@dataclass
class TrainingCertificateConfiguration:
    epochs: int = 400
    batch_size: int = 256
    hidden_width: int = 8
    smooth_width: int = 4
    icnn_layers: int = 2
    max_affine_pieces: int = 6
    enforce_global_concavity: bool = False
    normalize_certificate_output: bool = True
    normalize_constraint_losses: bool = True
    learning_rate: float = 3e-3
    boundary_loss_weight: float = 20.0
    initial_loss_weight: float | None = None
    unsafe_loss_weight: float | None = None
    domain_boundary_loss_weight: float = 20.0
    generator_loss_weight: float = 5.0
    nonnegativity_loss_weight: float = 20.0
    concavity_loss_weight: float = 10.0
    regularization_weight: float = 1e-6
    teacher_loss_weight: float = 0.0
    constraint_margin: float = 0.1
    generator_margin: float = 0.02
    nonnegativity_margin: float = 0.02
    gradient_clip: float = 10.0
    boundary_pretraining_epochs: int = 50
    boundary_sampling_probability: float = 0.9
    generator_boundary_sampling_probability: float = 0.1
    generator_grid_resolution: int = 17
    include_initial_in_generator_training: bool = False
    verifier_counterexample_interval: int = 0
    restore_best_verifier_checkpoint: bool = True
    generator_training_mode: Literal[
        "full_domain", "hard_sublevel", "disjunction"
    ] = "disjunction"
    train_generator_on_full_domain: bool | None = None
    record_network_weights_over_time: bool = True
    network_record_interval: int = 1
    torch_seed: int = 2026


@dataclass
class TrainingCertificateArtifact:
    network_over_time: dict[int, nn.Module]
    final_losses: dict[str, float] = field(default_factory=dict)
    epochs_completed: int = 0
    selected_checkpoint_epoch: int | None = None
    restored_best_checkpoint: bool = False


CertificateArchitecture = Literal[
    "residual_icnn", "residual_max_affine", "unconstrained_pwq"
]


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
    certificate = UnconstrainedPWQCertificate(
        sde.state_dim, config.hidden_width, dtype=dtype
    )
    certificate[-1] = PiecewiseQuadraticActivation(activation)
    return _train_certificate(
        certificate, sde, ras, config, penalize_concavity=True
    )


def train_certificate(
    sde: SDEND,
    ras: ReachAvoidProblem,
    architecture: CertificateArchitecture = "residual_icnn",
    training_configuration: TrainingCertificateConfiguration | None = None,
    initial_certificate: nn.Module | None = None,
    teacher_points: torch.Tensor | None = None,
    teacher_values: torch.Tensor | None = None,
) -> nn.Module:
    """Train one of the three architectures from the construction note.

    The objective follows the initial, unsafe, nonnegative, and generator
    losses of Neustroev et al. but intentionally omits their target loss. The
    paper equation and released implementation use opposite target-loss
    directions, and this repository currently addresses reach-avoid without
    the disputed stay/target condition.
    """
    config = training_configuration or TrainingCertificateConfiguration()
    _validate_inputs(sde, ras, config)
    if (teacher_points is None) != (teacher_values is None):
        raise ValueError("teacher_points and teacher_values must be provided together")
    if teacher_points is not None:
        if teacher_points.ndim != 2 or teacher_points.shape[1] != sde.state_dim:
            raise ValueError("teacher_points have the wrong shape")
        if teacher_values.shape not in ((len(teacher_points),), (len(teacher_points), 1)):
            raise ValueError("teacher_values have the wrong shape")
    torch.manual_seed(config.torch_seed)
    dtype = torch.get_default_dtype()
    created_certificate = initial_certificate is None
    if initial_certificate is not None:
        certificate = initial_certificate
        expected_type = {
            "residual_icnn": ResidualDeepICNNCertificate,
            "residual_max_affine": ResidualMaxAffineCertificate,
            "unconstrained_pwq": UnconstrainedPWQCertificate,
        }.get(architecture)
        if expected_type is None:
            raise ValueError(f"unknown certificate architecture: {architecture}")
        if not isinstance(certificate, expected_type):
            raise TypeError("initial_certificate does not match architecture")
        if isinstance(certificate, LocalTimeByConstructionCertificate):
            certificate.output_scale.fill_(
                ras.beta if config.normalize_certificate_output else 1.0
            )
    elif architecture == "residual_icnn":
        certificate = ResidualDeepICNNCertificate(
            sde.state_dim,
            smooth_width=config.smooth_width,
            icnn_width=config.hidden_width,
            icnn_layers=config.icnn_layers,
            enforce_global_concavity=config.enforce_global_concavity,
            output_scale=ras.beta if config.normalize_certificate_output else 1.0,
            dtype=dtype,
        )
    elif architecture == "residual_max_affine":
        certificate = ResidualMaxAffineCertificate(
            sde.state_dim,
            smooth_width=config.smooth_width,
            max_affine_pieces=config.max_affine_pieces,
            enforce_global_concavity=config.enforce_global_concavity,
            output_scale=ras.beta if config.normalize_certificate_output else 1.0,
            dtype=dtype,
        )
    elif architecture == "unconstrained_pwq":
        certificate = UnconstrainedPWQCertificate(
            sde.state_dim, config.hidden_width, dtype=dtype
        )
    else:
        raise ValueError(f"unknown certificate architecture: {architecture}")
    if created_certificate and isinstance(
        certificate, LocalTimeByConstructionCertificate
    ):
        _initialize_smooth_ridge_basis(certificate, ras.domain)
    return _train_certificate(
        certificate,
        sde,
        ras,
        config,
        # S - C already gives the required sign to the singular curvature.
        # Requiring the whole certificate to be concave would unnecessarily
        # constrain the regular curvature supplied by the smooth branch.
        penalize_concavity=not isinstance(
            certificate, LocalTimeByConstructionCertificate
        ),
        teacher_points=teacher_points,
        teacher_values=teacher_values,
    )


def _initialize_smooth_ridge_basis(certificate, domain) -> None:
    """Spread the C1 squared-ReLU ridges across a box-shaped domain.

    Directions live in normalized domain coordinates, so this initialization
    remains well scaled when state coordinates have different ranges.  Several
    knots are placed along each direction.  Its cost is linear in input
    dimension and ridge count; it does not require a PDE/grid solution.
    """
    smooth = certificate.smooth
    if (
        smooth.width == 0
        or not hasattr(domain, "lower")
        or not hasattr(domain, "upper")
    ):
        return
    dtype = smooth.hinge.weight.dtype
    device = smooth.hinge.weight.device
    lower = torch.tensor(domain.lower, dtype=dtype, device=device)
    upper = torch.tensor(domain.upper, dtype=dtype, device=device)
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    if bool((half_width <= 0.0).any()):
        return

    direction_count = min(
        smooth.width,
        max(smooth.input_dim, math.ceil(math.sqrt(smooth.width))),
    )
    normalized_directions = []
    eye = torch.eye(smooth.input_dim, dtype=dtype, device=device)
    for index in range(min(direction_count, smooth.input_dim)):
        normalized_directions.append(eye[index])
    if len(normalized_directions) < direction_count:
        random_directions = torch.randn(
            direction_count - len(normalized_directions),
            smooth.input_dim,
            dtype=dtype,
            device=device,
        )
        random_directions = random_directions / random_directions.norm(
            dim=1, keepdim=True
        ).clamp_min(torch.finfo(dtype).eps)
        normalized_directions.extend(random_directions)
    normalized_directions = torch.stack(normalized_directions)
    directions = normalized_directions / half_width

    units_per_direction = [
        sum(index % direction_count == direction for index in range(smooth.width))
        for direction in range(direction_count)
    ]
    used = [0] * direction_count
    weights = []
    biases = []
    for unit in range(smooth.width):
        direction = unit % direction_count
        used[direction] += 1
        # Keep knots away from the extreme supporting hyperplanes so every
        # ridge has active and inactive training volume at initialization.
        fraction = used[direction] / (units_per_direction[direction] + 1)
        relative_knot = (2.0 * fraction - 1.0) * 0.8
        weight = directions[direction]
        projection_radius = (weight.abs() * half_width).sum()
        knot = weight @ center + relative_knot * projection_radius
        weights.append(weight)
        biases.append(-knot)

    with torch.no_grad():
        smooth.hinge.weight.copy_(torch.stack(weights))
        smooth.hinge.bias.copy_(torch.stack(biases))
        # Nonzero coefficients allow gradients to reach ridge directions from
        # the first step while keeping their initial contribution small.
        smooth.hinge_coefficients.normal_(mean=0.0, std=0.02)


def _train_certificate(
    certificate,
    sde,
    ras,
    config,
    *,
    penalize_concavity,
    teacher_points=None,
    teacher_values=None,
):
    dtype = next(certificate.parameters()).dtype
    loss_scale = ras.beta if config.normalize_constraint_losses else 1.0
    optimizer = torch.optim.Adam(certificate.parameters(), lr=config.learning_rate)
    history: dict[int, nn.Module] = {}
    final_losses: dict[str, float] = {}
    counterexamples = torch.empty((0, sde.state_dim), dtype=dtype)
    best_verifier_score = float("inf")
    best_state = None
    best_epoch = None
    epochs_completed = 0

    for epoch in range(config.epochs):
        epochs_completed = epoch + 1
        optimizer.zero_grad()
        initial = _training_region_points(
            ras.initial, config.batch_size, dtype, config.boundary_sampling_probability
        )
        unsafe = _training_region_points(
            ras.unsafe, config.batch_size, dtype, config.boundary_sampling_probability
        )
        initial_loss = _upper_level_loss(
            certificate,
            initial,
            ras.alpha - config.constraint_margin,
            loss_scale,
        )
        # TODO: for now leave this, as the original Greg's paper had a minor issue,
        # so we need to stick to reach-avoid problem with no target loss.
        # target_loss = _upper_level_loss(
        #     certificate, target, 0 - config.constraint_margin
        # )
        unsafe_loss = _lower_level_loss(
            certificate,
            unsafe,
            ras.beta + config.constraint_margin,
            loss_scale,
        )
        # regional_loss = initial_loss + target_loss + unsafe_loss
        regional_loss = initial_loss + unsafe_loss

        domain_boundary = _training_domain_boundary_points(
            ras.domain,
            config.batch_size,
            dtype,
            config.generator_grid_resolution,
        )
        outside_target_boundary = torch.tensor(
            [not ras.target.contains(x) for x in domain_boundary.numpy()],
            dtype=torch.bool,
        )
        domain_boundary_loss = _lower_level_loss(
            certificate,
            domain_boundary[outside_target_boundary],
            ras.beta + config.constraint_margin,
            loss_scale,
        )

        nonnegative_points = _training_region_points(
            ras.domain,
            config.batch_size,
            dtype,
            config.generator_boundary_sampling_probability,
        )
        nonnegativity_loss = _lower_level_loss(
            certificate,
            nonnegative_points,
            config.nonnegativity_margin,
            loss_scale,
        )

        zero = sum((parameter.sum() * 0.0 for parameter in certificate.parameters()))
        teacher_loss = zero
        if teacher_points is not None and config.teacher_loss_weight > 0.0:
            count = min(config.batch_size, len(teacher_points))
            indices = torch.randint(len(teacher_points), (count,))
            selected_points = teacher_points[indices].to(dtype=dtype)
            selected_values = teacher_values[indices].to(dtype=dtype).reshape(-1)
            teacher_errors = (
                certificate(selected_points).squeeze(-1) - selected_values
            ) / loss_scale
            teacher_loss = teacher_errors.square().mean() + teacher_errors.abs().max()
        concavity_loss = zero
        if penalize_concavity and config.concavity_loss_weight > 0.0:
            concavity_x, concavity_y = _sample_region_pairs(
                ras.domain,
                config.batch_size,
                dtype,
                config.boundary_sampling_probability,
            )
            concavity_loss = _concavity_loss(
                certificate, concavity_x, concavity_y, loss_scale
            )

        generator_loss = zero
        if epoch >= config.boundary_pretraining_epochs:
            domain = _sample_region(
                ras.domain,
                config.batch_size,
                dtype,
                config.generator_boundary_sampling_probability,
            )
            if config.generator_grid_resolution > 1:
                domain = torch.cat(
                    (
                        domain,
                        _region_grid(
                            ras.domain,
                            config.generator_grid_resolution,
                            dtype,
                        ),
                    )
                )
            if config.include_initial_in_generator_training:
                domain = torch.cat(
                    (
                        domain,
                        _training_region_points(
                            ras.initial,
                            config.batch_size,
                            dtype,
                            config.boundary_sampling_probability,
                        ),
                    )
                )
            if counterexamples.numel():
                domain = torch.cat((domain, counterexamples))
            domain = domain.requires_grad_(True)
            values, generator = _values_and_generator(certificate, sde, domain)
            outside_target = torch.tensor(
                [not ras.target.contains(x) for x in domain.detach().cpu().numpy()],
                dtype=torch.bool,
            )
            if bool(outside_target.any()):
                generator_violations = torch.relu(
                    generator[outside_target] + ras.epsilon + config.generator_margin
                ) / loss_scale
                generator_mode = config.generator_training_mode
                if config.train_generator_on_full_domain is True:
                    generator_mode = "full_domain"
                elif config.train_generator_on_full_domain is False:
                    generator_mode = "disjunction"
                if generator_mode == "full_domain":
                    # This is stronger than the verified sublevel condition,
                    # but avoids learning irregular holes in {V <= beta} as a
                    # cheap substitute for decreasing the generator.
                    violations = generator_violations
                elif generator_mode == "hard_sublevel":
                    in_basin = (
                        values[outside_target].squeeze(-1) <= ras.beta
                    ).detach()
                    violations = generator_violations[in_basin]
                else:
                    basin_escape_violations = torch.relu(
                        ras.beta
                        + config.constraint_margin
                        - values[outside_target].squeeze(-1)
                    ) / loss_scale
                    violations = torch.minimum(
                        generator_violations, basin_escape_violations
                    )
                generator_loss = _worst_case_loss(violations)

        regularized_parameters = [
            parameter for parameter in certificate.parameters()
            if parameter.requires_grad and parameter.numel() > 0
        ]
        regularization = sum(
            (parameter.square().mean() for parameter in regularized_parameters),
            zero,
        )
        initial_weight = (
            config.boundary_loss_weight
            if config.initial_loss_weight is None
            else config.initial_loss_weight
        )
        unsafe_weight = (
            config.boundary_loss_weight
            if config.unsafe_loss_weight is None
            else config.unsafe_loss_weight
        )
        loss = initial_weight * initial_loss + unsafe_weight * unsafe_loss
        loss = loss + config.domain_boundary_loss_weight * domain_boundary_loss
        loss = loss + config.nonnegativity_loss_weight * nonnegativity_loss
        loss = loss + config.generator_loss_weight * generator_loss
        loss = loss + config.concavity_loss_weight * concavity_loss
        loss = loss + config.teacher_loss_weight * teacher_loss
        loss = loss + config.regularization_weight * regularization
        loss.backward()
        nn.utils.clip_grad_norm_(certificate.parameters(), config.gradient_clip)
        optimizer.step()
        final_losses = {
            "initial": float(initial_loss.detach()),
            "unsafe": float(unsafe_loss.detach()),
            "regional": float(regional_loss.detach()),
            "domain_boundary": float(domain_boundary_loss.detach()),
            "generator": float(generator_loss.detach()),
            "nonnegativity": float(nonnegativity_loss.detach()),
            "concavity": float(concavity_loss.detach()),
            "teacher": float(teacher_loss.detach()),
            "regularization": float(regularization.detach()),
            "total": float(loss.detach()),
        }
        if config.record_network_weights_over_time and (
            epoch % config.network_record_interval == 0 or epoch + 1 == config.epochs
        ):
            history[epoch] = deepcopy(certificate).requires_grad_(False)
        if (
            config.verifier_counterexample_interval > 0
            and (epoch + 1) % config.verifier_counterexample_interval == 0
            and epoch + 1 < config.epochs
        ):
            counterexamples, verifier_score = _augment_verifier_counterexamples(
                certificate, sde, ras, counterexamples, dtype
            )
            if verifier_score < best_verifier_score:
                best_verifier_score = verifier_score
                best_state = deepcopy(certificate.state_dict())
                best_epoch = epoch + 1
            if verifier_score == 0.0:
                break
    if (
        config.verifier_counterexample_interval > 0
    ):
        _, final_verifier_score = _augment_verifier_counterexamples(
            certificate, sde, ras, counterexamples, dtype
        )
        if final_verifier_score < best_verifier_score:
            best_verifier_score = final_verifier_score
            best_state = deepcopy(certificate.state_dict())
            best_epoch = epochs_completed
        restored_best = False
        if config.restore_best_verifier_checkpoint and best_state is not None:
            certificate.load_state_dict(best_state)
            restored_best = best_epoch != epochs_completed
            if config.record_network_weights_over_time and restored_best:
                # A sentinel frame after the final optimization epoch. This
                # guarantees that the GIF ends at the exact checkpoint used
                # for plotting, verification, and serialization.
                history[epochs_completed] = deepcopy(certificate).requires_grad_(False)
        final_losses["best_verifier_score"] = best_verifier_score
    else:
        best_epoch = epochs_completed
        restored_best = False
    certificate.training_artifact = TrainingCertificateArtifact(
        history,
        final_losses,
        epochs_completed,
        best_epoch,
        restored_best,
    )
    return certificate


def _augment_verifier_counterexamples(certificate, sde, ras, existing, dtype):
    """Add exact generator witnesses and a small clipped neighborhood."""
    from tanaka_certificates.verifier import (
        IssueKind,
        VerifierLocalTimeByConstruction,
        VerifierPiecewiseQuadratic,
    )

    was_training = certificate.training
    certificate.eval()
    verifier_type = (
        VerifierLocalTimeByConstruction
        if isinstance(certificate, LocalTimeByConstructionCertificate)
        else VerifierPiecewiseQuadratic
    )
    verifier = verifier_type(sde, ras, certificate)
    verifier.verify()
    score = max(
        (issue.margin / ras.beta for issue in verifier.issues), default=0.0
    )
    points = [
        issue.point for issue in verifier.issues if issue.kind is IssueKind.GENERATOR
    ]
    certificate.train(was_training)
    if not points:
        return existing, score
    domain = ras.domain
    radius = (np.asarray(domain.upper) - np.asarray(domain.lower)) / 100.0
    augmented = []
    for point in points:
        for signs in product((-1.0, 0.0, 1.0), repeat=sde.state_dim):
            neighbor = np.clip(
                point + radius * np.asarray(signs), domain.lower, domain.upper
            )
            augmented.append(neighbor)
    new = torch.as_tensor(np.asarray(augmented), dtype=dtype)
    return torch.unique(torch.cat((existing, new)), dim=0), score


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


def _training_domain_boundary_points(domain, count, dtype, resolution):
    """Sample every face of a box and add deterministic traces on each face."""
    if not hasattr(domain, "lower") or not hasattr(domain, "upper"):
        raise TypeError("domain-boundary training requires a hyperrectangle")
    lower = torch.tensor(np.asarray(domain.lower), dtype=dtype)
    upper = torch.tensor(np.asarray(domain.upper), dtype=dtype)
    dimension = len(lower)
    points = lower + torch.rand((count, dimension), dtype=dtype) * (upper - lower)
    faces = torch.randint(2 * dimension, (count,))
    face_dimensions = faces // 2
    use_upper = (faces % 2).bool()
    rows = torch.arange(count)
    points[rows, face_dimensions] = torch.where(
        use_upper, upper[face_dimensions], lower[face_dimensions]
    )

    rates = torch.linspace(0.0, 1.0, resolution, dtype=dtype)
    traces = []
    center = (lower + upper) / 2.0
    for face_dimension in range(dimension):
        for face_value in (lower[face_dimension], upper[face_dimension]):
            trace = center.repeat(resolution, 1)
            trace[:, face_dimension] = face_value
            varying_dimension = (face_dimension + 1) % dimension
            trace[:, varying_dimension] = (
                lower[varying_dimension]
                + rates * (upper[varying_dimension] - lower[varying_dimension])
            )
            traces.append(trace)
    return torch.cat((points, *traces))


def _concavity_loss(certificate, x, y, scale=1.0):
    """Penalize violations of V((x + y) / 2) >= (V(x) + V(y)) / 2."""
    midpoint_values = certificate((x + y) / 2.0).squeeze(-1)
    endpoint_average = (certificate(x).squeeze(-1) + certificate(y).squeeze(-1)) / 2.0
    # Preserve the baseline trainer's original aggregate midpoint penalty.
    # Regional and generator constraints use the more verifier-focused
    # max-plus-mean objective below.
    return (torch.relu(endpoint_average - midpoint_values) / scale).sum()


def _upper_level_loss(certificate, points, bound, scale=1.0):
    return _worst_case_loss(
        torch.relu(certificate(points).squeeze(-1) - bound) / scale
    )


def _lower_level_loss(certificate, points, bound, scale=1.0):
    return _worst_case_loss(
        torch.relu(bound - certificate(points).squeeze(-1)) / scale
    )


def _worst_case_loss(violations):
    if violations.numel() == 0:
        return violations.sum()
    # Keep gradients from the whole batch while explicitly concentrating on
    # the worst sampled counterexample. A plain sum allowed many tiny repairs
    # to dominate the isolated generator peaks seen by the exact verifier.
    return violations.max() + violations.mean()


def _validate_inputs(sde, ras, config) -> None:
    if not isinstance(sde, SDEND):
        raise TypeError("the PWQ baseline requires a multidimensional SDE")
    positive = (
        config.batch_size,
        config.hidden_width,
        config.smooth_width,
        config.icnn_layers,
        config.max_affine_pieces,
        config.generator_grid_resolution,
        config.network_record_interval,
    )
    if config.epochs < 0 or any(value <= 0 for value in positive):
        raise ValueError(
            "epochs must be nonnegative and sizes/intervals must be positive"
        )
    optional_weights = tuple(
        weight
        for weight in (config.initial_loss_weight, config.unsafe_loss_weight)
        if weight is not None
    )
    if (
        min(
            config.boundary_loss_weight,
            *optional_weights,
            config.domain_boundary_loss_weight,
            config.generator_loss_weight,
            config.nonnegativity_loss_weight,
            config.concavity_loss_weight,
            config.regularization_weight,
            config.teacher_loss_weight,
            config.constraint_margin,
            config.generator_margin,
            config.nonnegativity_margin,
        )
        < 0
    ):
        raise ValueError("loss weights and margins must be nonnegative")
    if not 0.0 <= config.boundary_sampling_probability <= 1.0:
        raise ValueError("boundary_sampling_probability must be between zero and one")
    if not 0.0 <= config.generator_boundary_sampling_probability <= 1.0:
        raise ValueError(
            "generator_boundary_sampling_probability must be between zero and one"
        )
    if config.verifier_counterexample_interval < 0:
        raise ValueError("verifier_counterexample_interval must be nonnegative")
    if config.generator_training_mode not in (
        "full_domain", "hard_sublevel", "disjunction"
    ):
        raise ValueError("unknown generator_training_mode")
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


def _region_grid(region, resolution: int, dtype: torch.dtype) -> torch.Tensor:
    """Return a deterministic Cartesian grid for every rectangle in a region."""
    rectangles = getattr(region, "hyperrectangles", (region,))
    grids = []
    for rectangle in rectangles:
        axes = [
            torch.linspace(float(low), float(high), resolution, dtype=dtype)
            for low, high in zip(rectangle.lower, rectangle.upper)
        ]
        grids.append(torch.cartesian_prod(*axes).reshape(-1, region.dimension))
    return torch.cat(grids)


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
