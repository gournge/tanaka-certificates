"""Train a verification-oriented ResidualDeepICNNCertificate with Adam.

The script is self-contained within this repository.  It can train the default
19,824-parameter residual ICNN from scratch or resume a checkpoint produced by
an earlier run.  Dense deterministic audits drive counterexample replay for
all certificate conditions, and the best worst-margin checkpoint is restored.

Examples
--------
Train from scratch::

    uv run python scripts/train_residual_icnn_adam.py --epochs 3000

Continue a local checkpoint::

    uv run python scripts/train_residual_icnn_adam.py \
        --checkpoint output/<run>/best_certificate.pt --epochs 2000
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.plot_trained_pwq_certificate import _add_regions
from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn import ResidualDeepICNNCertificate
from tanaka_certificates.nn.train_certificate import (
    _initialize_smooth_ridge_basis,
    _sample_region,
    _values_and_generator,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem


CONDITIONS = ("initial", "unsafe", "boundary", "nonnegative", "generator")


@dataclass(frozen=True)
class Configuration:
    epochs: int = 3_000
    batch_size: int = 1_024
    smooth_width: int = 16
    icnn_width: int = 79
    icnn_layers: int = 4
    alpha: float = 1.2
    epsilon: float = 0.1
    learning_rate: float = 3e-4
    final_learning_rate_fraction: float = 0.1
    generator_mode: str = "hard_sublevel"
    constraint_margin: float = 0.01
    generator_margin: float = 0.02
    nonnegative_margin: float = 0.005
    replay_weight: float = 2.0
    replay_capacity: int = 128
    replay_trigger_margin: float = 0.02
    hard_count: int = 8
    audit_interval: int = 50
    audit_resolution: int = 81
    record_interval: int = 10
    plot_resolution: int = 220
    train_feature_geometry: bool = True
    checkpoint: str | None = None
    seed: int = 2041


def _initialize(problem, config: Configuration) -> ResidualDeepICNNCertificate:
    model = ResidualDeepICNNCertificate(
        2,
        smooth_width=config.smooth_width,
        icnn_width=config.icnn_width,
        icnn_layers=config.icnn_layers,
        output_scale=problem.beta,
    )
    _initialize_smooth_ridge_basis(model, problem.domain)
    with torch.no_grad():
        model.convex_kink.raw_output_weights.fill_(-2.0)
        for recurrent in model.convex_kink.raw_recurrent_weights:
            recurrent.fill_(-2.0)
    if config.checkpoint is not None:
        checkpoint = Path(config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except RuntimeError as error:
            raise ValueError(
                "checkpoint is incompatible with the requested smooth/ICNN widths"
            ) from error
    if not config.train_feature_geometry:
        for parameter in model.smooth.hinge.parameters():
            parameter.requires_grad_(False)
        for parameter in model.convex_kink.input_layers.parameters():
            parameter.requires_grad_(False)
        for parameter in model.convex_kink.raw_recurrent_weights:
            parameter.requires_grad_(False)
    return model


def _outside_target(problem, points: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [not problem.target.contains(point) for point in points.detach().cpu().numpy()],
        dtype=torch.bool,
        device=points.device,
    )


def _boundary_points(problem, count: int, dtype: torch.dtype) -> torch.Tensor:
    """Stratify points over all four faces and always include corners."""
    lower = torch.tensor(problem.domain.lower, dtype=dtype)
    upper = torch.tensor(problem.domain.upper, dtype=dtype)
    per_face = max(1, int(np.ceil(count / 4)))
    rates = (torch.arange(per_face, dtype=dtype) + torch.rand(per_face)) / per_face
    vertical = lower[1] + rates * (upper[1] - lower[1])
    horizontal = lower[0] + rates[torch.randperm(per_face)] * (upper[0] - lower[0])
    points = torch.vstack(
        (
            torch.column_stack((torch.full_like(vertical, lower[0]), vertical)),
            torch.column_stack((torch.full_like(vertical, upper[0]), vertical)),
            torch.column_stack((horizontal, torch.full_like(horizontal, lower[1]))),
            torch.column_stack((horizontal, torch.full_like(horizontal, upper[1]))),
            torch.stack((lower, torch.tensor([lower[0], upper[1]], dtype=dtype),
                         torch.tensor([upper[0], lower[1]], dtype=dtype), upper)),
        )
    )
    return points[_outside_target(problem, points)]


def _target_exterior_points(problem, count: int, dtype: torch.dtype) -> torch.Tensor:
    """Sample just outside every target face, where generator peaks concentrate."""
    lower = torch.tensor(problem.target.lower, dtype=dtype)
    upper = torch.tensor(problem.target.upper, dtype=dtype)
    domain_lower = torch.tensor(problem.domain.lower, dtype=dtype)
    domain_upper = torch.tensor(problem.domain.upper, dtype=dtype)
    per_face = max(1, int(np.ceil(count / 4)))
    rates = (torch.arange(per_face, dtype=dtype) + torch.rand(per_face)) / per_face
    offsets = torch.exp(
        np.log(1e-4) + torch.rand(per_face, dtype=dtype) * (np.log(0.15) - np.log(1e-4))
    )
    x = lower[0] + rates * (upper[0] - lower[0])
    y = lower[1] + rates[torch.randperm(per_face)] * (upper[1] - lower[1])
    return torch.vstack(
        (
            torch.column_stack(((lower[0] - offsets).clamp_min(domain_lower[0]), y)),
            torch.column_stack(((upper[0] + offsets).clamp_max(domain_upper[0]), y)),
            torch.column_stack((x, (lower[1] - offsets).clamp_min(domain_lower[1]))),
            torch.column_stack((x, (upper[1] + offsets).clamp_max(domain_upper[1]))),
        )
    )


def _aggregate(residual: torch.Tensor, hard_count: int) -> torch.Tensor:
    violations = torch.relu(residual)
    if violations.numel() == 0:
        return violations.sum()
    count = min(hard_count, violations.numel())
    return violations.mean() + torch.topk(violations.reshape(-1), count).values.mean()


def _generator_loss(values, generator, outside, problem, config) -> torch.Tensor:
    residual = (generator[outside] + problem.epsilon + config.generator_margin) / problem.beta
    if config.generator_mode == "full_domain":
        selected = residual
    else:
        selected = residual[values[outside].squeeze(-1) <= problem.beta]
    return _aggregate(selected, config.hard_count)


def _sampled_losses(model, sde, problem, config, replay, dtype):
    initial = _sample_region(problem.initial, config.batch_size, dtype, 0.5)
    unsafe = _sample_region(problem.unsafe, config.batch_size, dtype, 0.5)
    boundary = _boundary_points(problem, config.batch_size, dtype)
    domain = torch.vstack(
        (
            _sample_region(problem.domain, config.batch_size, dtype, 0.15),
            _target_exterior_points(problem, config.batch_size // 2, dtype),
            _sample_region(problem.initial, max(1, config.batch_size // 4), dtype, 0.5),
            replay["generator"],
        )
    ).requires_grad_(True)
    values, generator = _values_and_generator(model, sde, domain)
    losses = {
        "initial": _aggregate(
            (model(initial).squeeze(-1) - (problem.alpha - config.constraint_margin))
            / problem.beta,
            config.hard_count,
        ),
        "unsafe": _aggregate(
            ((problem.beta + config.constraint_margin) - model(unsafe).squeeze(-1))
            / problem.beta,
            config.hard_count,
        ),
        "boundary": _aggregate(
            ((problem.beta + config.constraint_margin) - model(boundary).squeeze(-1))
            / problem.beta,
            config.hard_count,
        ),
        "nonnegative": _aggregate(
            (config.nonnegative_margin - values.squeeze(-1)) / problem.beta,
            config.hard_count,
        ),
        "generator": _generator_loss(
            values, generator, _outside_target(problem, domain), problem, config
        ),
    }
    replay_losses = _counterexample_losses(model, sde, problem, config, replay)
    return {
        name: losses[name] + config.replay_weight * replay_losses[name]
        for name in CONDITIONS
    }


def _counterexample_losses(model, sde, problem, config, replay):
    zero = next(model.parameters()).sum() * 0.0
    losses = {name: zero for name in CONDITIONS}
    if replay["initial"].numel():
        losses["initial"] = _aggregate(
            (model(replay["initial"]).squeeze(-1) - (problem.alpha - config.constraint_margin))
            / problem.beta,
            config.hard_count,
        )
    for name in ("unsafe", "boundary"):
        if replay[name].numel():
            losses[name] = _aggregate(
                ((problem.beta + config.constraint_margin) - model(replay[name]).squeeze(-1))
                / problem.beta,
                config.hard_count,
            )
    if replay["nonnegative"].numel():
        losses["nonnegative"] = _aggregate(
            (config.nonnegative_margin - model(replay["nonnegative"]).squeeze(-1))
            / problem.beta,
            config.hard_count,
        )
    if replay["generator"].numel():
        points = replay["generator"].detach().clone().requires_grad_(True)
        values, generator = _values_and_generator(model, sde, points)
        losses["generator"] = _generator_loss(
            values, generator, _outside_target(problem, points), problem, config
        )
    return losses


def _total_loss(losses) -> torch.Tensor:
    conditions = torch.stack([losses[name] for name in CONDITIONS])
    return conditions.mean() + conditions.max()


def _rectangle_grid(rectangle, resolution: int) -> np.ndarray:
    x = np.linspace(rectangle.lower[0], rectangle.upper[0], resolution)
    y = np.linspace(rectangle.lower[1], rectangle.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _boundary_grid(problem, resolution: int) -> np.ndarray:
    rate = np.linspace(0.0, 1.0, resolution)
    lower, upper = problem.domain.lower, problem.domain.upper
    return np.vstack(
        (
            np.column_stack((np.full_like(rate, lower[0]), lower[1] + rate * (upper[1] - lower[1]))),
            np.column_stack((np.full_like(rate, upper[0]), lower[1] + rate * (upper[1] - lower[1]))),
            np.column_stack((lower[0] + rate * (upper[0] - lower[0]), np.full_like(rate, lower[1]))),
            np.column_stack((lower[0] + rate * (upper[0] - lower[0]), np.full_like(rate, upper[1]))),
        )
    )


def _audit(model, sde, problem, config, dtype):
    arrays = {
        "initial": _rectangle_grid(problem.initial, config.audit_resolution),
        "unsafe": np.vstack(
            [_rectangle_grid(rectangle, config.audit_resolution) for rectangle in problem.unsafe]
        ),
        "boundary": _boundary_grid(problem, config.audit_resolution),
        "nonnegative": _rectangle_grid(problem.domain, config.audit_resolution),
    }
    points = {name: torch.as_tensor(array, dtype=dtype) for name, array in arrays.items()}
    with torch.no_grad():
        values = {name: model(value).squeeze(-1) for name, value in points.items()}
    generator_points = torch.vstack(
        (
            points["nonnegative"],
            _target_exterior_points(problem, 4 * config.audit_resolution, dtype),
        )
    ).requires_grad_(True)
    generator_values, generator = _values_and_generator(model, sde, generator_points)
    generator_mask = _outside_target(problem, generator_points)
    if config.generator_mode == "hard_sublevel":
        generator_mask &= generator_values.squeeze(-1) <= problem.beta
    generator_indices = torch.nonzero(generator_mask, as_tuple=False).squeeze(-1)
    generator_index = int(generator_indices[int(generator[generator_mask].argmax())])
    margins = {
        "initial": problem.alpha - float(values["initial"].max()),
        "unsafe": float(values["unsafe"].min()) - problem.beta,
        "boundary": float(values["boundary"].min()) - problem.beta,
        "nonnegative": float(values["nonnegative"].min()),
        "generator": -problem.epsilon - float(generator[generator_index].detach()),
    }
    worst = {
        "initial": points["initial"][int(values["initial"].argmax())],
        "unsafe": points["unsafe"][int(values["unsafe"].argmin())],
        "boundary": points["boundary"][int(values["boundary"].argmin())],
        "nonnegative": points["nonnegative"][int(values["nonnegative"].argmin())],
        "generator": generator_points[generator_index].detach(),
    }
    return margins, worst


def _update_replay(replay, margins, worst, config):
    for name in CONDITIONS:
        if margins[name] >= config.replay_trigger_margin:
            continue
        replay[name] = torch.vstack((replay[name], worst[name].reshape(1, -1)))
        replay[name] = replay[name][-config.replay_capacity :]


def _score(margins) -> float:
    """Minimize the negative minimum margin, with failed conditions as a tie-break."""
    return -min(margins.values()) + 0.01 * sum(max(0.0, -value) for value in margins.values())


def _gradient_norm(parameters) -> float:
    return float(
        np.sqrt(
            sum(
                float(parameter.grad.detach().square().sum())
                for parameter in parameters
                if parameter.grad is not None
            )
        )
    )


def _copy_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _evaluate(model, sde, points: np.ndarray):
    dtype = next(model.parameters()).dtype
    all_values, all_generators = [], []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points) / 4096)))):
        inputs = torch.as_tensor(batch, dtype=dtype).requires_grad_(True)
        values, generator = _values_and_generator(model, sde, inputs)
        all_values.append(values.detach().squeeze(-1).numpy())
        all_generators.append(generator.detach().numpy())
    return np.concatenate(all_values), np.concatenate(all_generators)


def _plot(artifact, model, sde, problem, config, history):
    audit_epochs = history["audit_epoch"]
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for name in CONDITIONS:
        axis.plot(audit_epochs, history[f"audit_{name}"], marker="o", label=name)
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set(
        xlabel="epoch", ylabel="signed margin (positive is feasible)",
        title="Verifier-aligned dense audit",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(artifact.path("audit_margins.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for name in ("total", *CONDITIONS):
        axes[0].plot(history["train_epoch"], history[f"loss_{name}"], label=name)
    axes[0].set(xlabel="epoch", ylabel="loss", title="Adam losses including replay")
    axes[0].set_yscale("symlog", linthresh=1e-7)
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    for name in ("total", "smooth", "icnn"):
        axes[1].plot(history["train_epoch"], history[f"gradient_{name}"], label=name)
    axes[1].set(xlabel="epoch", ylabel="L2 norm", title="Gradient flow")
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(artifact.path("optimization_diagnostics.png"), dpi=180)
    plt.close(figure)

    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], config.plot_resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], config.plot_resolution)
    xx, yy = np.meshgrid(x, y)
    values, generator = _evaluate(model, sde, np.column_stack((xx.ravel(), yy.ravel())))
    values = values.reshape(xx.shape)
    generator = generator.reshape(xx.shape)
    residual = generator + problem.epsilon
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), constrained_layout=True)
    for axis, data, cmap, title in (
        (axes[0], values, "viridis", "certificate V"),
        (axes[1], generator, "coolwarm", "generator LV"),
        (axes[2], residual, "coolwarm", "residual LV + epsilon"),
    ):
        if cmap == "coolwarm":
            limit = max(float(np.abs(data).max()), np.finfo(float).eps)
            levels = np.linspace(-limit, limit, 31)
        else:
            levels = 31
        contour = axis.contourf(xx, yy, data, levels=levels, cmap=cmap, extend="both")
        figure.colorbar(contour, ax=axis, fraction=0.046)
        _add_regions(axis, problem, legend=axis is axes[0])
        axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal", title=title)
    axes[0].contour(
        xx, yy, values, levels=[problem.alpha, problem.beta],
        colors=["#ffcf33", "white"], linewidths=1.8,
    )
    axes[1].contour(xx, yy, generator, levels=[-problem.epsilon], colors="black")
    axes[1].contour(xx, yy, values, levels=[problem.beta], colors="white", linestyles="--")
    axes[2].contour(xx, yy, residual, levels=[0.0], colors="black")
    figure.savefig(artifact.path("certificate_diagnostics.png"), dpi=180)
    plt.close(figure)


def _validate(config):
    if (
        config.epochs <= 0 or config.batch_size <= 0 or config.smooth_width <= 0
        or config.icnn_width <= 0 or config.icnn_layers <= 0
        or config.learning_rate <= 0.0 or config.audit_interval <= 0
        or config.audit_resolution < 10 or config.record_interval <= 0
        or config.plot_resolution < 40 or config.replay_capacity <= 0
        or config.replay_weight < 0.0 or config.hard_count <= 0
        or not 0.0 < config.alpha < 2.0 or config.epsilon < 0.0
    ):
        raise ValueError("invalid training configuration")
    if config.generator_mode not in {"hard_sublevel", "full_domain"}:
        raise ValueError("unknown generator mode")


def run(config: Configuration, output_root: str | Path = "output") -> ResultArtifact:
    _validate(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    sde, problem = make_enlarged_target_ou_problem(alpha=config.alpha, epsilon=config.epsilon)
    model = _initialize(problem, config)
    dtype = next(model.parameters()).dtype
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
        eta_min=config.learning_rate * config.final_learning_rate_fraction,
    )
    replay = {name: torch.empty((0, 2), dtype=dtype) for name in CONDITIONS}
    history = {"audit_epoch": [], "train_epoch": []}
    for name in CONDITIONS:
        history[f"audit_{name}"] = []
        history[f"loss_{name}"] = []
    history["loss_total"] = []
    for name in ("total", "smooth", "icnn"):
        history[f"gradient_{name}"] = []

    artifact = ResultArtifact.create("residual_icnn_adam", output_root)
    margins, worst = _audit(model, sde, problem, config, dtype)
    _update_replay(replay, margins, worst, config)
    best_score, best_epoch = _score(margins), 0
    best_margins, best_state = margins.copy(), _copy_state(model)

    def record_audit(epoch, current):
        history["audit_epoch"].append(epoch)
        for name in CONDITIONS:
            history[f"audit_{name}"].append(current[name])
        print(
            f"epoch={epoch:5d} score={_score(current):+.5f} "
            + " ".join(f"{name}={current[name]:+.4f}" for name in CONDITIONS),
            flush=True,
        )

    record_audit(0, margins)
    torch.save(best_state, artifact.path("best_certificate.pt"))
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        losses = _sampled_losses(model, sde, problem, config, replay, dtype)
        total = _total_loss(losses)
        total.backward()
        smooth_gradient = _gradient_norm(model.smooth.parameters())
        icnn_gradient = _gradient_norm(model.convex_kink.parameters())
        total_gradient = float(np.hypot(smooth_gradient, icnn_gradient))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % config.record_interval == 0 or epoch == config.epochs:
            history["train_epoch"].append(epoch)
            history["loss_total"].append(float(total.detach()))
            for name in CONDITIONS:
                history[f"loss_{name}"].append(float(losses[name].detach()))
            history["gradient_total"].append(total_gradient)
            history["gradient_smooth"].append(smooth_gradient)
            history["gradient_icnn"].append(icnn_gradient)
        if epoch % config.audit_interval == 0 or epoch == config.epochs:
            margins, worst = _audit(model, sde, problem, config, dtype)
            _update_replay(replay, margins, worst, config)
            record_audit(epoch, margins)
            torch.save(model.state_dict(), artifact.path("latest_certificate.pt"))
            if _score(margins) < best_score:
                best_score, best_epoch = _score(margins), epoch
                best_margins, best_state = margins.copy(), _copy_state(model)
                torch.save(best_state, artifact.path("best_certificate.pt"))

    model.load_state_dict(best_state)
    restored_margins, _ = _audit(model, sde, problem, config, dtype)
    _plot(artifact, model, sde, problem, config, history)
    artifact.path("configuration.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    artifact.path("history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    artifact.path("run.log").write_text(
        "\n".join(
            (
                "Residual ICNN Adam training (numerical pre-verification)",
                f"source_checkpoint={config.checkpoint or 'scratch'}",
                f"parameters={sum(parameter.numel() for parameter in model.parameters())}",
                f"best_epoch={best_epoch}",
                f"best_score={best_score:+.10g}",
                *(f"best_{name}_margin={best_margins[name]:+.10g}" for name in CONDITIONS),
                *(f"restored_{name}_margin={restored_margins[name]:+.10g}" for name in CONDITIONS),
                f"all_numerically_satisfied={all(value >= 0.0 for value in restored_margins.values())}",
                "formal_verification=not_run",
            )
        ),
        encoding="utf-8",
    )
    return artifact


def _arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--smooth-width", type=int, default=16)
    parser.add_argument("--icnn-width", type=int, default=79)
    parser.add_argument("--icnn-layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate-fraction", type=float, default=0.1)
    parser.add_argument("--generator-mode", choices=("hard_sublevel", "full_domain"), default="hard_sublevel")
    parser.add_argument("--constraint-margin", type=float, default=0.01)
    parser.add_argument("--generator-margin", type=float, default=0.02)
    parser.add_argument("--nonnegative-margin", type=float, default=0.005)
    parser.add_argument("--replay-weight", type=float, default=2.0)
    parser.add_argument("--replay-capacity", type=int, default=128)
    parser.add_argument("--replay-trigger-margin", type=float, default=0.02)
    parser.add_argument("--hard-count", type=int, default=8)
    parser.add_argument("--audit-interval", type=int, default=50)
    parser.add_argument("--audit-resolution", type=int, default=81)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--plot-resolution", type=int, default=220)
    parser.add_argument("--train-feature-geometry", action=BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    configuration = Configuration(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        smooth_width=arguments.smooth_width,
        icnn_width=arguments.icnn_width,
        icnn_layers=arguments.icnn_layers,
        alpha=arguments.alpha,
        epsilon=arguments.epsilon,
        learning_rate=arguments.learning_rate,
        final_learning_rate_fraction=arguments.final_learning_rate_fraction,
        generator_mode=arguments.generator_mode,
        constraint_margin=arguments.constraint_margin,
        generator_margin=arguments.generator_margin,
        nonnegative_margin=arguments.nonnegative_margin,
        replay_weight=arguments.replay_weight,
        replay_capacity=arguments.replay_capacity,
        replay_trigger_margin=arguments.replay_trigger_margin,
        hard_count=arguments.hard_count,
        audit_interval=arguments.audit_interval,
        audit_resolution=arguments.audit_resolution,
        record_interval=arguments.record_interval,
        plot_resolution=arguments.plot_resolution,
        train_feature_geometry=arguments.train_feature_geometry,
        checkpoint=str(arguments.checkpoint) if arguments.checkpoint is not None else None,
        seed=arguments.seed,
    )
    print(run(configuration, arguments.output).directory)
