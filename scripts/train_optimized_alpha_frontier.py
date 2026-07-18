"""Minimize and formally verify the certificate bound across drift strictnesses."""

from argparse import ArgumentParser
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt

from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn.train_fixed_pwq_lp import (
    format_lp_statistics,
    train_optimized_alpha_fixed_pwq_lp,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_optimized_alpha_frontier(
    *,
    epsilons: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1),
    smooth_width: int = 48,
    teacher_offset: float = 0.03,
    alpha_slack: float = 0.02,
    seed: int = 2040,
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Run the lexicographic LP and formally verify every frontier point."""
    if not epsilons or any(epsilon < 0.0 for epsilon in epsilons):
        raise ValueError("epsilons must be a non-empty sequence of nonnegative values")
    if smooth_width <= 0 or teacher_offset < 0.0 or alpha_slack <= 0.0:
        raise ValueError("invalid LP width, teacher offset, or alpha slack")

    artifact = ResultArtifact.create("optimized_alpha_frontier", output_root)
    records = []
    for epsilon in epsilons:
        label = f"{epsilon:g}".replace(".", "p")
        checkpoint = artifact.path(f"certificate_epsilon_{label}.pt")
        model, alpha, teacher_error = train_optimized_alpha_fixed_pwq_lp(
            epsilon=epsilon,
            smooth_width=smooth_width,
            teacher_offset=teacher_offset,
            alpha_slack=alpha_slack,
            seed=seed,
            output=checkpoint,
        )
        sde, problem = make_enlarged_target_ou_problem(
            alpha=alpha,
            epsilon=epsilon,
        )
        verifier = VerifierLocalTimeByConstruction(sde, problem, model)
        verification = verifier.verify()
        records.append(
            {
                "epsilon": epsilon,
                "alpha": alpha,
                "teacher_error": teacher_error,
                "verification": verification.value,
                "cells": len(verifier.cells),
                "unresolved_cells": len(verifier.cell_discovery.unresolved_regions),
                **model.lp_statistics,
            }
        )
        if verification.value != "verified":
            issue_summary = ", ".join(issue.kind.value for issue in verifier.issues)
            raise RuntimeError(
                f"epsilon={epsilon:g}, alpha={alpha:.10g} was not verified: "
                f"{verification.value} ({issue_summary})"
            )

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(
        [record["epsilon"] for record in records],
        [record["alpha"] for record in records],
        "o-",
        color="#1565c0",
        linewidth=2.0,
        label=r"verified $\alpha^\star(\epsilon)+\delta$",
    )
    axis.axhline(1.11, color="#ef6c00", linestyle="--", label="harmonic reference 1.11")
    axis.set(
        xlabel=r"strict generator decrease $\epsilon$",
        ylabel=r"verified probability bound $\alpha$",
        title="Strict drift versus optimized certificate bound",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(artifact.path("alpha_frontier.png"), dpi=190)
    plt.close(figure)

    solver_path = Path(__file__).resolve().parents[1] / "tanaka_certificates/nn/train_fixed_pwq_lp.py"
    lines = [
        "Lexicographic fixed-feature LP alpha frontier",
        f"smooth_width={smooth_width}",
        f"teacher_offset={teacher_offset:g}",
        f"alpha_slack={alpha_slack:g}",
        f"seed={seed}",
        f"experiment_sha256={_sha256(Path(__file__))}",
        f"solver_sha256={_sha256(solver_path)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"[epsilon={record['epsilon']:g}]",
                f"alpha={record['alpha']:.10g}",
                f"teacher_error={record['teacher_error']:.10g}",
                f"verification={record['verification']}",
                f"cells={record['cells']}",
                f"unresolved_cells={record['unresolved_cells']}",
                *format_lp_statistics(
                    {
                        name: value
                        for name, value in record.items()
                        if name
                        not in {
                            "epsilon",
                            "alpha",
                            "teacher_error",
                            "verification",
                            "cells",
                            "unresolved_cells",
                        }
                    }
                ),
                "",
            ]
        )
    artifact.path("frontier.log").write_text("\n".join(lines), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.025, 0.05, 0.1),
    )
    parser.add_argument("--smooth-width", type=int, default=48)
    parser.add_argument("--teacher-offset", type=float, default=0.03)
    parser.add_argument("--alpha-slack", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--output", type=Path, default=Path("output"))
    arguments = parser.parse_args()
    result = train_optimized_alpha_frontier(
        epsilons=tuple(arguments.epsilons),
        smooth_width=arguments.smooth_width,
        teacher_offset=arguments.teacher_offset,
        alpha_slack=arguments.alpha_slack,
        seed=arguments.seed,
        output_root=arguments.output,
    )
    print(result.directory)
