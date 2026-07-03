"""Plot the normalized piecewise-linear counterexample and curved remedies."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("docs/research/log/figures/motivating_counterexample.pdf")


def create_plot(output: Path) -> None:
    """Compare certificates and generators for dX = dt + sqrt(2) dW."""
    x = np.linspace(0.0, 1.0, 501)

    # Every candidate satisfies V(0)=0 and V(1)=1.
    v_pwl = np.where(x <= 0.5, 1.5 * x, 0.5 * x + 0.5)
    v_quadratic = 2.0 * x - x**2
    v_exact = (1.0 - np.exp(-x)) / (1.0 - np.exp(-1.0))
    generator_pwl = np.where(x < 0.5, 1.5, 0.5)
    generator_quadratic = -2.0 * x

    red, blue, green = "#b44b4b", "#315f8c", "#3b8b5a"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))

    axes[0].plot(x, v_pwl, color=red, lw=2, label="concave PWL candidate")
    axes[0].plot(x, v_quadratic, color=blue, lw=2, label=r"quadratic $2x-x^2$")
    axes[0].plot(x, v_exact, color=green, lw=2, ls="--", label="exact martingale certificate")
    axes[0].scatter([0, 1], [0, 1], color="black", s=18, zorder=4)
    axes[0].annotate(r"$V(0)=0$", (0, 0), xytext=(7, 8), textcoords="offset points")
    axes[0].annotate(r"$V(1)=1$", (1, 1), xytext=(-47, 8), textcoords="offset points")
    axes[0].set(xlabel=r"$x$", ylabel=r"$V(x)$", title="Same boundary conditions")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[1].step(x, generator_pwl, where="post", color=red, lw=2, label=r"PWL: $\mathcal{L}V>0$ (fails)")
    axes[1].plot(x, generator_quadratic, color=blue, lw=2, label=r"quadratic: $\mathcal{L}V=-2x\leq0$")
    axes[1].plot(x, np.zeros_like(x), color=green, lw=2, ls="--", label=r"exact: $\mathcal{L}V=0$")
    axes[1].fill_between(x, 0, generator_pwl, color=red, alpha=0.1)
    axes[1].fill_between(x, generator_quadratic, 0, color=blue, alpha=0.1)
    axes[1].set(
        xlabel=r"$x$",
        ylabel=r"$\mathcal{L}V(x)$",
        title=r"Generator for $dX_t=dt+\sqrt{2}\,dW_t$",
    )
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create_plot(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
