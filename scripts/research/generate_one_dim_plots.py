"""Generate the one-dimensional figures used in the research log."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tanaka_certificates.sde import BrownianMotion, EulerMaruyama, OrnsteinUhlenbeck


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs/research/log/figures"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def simulate_ou(output: Path) -> None:
    """Plot the OU decomposition for three noise-to-mean-reversion regimes."""
    rng = np.random.default_rng(7)
    mean_reversion, x0 = 1.0, 1.0
    ratios = (0.5, 1.0, 1.5)
    models = tuple(
        OrnsteinUhlenbeck(mean_reversion, np.sqrt(2.0 * mean_reversion) * ratio)
        for ratio in ratios
    )
    dt, horizon = 1e-3, 1.5
    n_steps = round(horizon / dt)
    time = np.linspace(0.0, horizon, n_steps + 1)
    n_paths = 8000

    x = np.full((len(models), n_paths), x0)
    martingales = np.zeros_like(x)
    drift_terms = np.zeros_like(x)
    mean_m = np.zeros((len(models), n_steps + 1))
    se_m = np.zeros_like(mean_m)
    mean_d = np.zeros_like(mean_m)
    mean_delta_v = np.zeros_like(mean_m)

    for k, t in enumerate(time[:-1]):
        # Use common random numbers across the three regimes for a cleaner comparison.
        standard_noise = rng.standard_normal(n_paths)
        dw = np.sqrt(dt) * standard_noise
        for j, model in enumerate(models):
            martingales[j] += 2.0 * model.volatility * x[j] * dw
            drift_terms[j] += -2.0 * mean_reversion * x[j] ** 2 * dt
            x[j] += model.drift(t, x[j]) * dt + model.diffusion(t, x[j]) * dw
            mean_m[j, k + 1] = np.mean(martingales[j])
            se_m[j, k + 1] = np.std(martingales[j], ddof=1) / np.sqrt(n_paths)
            mean_d[j, k + 1] = np.mean(drift_terms[j])
            mean_delta_v[j, k + 1] = np.mean(x[j] ** 2 - x0**2)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharex=True)
    regime = ("negative", "zero", "positive")
    for j, (axis, ratio, model) in enumerate(zip(axes, ratios, models)):
        mean_c = model.volatility**2 * time
        axis.axhline(0.0, color="0.4", lw=0.7)
        axis.fill_between(time, mean_m[j] - 2 * se_m[j], mean_m[j] + 2 * se_m[j], color="#315f8c", alpha=0.13, label=r"95\% MC band for $\mathbb{E}[M_t]$")
        axis.plot(time, mean_m[j], color="#315f8c", label=r"$\mathbb{E}[M_t]$")
        axis.plot(time, mean_d[j], color="#b44b4b", label=r"$\mathbb{E}[D_t]$")
        axis.plot(time, mean_c, color="#3b8b5a", label=r"$C_t=\sigma^2t$")
        axis.plot(time, mean_delta_v[j], color="black", lw=1.4, label=r"$\mathbb{E}[X_t^2-X_0^2]$")
        axis.set(xlabel=r"$t$", title=rf"{regime[j]}: $\sigma/\sqrt{{2\lambda}}={ratio:g}$")
    axes[0].set_ylabel("Monte Carlo expectation")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(r"OU process with $V(x)=x^2$, $\lambda=1$, and $X_0=1$")
    fig.tight_layout()
    fig.savefig(output / "ou_generator_three_cases.pdf", bbox_inches="tight")
    plt.close(fig)


def compare_linear_and_curved(output: Path) -> None:
    """Compare V=x and V=x-x^2/2 for constant drift and diffusion."""
    x = np.linspace(0.0, 1.0, 501)
    v_linear, v_curved = x, x - 0.5 * x**2
    generator_linear, generator_curved = np.ones_like(x), -x

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    axes[0].plot(x, v_linear, color="#b44b4b", lw=1.6, label=r"piecewise linear: $V=x$")
    axes[0].plot(x, v_curved, color="#315f8c", lw=1.6, label=r"$\mathcal{C}^2$: $V=x-x^2/2$")
    axes[0].scatter([0, 1], [0, 1], color="#b44b4b", s=15)
    axes[0].scatter([0, 1], [0, 0.5], color="#315f8c", s=15)
    axes[0].set(xlabel=r"$x$", ylabel=r"$V(x)$", title="Increasing certificate candidates")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="0.25", lw=0.8)
    axes[1].plot(x, generator_linear, color="#b44b4b", lw=1.6, label=r"$\mathcal{L}V_{\rm PL}=1$ (failure)")
    axes[1].plot(x, generator_curved, color="#315f8c", lw=1.6, label=r"$\mathcal{L}V_{\mathcal{C}^2}=-x$ (success)")
    axes[1].fill_between(x, generator_curved, 0.0, color="#315f8c", alpha=0.12)
    axes[1].set(xlabel=r"$x$", ylabel=r"$\mathcal{L}V(x)$", title=r"Generator for $f=1$, $g=\sqrt{2}$")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "linear_failure_curvature_success.pdf", bbox_inches="tight")
    plt.close(fig)


def simulate_hybrid(output: Path) -> None:
    """Plot a Brownian path and generalized Ito terms for the hybrid certificate."""
    dt, horizon = 5e-4, 3.0
    n_steps = round(horizon / dt)
    time, x = EulerMaruyama().simulate(BrownianMotion(), -0.2, horizon, n_steps, seed=14)
    dw = np.diff(x)

    def value(z: np.ndarray) -> np.ndarray:
        return np.where(z < 0.0, -0.5 * z**2, -z)

    derivative = np.where(x[:-1] < 0.0, -x[:-1], -1.0)
    m_term = np.r_[0.0, np.cumsum(derivative * dw)]
    c_term = np.r_[0.0, np.cumsum(-0.5 * (x[:-1] < 0.0) * dt)]
    delta_v = value(x) - value(np.asarray(-0.2))
    epsilon = 0.06
    local_time = np.r_[0.0, np.cumsum((np.abs(x[:-1]) < epsilon) * dt / (2 * epsilon))]
    k_term = -0.5 * local_time
    total = m_term + c_term + k_term

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    axes[0].plot(time, x, color="#315f8c", lw=1.0)
    axes[0].axhline(0.0, color="#b44b4b", lw=0.8, label="kink at 0")
    axes[0].fill_between(time, x, 0.0, where=x < 0.0, color="#315f8c", alpha=0.12)
    axes[0].set(xlabel=r"$t$", ylabel=r"$X_t$", title="Brownian path and visits to the curved region")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(time, m_term, label=r"$M_t$", color="#315f8c")
    axes[1].plot(time, c_term, label=r"$C_t$", color="#3b8b5a")
    axes[1].plot(time, k_term, label=r"$K_t$", color="#b44b4b")
    axes[1].plot(time, delta_v, label=r"$V(X_t)-V(X_0)$", color="black", lw=1.5)
    axes[1].plot(time, total, "--", label=r"$M_t+C_t+K_t$", color="#9467bd", lw=1.0)
    axes[1].set(xlabel=r"$t$", title=r"Terms for $V=-x^2/2$ ($x<0$), $V=-x$ ($x\geq0$)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "hybrid_piecewise_c2_terms.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    simulate_ou(args.output)
    compare_linear_and_curved(args.output)
    simulate_hybrid(args.output)


if __name__ == "__main__":
    main()
