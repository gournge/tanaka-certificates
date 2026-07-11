from scripts.plot_sde_trajectories import plot_trajectories
from tanaka_certificates.sde import BrownianMotion


def test_plot_trajectories_returns_artifact(tmp_path) -> None:
    artifact = plot_trajectories(
        {"Brownian motion": BrownianMotion()},
        {"Brownian motion": 0.0},
        horizon=0.1,
        n_steps=10,
        seeds=(1,),
        output_root=tmp_path,
    )

    assert artifact.name == "sde_trajectories"
    assert artifact.files == [artifact.directory / "trajectories.pdf"]
    assert artifact.files[0].is_file()
