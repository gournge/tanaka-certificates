from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.train_optimized_alpha_frontier as experiment


def test_frontier_experiment_records_verified_results(monkeypatch, tmp_path):
    def fake_train(*, epsilon, output, **kwargs):
        Path(output).write_bytes(b"checkpoint")
        model = SimpleNamespace(
            lp_statistics={
                "seed": kwargs["seed"],
                "smooth_width": kwargs["smooth_width"],
                "constraint_count": 12,
                "refinement_iterations": 1,
                "solver_status": 0,
                "solver_message": "optimal",
                "solver_iterations": 3,
                "solve_seconds": 0.1,
            }
        )
        return model, 1.12 + epsilon, 0.04

    class FakeVerifier:
        cells = [object(), object()]
        cell_discovery = SimpleNamespace(unresolved_regions=[])
        issues = []

        def __init__(self, *args):
            pass

        def verify(self):
            return SimpleNamespace(value="verified")

    monkeypatch.setattr(experiment, "train_optimized_alpha_fixed_pwq_lp", fake_train)
    monkeypatch.setattr(experiment, "VerifierLocalTimeByConstruction", FakeVerifier)

    artifact = experiment.train_optimized_alpha_frontier(
        epsilons=(0.0, 0.1), output_root=tmp_path
    )

    assert [path.name for path in artifact.files] == [
        "certificate_epsilon_0.pt",
        "certificate_epsilon_0p1.pt",
        "alpha_frontier.png",
        "frontier.log",
    ]
    log = artifact.files[-1].read_text()
    assert "[epsilon=0]" in log
    assert "alpha=1.12" in log
    assert "[epsilon=0.1]" in log
    assert "alpha=1.22" in log
    assert log.count("verification=verified") == 2
    assert "experiment_sha256=" in log
    assert "solver_sha256=" in log


@pytest.mark.parametrize("epsilons", [(), (-0.1,)])
def test_frontier_experiment_rejects_invalid_epsilons(epsilons, tmp_path):
    with pytest.raises(ValueError, match="epsilons"):
        experiment.train_optimized_alpha_frontier(
            epsilons=epsilons, output_root=tmp_path
        )
