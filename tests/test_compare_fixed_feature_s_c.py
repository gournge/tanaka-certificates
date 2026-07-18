from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.compare_fixed_feature_s_c as experiment


def test_comparison_runs_paired_architectures_and_records_metrics(
    monkeypatch, tmp_path
):
    calls = []

    def fake_train(*, convex_width, seed, output, **kwargs):
        calls.append((convex_width, seed))
        Path(output).write_bytes(b"checkpoint")
        model = SimpleNamespace(
            lp_statistics={
                "active_convex_facets": convex_width,
                "solve_seconds": 0.1 + convex_width,
            }
        )
        return model, 1.2 - 0.001 * convex_width, 0.04

    def fake_adam(*, convex_width, seed, output, **kwargs):
        Path(output).write_bytes(b"checkpoint")
        model = SimpleNamespace(
            lp_statistics={
                "active_convex_facets": convex_width,
                "solve_seconds": 2.0,
            },
            parameters=lambda: iter([__import__("torch").zeros(1)]),
            __call__=lambda points: points[:, :1] * 0.0,
        )
        return model, 1.99, 0.08

    class FakeVerifier:
        issues = []

        def __init__(self, *args):
            self.cells = [object()] * (3 + args[-1].lp_statistics["active_convex_facets"])
            self.cell_discovery = SimpleNamespace(unresolved_regions=[])

        def verify(self):
            return SimpleNamespace(value="verified")

    monkeypatch.setattr(experiment, "train_optimized_alpha_fixed_pwq_lp", fake_train)
    monkeypatch.setattr(experiment, "_train_adam_s_c", fake_adam)
    monkeypatch.setattr(experiment, "VerifierLocalTimeByConstruction", FakeVerifier)
    monkeypatch.setattr(
        experiment,
        "_certificate_values",
        lambda model, points: __import__("numpy").zeros(len(points)),
    )

    artifact = experiment.compare_fixed_feature_s_c(
        seeds=(7, 8), convex_width=2, output_root=tmp_path
    )

    assert calls == [(0, 7), (2, 7), (0, 8), (2, 8)]
    assert [path.name for path in artifact.files] == [
        "certificate_s_lp_seed_7.pt",
        "certificate_s_c_lp_seed_7.pt",
        "certificate_s_c_adam_seed_7.pt",
        "certificate_s_lp_seed_8.pt",
        "certificate_s_c_lp_seed_8.pt",
        "certificate_s_c_adam_seed_8.pt",
        "comparison.png",
        "certificates.png",
        "results.csv",
        "comparison.log",
    ]
    results = artifact.files[-2].read_text()
    assert "architecture,optimizer,seed,alpha" in results
    assert "S-C,LP,8,1.198" in results
    assert "S-C,Adam,8,1.99" in results
    log = artifact.files[-1].read_text()
    assert "[S (LP)]" in log and "[S-C (Adam)]" in log
    assert log.count("active_convex_facets=2,2") == 2
    assert "experiment_sha256=" in log and "solver_sha256=" in log


@pytest.mark.parametrize(
    "arguments",
    [
        {"epsilon": -0.1},
        {"smooth_width": 0},
        {"convex_width": 0},
        {"teacher_offset": -0.1},
        {"alpha_slack": 0.0},
        {"adam_epochs": 0},
        {"seeds": ()},
    ],
)
def test_comparison_rejects_invalid_configuration(arguments, tmp_path):
    with pytest.raises(ValueError):
        experiment.compare_fixed_feature_s_c(output_root=tmp_path, **arguments)
