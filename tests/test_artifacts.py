from datetime import datetime
from pathlib import Path

import pytest

from tanaka_certificates import ResultArtifact


def test_create_timestamped_artifact(tmp_path: Path) -> None:
    artifact = ResultArtifact.create(
        "example",
        tmp_path,
        timestamp=datetime(2026, 7, 1, 14, 30, 5),
        git_hash="abc1234",
    )

    assert artifact.directory == tmp_path / "abc1234_2026-07-01_14-30-05_example"
    assert artifact.directory.is_dir()
    assert artifact.path("figure.pdf") == artifact.directory / "figure.pdf"
    assert artifact.files == [artifact.directory / "figure.pdf"]


@pytest.mark.parametrize("filename", ["../escape.pdf", "/tmp/escape.pdf"])
def test_artifact_paths_cannot_escape_directory(tmp_path: Path, filename: str) -> None:
    artifact = ResultArtifact.create("example", tmp_path)
    with pytest.raises(ValueError):
        artifact.path(filename)
