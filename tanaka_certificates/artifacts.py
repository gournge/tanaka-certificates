"""Shared representation of files produced by experiments and scripts."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess


@dataclass
class ResultArtifact:
    """A named, timestamped directory containing files from one computation."""

    name: str
    directory: Path
    files: list[Path] = field(default_factory=list)

    @staticmethod
    def current_git_hash() -> str:
        """Return the short hash of the commit containing this package."""
        try:
            return subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    @classmethod
    def create(
        cls,
        name: str,
        output_root: str | Path = "output",
        *,
        timestamp: datetime | None = None,
        git_hash: str | None = None,
    ) -> "ResultArtifact":
        """Create ``output_root/<hash>_YYYY-MM-DD_HH-MM-SS_<name>``."""
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError("artifact name must be a non-empty path component")

        created_at = timestamp or datetime.now()
        revision = git_hash or cls.current_git_hash()
        if Path(revision).name != revision or not revision:
            raise ValueError("git hash must be a non-empty path component")
        directory = Path(output_root) / f"{revision}_{created_at:%Y-%m-%d_%H-%M-%S}_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        return cls(name=name, directory=directory)

    def path(self, filename: str | Path) -> Path:
        """Reserve a path in this artifact and return it to the producer."""
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact filename must stay inside its directory")

        result = self.directory / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        if result not in self.files:
            self.files.append(result)
        return result
