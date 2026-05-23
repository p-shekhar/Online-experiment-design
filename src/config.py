from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem paths shared by notebooks and reusable modules."""

    project_root: Path
    code_dir: Path
    repo_root: Path
    data_dir: Path
    artifact_dir: Path
    figure_dir: Path
    table_dir: Path
    workspace_dir: Path

    @classmethod
    def from_anywhere(cls, start: Path | None = None) -> ProjectPaths:
        start = (start or Path.cwd()).resolve()
        current = start
        while current.name != "online_experiment_designs_under_interference":
            if current == current.parent:
                raise RuntimeError(
                    "Could not find online_experiment_designs_under_interference root."
                )
            current = current.parent

        project_root = current
        code_dir = project_root / "code"
        repo_root = project_root.parents[2]
        artifact_dir = code_dir / "artifacts"
        return cls(
            project_root=project_root,
            code_dir=code_dir,
            repo_root=repo_root,
            data_dir=repo_root / "data",
            artifact_dir=artifact_dir,
            figure_dir=artifact_dir / "figures",
            table_dir=artifact_dir / "tables",
            workspace_dir=artifact_dir / "workspace",
        )

    def ensure(self) -> None:
        for path in [self.artifact_dir, self.figure_dir, self.table_dir, self.workspace_dir]:
            path.mkdir(parents=True, exist_ok=True)


def notebook_bootstrap() -> ProjectPaths:
    """Locate the project, add src to sys.path in notebooks, and create artifact dirs."""
    import sys

    paths = ProjectPaths.from_anywhere()
    src_dir = paths.code_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    paths.ensure()
    return paths
