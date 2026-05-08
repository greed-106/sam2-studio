from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    """Static defaults for the desktop app."""

    project_root: Path = Path(__file__).resolve().parents[3]
    default_model_name: str = "sam2.1_l.pt"
    default_imgsz: int = 1024
    default_conf: float = 0.25

    @property
    def default_model_path(self) -> Path:
        return self.project_root / self.default_model_name
