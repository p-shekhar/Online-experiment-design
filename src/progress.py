from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProgressLogger:
    """Small print logger so long-running notebooks do not feel frozen."""

    prefix: str = "experiment-designs"

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {self.prefix}: {message}", flush=True)

