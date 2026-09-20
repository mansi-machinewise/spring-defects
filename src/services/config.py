from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str = "config/tolerances.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
