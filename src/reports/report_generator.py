from __future__ import annotations

import json
from pathlib import Path
from src.models.schemas import InspectionResult


class DefectReportGenerator:
    def __init__(self, root: str = "reports") -> None: self.root = Path(root)

    def generate(self, result: InspectionResult) -> str | None:
        if result.decision != "FAIL": return None
        folder = self.root / result.inspection_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "defect_report.json"
        path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
        return str(path)
