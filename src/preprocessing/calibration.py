from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    px_per_mm: float

    def px_to_mm(self, pixels: float) -> float:
        return pixels / self.px_per_mm

    @classmethod
    def from_reference(cls, known_size_px: float, known_size_mm: float) -> "Calibration":
        if known_size_px <= 0 or known_size_mm <= 0:
            raise ValueError("Calibration reference dimensions must be positive.")
        return cls(known_size_px / known_size_mm)
