from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[int] | None = None      # [x1, y1, x2, y2]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    detector: str
    triggered: bool
    detections: list[Detection] = field(default_factory=list)
    message: str = ""
    error: str = ""


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image_bytes: bytes) -> DetectionResult:
        """이미지 바이트를 받아 탐지 결과를 반환합니다."""
