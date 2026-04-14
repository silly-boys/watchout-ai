"""
위험구역 침입 감지기 (Virtual Fence)
 - 관리자가 API를 통해 폴리곤(다각형 구역)을 등록
 - YOLOv8로 사람을 탐지 후 bbox 하단 중심점이 폴리곤 내부인지 확인
 - 여러 구역(zone) 동시 관리 가능
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import PERSON_CONF, PERSON_MODEL
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)

FENCE_CONFIG_PATH = Path("fence_config.json")


def _load_zones() -> dict[str, list[list[int]]]:
    """저장된 폴리곤 구역을 불러옵니다."""
    if FENCE_CONFIG_PATH.exists():
        return json.loads(FENCE_CONFIG_PATH.read_text())
    return {}


def _save_zones(zones: dict[str, list[list[int]]]) -> None:
    FENCE_CONFIG_PATH.write_text(json.dumps(zones, ensure_ascii=False, indent=2))


def get_zones() -> dict[str, list[list[int]]]:
    return _load_zones()


def set_zone(name: str, polygon: list[list[int]]) -> None:
    """구역을 추가하거나 업데이트합니다. polygon: [[x,y], ...]"""
    zones = _load_zones()
    zones[name] = polygon
    _save_zones(zones)


def delete_zone(name: str) -> bool:
    zones = _load_zones()
    if name not in zones:
        return False
    del zones[name]
    _save_zones(zones)
    return True


def _point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    poly = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


class VirtualFenceDetector(BaseDetector):
    name = "virtual_fence"

    def __init__(self) -> None:
        from ultralytics import YOLO

        model_path = str(PERSON_MODEL) if PERSON_MODEL.exists() else "yolov8n.pt"
        self._model = YOLO(model_path)
        logger.info("VirtualFenceDetector: 모델 로드 (%s)", model_path)

    def detect(self, image_bytes: bytes) -> DetectionResult:
        try:
            zones = _load_zones()
            if not zones:
                return DetectionResult(
                    detector=self.name,
                    triggered=False,
                    message="등록된 위험구역 없음",
                )

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img)

            results = self._model(img_np, conf=PERSON_CONF, verbose=False)[0]
            intrusions: list[Detection] = []

            for box in results.boxes:
                cls_id = int(box.cls)
                if results.names[cls_id] != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 발 위치 = bbox 하단 중심점
                foot_x = (x1 + x2) // 2
                foot_y = y2

                for zone_name, polygon in zones.items():
                    if _point_in_polygon((foot_x, foot_y), polygon):
                        intrusions.append(
                            Detection(
                                label="intrusion",
                                confidence=float(box.conf),
                                bbox=[x1, y1, x2, y2],
                                metadata={
                                    "zone": zone_name,
                                    "foot_point": [foot_x, foot_y],
                                },
                            )
                        )

            triggered = len(intrusions) > 0
            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=intrusions,
                message=(
                    f"위험구역 침입 {len(intrusions)}건 감지"
                    if triggered
                    else "침입 없음"
                ),
            )
        except Exception as e:
            logger.exception("VirtualFenceDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
