"""
쓰러짐·낙상 감지기
 - Optical Flow(Farneback)로 프레임 간 움직임 벡터 분석
 - Person bbox 가로/세로 비율 변화로 낙상(눕힘) 판단
 - N프레임 연속 미동 시 장시간 쓰러짐 의심
 - 이전 프레임 상태는 인스턴스가 보유 (서버 재시작 시 초기화)
"""

from __future__ import annotations

import io
import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import (
    FALL_ASPECT_THRESHOLD,
    FALL_FLOW_THRESHOLD,
    FALL_MAX_BBOX_RATIO,
    FALL_STILLNESS_FLOW_MIN,
    FALL_STILLNESS_FRAMES,
    PERSON_CONF,
    PERSON_MODEL,
)
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class PersonState:
    bbox: list[int]
    aspect_ratio: float   # width / height


class FallDetector(BaseDetector):
    name = "fall"

    def __init__(self) -> None:
        from ultralytics import YOLO

        model_path = str(PERSON_MODEL) if PERSON_MODEL.exists() else "yolov8n.pt"
        self._model = YOLO(model_path)

        self._prev_gray: np.ndarray | None = None
        # deque에 (flow_mean, aspect_ratio) 저장 → stillness 판단에 사용
        self._history: deque[dict] = deque(maxlen=FALL_STILLNESS_FRAMES)
        logger.info("FallDetector: 모델 로드 (%s)", model_path)

    def _to_gray(self, img_np: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    def detect(self, image_bytes: bytes) -> DetectionResult:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img)
            gray = self._to_gray(img_np)

            # ── Optical Flow ──────────────────────────────────────────
            flow_mean = 0.0
            if self._prev_gray is not None:
                prev = cv2.resize(self._prev_gray, (gray.shape[1], gray.shape[0]))
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mean = float(magnitude.mean())

            self._prev_gray = gray

            # ── Person 탐지 ───────────────────────────────────────────
            results = self._model(img_np, conf=PERSON_CONF, verbose=False)[0]
            persons: list[PersonState] = []

            img_w = img_np.shape[1]
            for box in results.boxes:
                cls_id = int(box.cls)
                if results.names[cls_id] != "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1 or 1
                # 이미지 너비 대비 bbox가 너무 넓으면 여러 명 겹침으로 판단 → 스킵
                if w / img_w > FALL_MAX_BBOX_RATIO:
                    continue
                persons.append(PersonState(bbox=[x1, y1, x2, y2], aspect_ratio=w / h))

            # ── 히스토리 기록 ─────────────────────────────────────────
            self._history.append({
                "flow_mean": flow_mean,
                "persons": persons,
            })

            detections: list[Detection] = []

            for p in persons:
                reasons: list[str] = []

                # 낙상 판단 1: 가로/세로 비율 (눕힌 상태)
                if p.aspect_ratio >= FALL_ASPECT_THRESHOLD:
                    reasons.append(f"비율={p.aspect_ratio:.2f}")

                # 낙상 판단 2: optical flow 급격한 변화 후 정지
                if (
                    len(self._history) >= 2
                    and self._history[-2]["flow_mean"] > FALL_FLOW_THRESHOLD
                    and flow_mean < 1.0
                ):
                    reasons.append("급정지")

                if reasons:
                    detections.append(
                        Detection(
                            label="fall",
                            confidence=0.0,
                            bbox=p.bbox,
                            metadata={"reasons": reasons, "aspect_ratio": p.aspect_ratio},
                        )
                    )

            # 낙상 판단 3: 움직이다 장시간 미동 (이전에 움직임 있었고 N프레임 연속 정지)
            recent = list(self._history)
            had_movement = any(h["flow_mean"] >= FALL_STILLNESS_FLOW_MIN for h in recent[:-3])
            all_still = all(h["flow_mean"] < 1.5 for h in recent[-3:])
            all_person = all(len(h["persons"]) > 0 for h in recent[-3:])
            if (
                len(self._history) == FALL_STILLNESS_FRAMES
                and had_movement
                and all_still
                and all_person
                and not detections
            ):
                for p in persons:
                    detections.append(
                        Detection(
                            label="stillness",
                            confidence=0.0,
                            bbox=p.bbox,
                            metadata={"reason": f"이전 움직임 후 {FALL_STILLNESS_FRAMES}프레임 연속 미동"},
                        )
                    )

            triggered = len(detections) > 0
            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=detections,
                message=(
                    f"낙상/쓰러짐 {len(detections)}건 감지 (flow={flow_mean:.2f})"
                    if triggered
                    else f"이상 없음 (flow={flow_mean:.2f})"
                ),
            )
        except Exception as e:
            logger.exception("FallDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
