"""
안전모 미착용 감지기 — YOLOv8-Pose + 머리 영역 색상 분류
 - yolov8n-pose.pt (ultralytics 공식 자동 다운로드)로 17개 키포인트 탐지
 - nose(0), left_eye(1), right_eye(2) 키포인트로 머리 위치 특정
 - 눈 위~bbox 상단 영역의 HSV 색상 분포로 안전모 착용 판단
 - 안전모 색: 노랑, 흰색, 주황, 빨강, 파랑, 초록 (건설 현장 표준)
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

from config import PERSON_CONF
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)

# ── 안전모 HSV 범위 ──────────────────────────────────────────
# 각 항목: (lower, upper) in HSV (H: 0-180, S: 0-255, V: 0-255)
HARDHAT_COLORS: list[tuple[np.ndarray, np.ndarray, str]] = [
    # 채도(S) 기준을 높여 피부톤(S<120) 오탐 방지
    (np.array([18, 160, 120]), np.array([38, 255, 255]), "yellow"),   # 노랑 (S>160)
    (np.array([0,   0, 210]), np.array([180,  50, 255]), "white"),    # 흰색 (V>210, S<50)
    (np.array([5,  160, 120]), np.array([18, 255, 255]), "orange"),   # 주황 (S>160)
    (np.array([0,  160, 100]), np.array([8, 255, 255]),  "red_l"),    # 빨강(저) (S>160)
    (np.array([170, 160, 100]), np.array([180, 255, 255]), "red_h"),  # 빨강(고) (S>160)
    (np.array([100, 140,  80]), np.array([130, 255, 255]), "blue"),   # 파랑 (S>140)
    (np.array([60,  140,  80]), np.array([85, 255, 255]),  "green"),  # 초록 (S>140)
]
HELMET_PIXEL_RATIO = 0.15    # 머리 영역 내 안전모 색 픽셀 비율 임계값

# 키포인트 인덱스
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2


def _has_hardhat(bgr: np.ndarray) -> bool:
    """BGR 이미지 패치에서 안전모 색상 여부를 판단합니다."""
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    for lo, hi, _ in HARDHAT_COLORS:
        ratio = np.count_nonzero(cv2.inRange(hsv, lo, hi)) / total
        if ratio >= HELMET_PIXEL_RATIO:
            return True
    return False


class HardhatDetector(BaseDetector):
    name = "hardhat"

    def __init__(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO("yolov8n-pose.pt")
        logger.info("HardhatDetector: yolov8n-pose.pt 로드 완료")

    def detect(self, image_bytes: bytes) -> DetectionResult:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h_img, w_img = bgr.shape[:2]

            results = self._model(bgr, conf=PERSON_CONF, verbose=False)[0]
            violations: list[Detection] = []

            if results.keypoints is None or len(results.boxes) == 0:
                return DetectionResult(
                    detector=self.name,
                    triggered=False,
                    message="사람 미탐지",
                )

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                kpts = results.keypoints.xy[i].cpu().numpy()  # (17, 2) — (x, y)
                conf_kpts = results.keypoints.conf[i].cpu().numpy()  # (17,)

                # 눈·코 키포인트 유효성 확인 (신뢰도 > 0.3)
                eye_y_vals = []
                for kp_idx in [KP_LEFT_EYE, KP_RIGHT_EYE]:
                    if conf_kpts[kp_idx] > 0.3:
                        eye_y_vals.append(int(kpts[kp_idx][1]))

                nose_y = int(kpts[KP_NOSE][1]) if conf_kpts[KP_NOSE] > 0.3 else None

                # 머리 영역 추정
                if eye_y_vals:
                    eye_level = min(eye_y_vals)
                elif nose_y is not None:
                    eye_level = nose_y - abs(nose_y - y1) // 3
                else:
                    # 키포인트 없으면 bbox 상단 30% 사용
                    eye_level = y1 + (y2 - y1) // 3

                # 안전모 영역: y1 ~ eye_level, 가로 중앙 60%
                head_top = max(0, y1)
                head_bot = max(head_top + 5, eye_level)
                cx = (x1 + x2) // 2
                hw = (x2 - x1) // 3
                head_left = max(0, cx - hw)
                head_right = min(w_img, cx + hw)

                head_patch = bgr[head_top:head_bot, head_left:head_right]

                if not _has_hardhat(head_patch):
                    violations.append(
                        Detection(
                            label="no_helmet",
                            confidence=float(box.conf),
                            bbox=[x1, y1, x2, y2],
                            metadata={
                                "head_region": [head_left, head_top, head_right, head_bot],
                            },
                        )
                    )

            triggered = len(violations) > 0
            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=violations,
                message=(
                    f"안전모 미착용 {len(violations)}명 감지"
                    if triggered
                    else "전원 안전모 착용 확인"
                ),
            )
        except Exception as e:
            logger.exception("HardhatDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
