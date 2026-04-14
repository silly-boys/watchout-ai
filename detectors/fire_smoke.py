"""
화재·연기 감지기 — OpenCV 색상 분석 기반
 - 화재: RGB/YCbCr 색공간에서 fire-pixel 판정 → 연결 성분 분석
 - 연기: HSV 저채도·중간 밝기 영역 + Laplacian 저분산(균질 텍스처)
 - 별도 모델 불필요, 즉시 동작
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

from config import FIRE_CONF
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)

# ── 파라미터 ────────────────────────────────────────────────
FIRE_PIXEL_RATIO = 0.005     # 전체 픽셀 대비 fire-pixel 비율 (0.5%)
SMOKE_PIXEL_RATIO = 0.12     # 전체 픽셀 대비 smoke-pixel 비율 (12%)
MIN_BLOB_AREA_FIRE = 800     # 화재 최소 연결 성분 면적 (픽셀²)
MIN_BLOB_AREA_SMOKE = 6000   # 연기 최소 연결 성분 면적 (픽셀²) — 충분히 큰 연기 덩어리만
MAX_DETECTIONS = 5           # 최대 반환 감지 건수 (노이즈 억제)
SMOKE_LAP_VAR_MAX = 15.0     # Laplacian 분산 상한 — 매우 낮아야 균질한 연기로 판정


def _fire_mask(bgr: np.ndarray) -> np.ndarray:
    """RGB + HSV + YCbCr 삼중 조건 AND로 fire-pixel 마스크 생성.
    피부톤 오탐을 줄이기 위해 세 조건 모두 충족 시에만 fire-pixel로 판정.
    """
    r = bgr[:, :, 2].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    b = bgr[:, :, 0].astype(np.float32)

    # RGB: R 강한 우위, B 매우 낮음 (피부는 B가 상대적으로 높음)
    cond_rgb = (r > 200) & (r > g * 1.5) & (b < 80) & (g > 50)

    # HSV: 높은 채도(S>150) + 오렌지-빨강 색조(H<25) + 높은 밝기
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cond_hsv = (h < 25) & (s > 150) & (v > 160)

    # YCbCr: 낮은 청색 성분, 높은 적색 성분
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cb = ycbcr[:, :, 2].astype(np.float32)
    cr = ycbcr[:, :, 1].astype(np.float32)
    cond_ycc = (cb < 110) & (cr > 160)

    mask = (cond_rgb & cond_hsv & cond_ycc).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _smoke_mask(bgr: np.ndarray) -> np.ndarray:
    """HSV 저채도 + Laplacian 저분산으로 smoke-pixel 마스크 생성."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    cond = (s < 55) & (v > 70) & (v < 210)
    mask = cond.astype(np.uint8) * 255

    # 균질 텍스처 필터: 블록별 Laplacian 분산이 낮은 영역만 유지
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = np.abs(lap_var)
    smooth_var = cv2.GaussianBlur(lap_var, (21, 21), 0)
    uniform_mask = (smooth_var < SMOKE_LAP_VAR_MAX).astype(np.uint8) * 255

    combined = cv2.bitwise_and(mask, uniform_mask)
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _blobs_to_detections(mask: np.ndarray, label: str, img_area: int) -> list[Detection]:
    """연결 성분 분석으로 감지 결과 목록 생성."""
    pixel_count = int(np.count_nonzero(mask))
    ratio = pixel_count / img_area

    threshold = FIRE_PIXEL_RATIO if label == "fire" else SMOKE_PIXEL_RATIO
    if ratio < threshold:
        return []

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_labels = len(stats)
    detections: list[Detection] = []

    min_area = MIN_BLOB_AREA_FIRE if label == "fire" else MIN_BLOB_AREA_SMOKE
    blobs = sorted(
        [stats[i] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area],
        key=lambda s: s[cv2.CC_STAT_AREA],
        reverse=True,
    )[:MAX_DETECTIONS]

    conf = min(1.0, ratio * 20)
    for s in blobs:
        x = int(s[cv2.CC_STAT_LEFT])
        y = int(s[cv2.CC_STAT_TOP])
        w = int(s[cv2.CC_STAT_WIDTH])
        h = int(s[cv2.CC_STAT_HEIGHT])
        detections.append(
            Detection(
                label=label,
                confidence=round(conf, 3),
                bbox=[x, y, x + w, y + h],
                metadata={"pixel_ratio": round(ratio, 4)},
            )
        )
    return detections


class FireSmokeDetector(BaseDetector):
    name = "fire_smoke"

    def detect(self, image_bytes: bytes) -> DetectionResult:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_area = bgr.shape[0] * bgr.shape[1]

            fire_det = _blobs_to_detections(_fire_mask(bgr), "fire", img_area)
            # 연기는 화재와 동반될 때만 보고 (정적 이미지에서 연기 단독 판별 불안정)
            smoke_det = _blobs_to_detections(_smoke_mask(bgr), "smoke", img_area) if fire_det else []
            all_det = fire_det + smoke_det

            triggered = len(all_det) > 0
            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=all_det,
                message=(
                    f"화재·연기 감지 {len(all_det)}건"
                    if triggered
                    else "이상 없음"
                ),
            )
        except Exception as e:
            logger.exception("FireSmokeDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
