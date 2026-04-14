"""
설비 이상 감지기 — SSIM + 픽셀 차이 기반 레퍼런스 비교
 - POST /anomaly/reference 로 '정상' 기준 이미지를 등록
 - 이후 프레임마다 SSIM 유사도 + 평균 픽셀 차이로 이상 판단
 - 모델 학습 불필요, 레퍼런스 등록 후 즉시 사용 가능
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from config import ANOMALY_DIFF_THRESHOLD, ANOMALY_SSIM_THRESHOLD, MODELS_DIR
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)

REFERENCE_PATH = MODELS_DIR / "anomaly_reference.npy"
_COMPARE_SIZE = (256, 256)


def _load_reference() -> np.ndarray | None:
    if REFERENCE_PATH.exists():
        return np.load(str(REFERENCE_PATH))
    return None


def _save_reference(img_gray: np.ndarray) -> None:
    np.save(str(REFERENCE_PATH), img_gray)


def set_reference(image_bytes: bytes) -> str:
    """레퍼런스 이미지를 등록합니다."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # 그레이스케일
    img_resized = img.resize(_COMPARE_SIZE)
    arr = np.array(img_resized, dtype=np.uint8)
    _save_reference(arr)
    logger.info("AnomalyDetector: 레퍼런스 이미지 등록 완료 (%s)", REFERENCE_PATH)
    return "레퍼런스 등록 완료"


class AnomalyDetector(BaseDetector):
    name = "anomaly"

    def __init__(self) -> None:
        self._ref = _load_reference()
        if self._ref is not None:
            logger.info("AnomalyDetector: 저장된 레퍼런스 로드 완료")
        else:
            logger.warning("AnomalyDetector: 레퍼런스 없음 — POST /anomaly/reference 로 등록 필요")

    def reload(self) -> None:
        self._ref = _load_reference()

    def detect(self, image_bytes: bytes) -> DetectionResult:
        if self._ref is None:
            return DetectionResult(
                detector=self.name,
                triggered=False,
                message="레퍼런스 미등록 — POST /anomaly/reference 로 정상 이미지 등록 필요",
            )
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            curr = np.array(img.resize(_COMPARE_SIZE), dtype=np.uint8)

            # SSIM (1.0 = 동일, 낮을수록 이상)
            score_ssim, diff_map = ssim(self._ref, curr, full=True)
            score_ssim = float(score_ssim)

            # 픽셀 평균 절대 차이
            diff_abs = np.abs(curr.astype(np.float32) - self._ref.astype(np.float32))
            mean_diff = float(diff_abs.mean())

            # 이상 판단: SSIM 낮거나 픽셀 차이 큼
            triggered = (score_ssim < ANOMALY_SSIM_THRESHOLD) or (mean_diff > ANOMALY_DIFF_THRESHOLD)

            detections: list[Detection] = []
            if triggered:
                # 이상 핫스팟: diff_map이 낮은(=많이 다른) 좌표
                diff_norm = ((1.0 - diff_map) * 255).astype(np.uint8)
                idx = np.unravel_index(diff_norm.argmax(), diff_norm.shape)
                # 원본 이미지 좌표로 변환
                orig_w, orig_h = img.size
                hx = int(idx[1] / _COMPARE_SIZE[0] * orig_w)
                hy = int(idx[0] / _COMPARE_SIZE[1] * orig_h)

                detections.append(
                    Detection(
                        label="anomaly",
                        confidence=round(1.0 - score_ssim, 3),
                        metadata={
                            "ssim": round(score_ssim, 4),
                            "mean_diff": round(mean_diff, 2),
                            "hotspot": [hx, hy],
                        },
                    )
                )

            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=detections,
                message=(
                    f"설비 이상 감지 (SSIM={score_ssim:.3f}, diff={mean_diff:.1f})"
                    if triggered
                    else f"설비 정상 (SSIM={score_ssim:.3f}, diff={mean_diff:.1f})"
                ),
            )
        except Exception as e:
            logger.exception("AnomalyDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
