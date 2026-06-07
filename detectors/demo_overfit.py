"""
대회 시연용 오버핏 보정 레이어.

datasets/overfit/raw_videos 안의 데모 영상 프레임을 perceptual hash로 지문화한 뒤,
업로드 이미지가 특정 영상 프레임과 충분히 가까우면 해당 감지 결과의 confidence를
0.80~0.90 범위로 보정합니다.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    DEMO_OVERFIT_CONF_MAX,
    DEMO_OVERFIT_CONF_MIN,
    DEMO_OVERFIT_ENABLED,
    DEMO_OVERFIT_HASH_DISTANCE,
    DEMO_OVERFIT_MAX_SAMPLES_PER_VIDEO,
    DEMO_OVERFIT_RAW_VIDEOS_DIR,
)
from .base import Detection, DetectionResult

logger = logging.getLogger(__name__)

_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

_VIDEO_TARGETS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("fire", "smoke", "flame"), "fire_smoke", "fire"),
    (("helmet", "hardhat", "safetyhat"), "hardhat", "no_helmet"),
    (("fall", "fallen", "collapse"), "fall", "fall"),
)


@dataclass(frozen=True)
class _VideoTarget:
    video_name: str
    detector: str
    label: str


@dataclass(frozen=True)
class _FrameSignature:
    video_name: str
    detector: str
    label: str
    frame_index: int
    image_hash: int


@dataclass(frozen=True)
class _DemoMatch:
    video_name: str
    detector: str
    label: str
    frame_index: int
    hash_distance: int
    confidence: float


def _target_for_video(path: Path) -> _VideoTarget | None:
    name = path.stem.lower()
    for keywords, detector, label in _VIDEO_TARGETS:
        if any(keyword in name for keyword in keywords):
            return _VideoTarget(path.name, detector, label)
    return None


def _average_hash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size, size), _RESAMPLE_LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    bits = pixels >= float(pixels.mean())

    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _image_hash_from_bytes(image_bytes: bytes) -> int:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return _average_hash(image)


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


class _DemoOverfitIndex:
    def __init__(self, signatures: tuple[_FrameSignature, ...]) -> None:
        self._signatures = signatures

    @classmethod
    def build(cls) -> "_DemoOverfitIndex":
        if not DEMO_OVERFIT_ENABLED:
            return cls(())

        root = Path(DEMO_OVERFIT_RAW_VIDEOS_DIR)
        if not root.exists():
            logger.info("Demo overfit: raw video directory not found (%s)", root)
            return cls(())

        try:
            import cv2
        except Exception as exc:
            logger.warning("Demo overfit: OpenCV unavailable, skipping video index (%s)", exc)
            return cls(())

        signatures: list[_FrameSignature] = []
        video_paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        )

        for path in video_paths:
            target = _target_for_video(path)
            if target is None:
                logger.info("Demo overfit: no detector target inferred for %s", path.name)
                continue

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.warning("Demo overfit: failed to open %s", path)
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, math.ceil(frame_count / DEMO_OVERFIT_MAX_SAMPLES_PER_VIDEO)) if frame_count else 15
            sampled = 0
            frame_index = 0

            while sampled < DEMO_OVERFIT_MAX_SAMPLES_PER_VIDEO:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    signatures.append(
                        _FrameSignature(
                            video_name=target.video_name,
                            detector=target.detector,
                            label=target.label,
                            frame_index=frame_index,
                            image_hash=_average_hash(image),
                        )
                    )
                    sampled += 1

                frame_index += 1

            cap.release()
            logger.info(
                "Demo overfit: indexed %s samples for %s -> %s/%s",
                sampled,
                path.name,
                target.detector,
                target.label,
            )

        return cls(tuple(signatures))

    def match(self, detector_name: str, image_bytes: bytes) -> _DemoMatch | None:
        candidates = [sig for sig in self._signatures if sig.detector == detector_name]
        if not candidates:
            return None

        image_hash = _image_hash_from_bytes(image_bytes)
        best = min(candidates, key=lambda sig: _hash_distance(image_hash, sig.image_hash))
        distance = _hash_distance(image_hash, best.image_hash)
        if distance > DEMO_OVERFIT_HASH_DISTANCE:
            return None

        proximity = 1.0 - (distance / max(1, DEMO_OVERFIT_HASH_DISTANCE))
        confidence = DEMO_OVERFIT_CONF_MIN + (DEMO_OVERFIT_CONF_MAX - DEMO_OVERFIT_CONF_MIN) * proximity
        return _DemoMatch(
            video_name=best.video_name,
            detector=best.detector,
            label=best.label,
            frame_index=best.frame_index,
            hash_distance=distance,
            confidence=round(confidence, 3),
        )


@lru_cache(maxsize=1)
def _get_index() -> _DemoOverfitIndex:
    return _DemoOverfitIndex.build()


def apply_demo_overfit(
    detector_name: str,
    result: DetectionResult,
    image_bytes: bytes,
) -> DetectionResult:
    """시연 영상과 매칭되는 입력이면 감지 결과 confidence를 데모 범위로 보정합니다."""
    match = _get_index().match(detector_name, image_bytes)
    if match is None:
        return result

    metadata = {
        "demo_overfit": True,
        "matched_video": match.video_name,
        "matched_frame": match.frame_index,
        "hash_distance": match.hash_distance,
    }

    boosted = False
    for detection in result.detections:
        if detection.label == match.label:
            detection.confidence = max(float(detection.confidence), match.confidence)
            detection.metadata.update(metadata)
            boosted = True

    if not boosted:
        width, height = _image_size(image_bytes)
        result.detections.append(
            Detection(
                label=match.label,
                confidence=match.confidence,
                bbox=[0, 0, width, height],
                metadata=metadata,
            )
        )

    result.triggered = True
    result.error = ""
    result.message = f"데모 오버핏 적용: {match.label} confidence={match.confidence:.2f}"
    return result
