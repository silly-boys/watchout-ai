import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import ALERTS_DIR, MAX_SAVED_IMAGES, UPLOADS_DIR
from detectors.anomaly import set_reference
from detectors import (
    AnomalyDetector,
    FallDetector,
    FireSmokeDetector,
    HardhatDetector,
    VirtualFenceDetector,
)
from detectors.virtual_fence import delete_zone, get_zones, set_zone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DETECTOR_LABELS = {
    "fire_smoke":     "🔥 화재·연기",
    "hardhat":        "⛑️  안전모 미착용",
    "virtual_fence":  "🚧 위험구역 침입",
    "fall":           "🆘 낙상·쓰러짐",
    "anomaly":        "⚠️  설비 이상",
}

DETECTION_ICONS = {
    "fire":       "🔥",
    "smoke":      "💨",
    "no_helmet":  "👷",
    "intrusion":  "🚷",
    "fall":       "💥",
    "stillness":  "🛑",
    "anomaly":    "🔧",
}


def _print_alerts(timestamp: str, alerts: list[str], summary: dict) -> None:
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  [이상 감지] {timestamp}")
    print(sep)
    for name in alerts:
        label = DETECTOR_LABELS.get(name, name)
        info  = summary[name]
        print(f"\n  {label}")
        print(f"  └ {info['message']}")
        for det in info["detections"]:
            icon = DETECTION_ICONS.get(det["label"], "•")
            conf_str = f"conf={det['confidence']:.2f}" if det["confidence"] > 0 else ""
            bbox_str = f"bbox={det['bbox']}" if det["bbox"] else ""
            meta_str = ""
            if det["metadata"]:
                parts = []
                m = det["metadata"]
                if "zone"        in m: parts.append(f"구역={m['zone']}")
                if "reasons"     in m: parts.append(f"사유={m['reasons']}")
                if "ssim"        in m: parts.append(f"SSIM={m['ssim']}")
                if "mean_diff"   in m: parts.append(f"diff={m['mean_diff']}")
                if "hotspot"     in m: parts.append(f"핫스팟={m['hotspot']}")
                if "pixel_ratio" in m: parts.append(f"픽셀비율={m['pixel_ratio']:.2%}")
                if "aspect_ratio"in m: parts.append(f"비율={m['aspect_ratio']:.2f}")
                meta_str = "  " + " | ".join(parts)
            detail = "  ".join(filter(None, [conf_str, bbox_str]))
            print(f"     {icon} {det['label']:12s}  {detail}")
            if meta_str:
                print(f"              {meta_str}")
    print(f"{sep}\n")

executor = ThreadPoolExecutor(max_workers=5)
_detectors: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 시작 ──────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    def _init():
        return {
            "fire_smoke": FireSmokeDetector(),
            "hardhat": HardhatDetector(),
            "virtual_fence": VirtualFenceDetector(),
            "fall": FallDetector(),
            "anomaly": AnomalyDetector(),
        }

    _detectors.update(await loop.run_in_executor(executor, _init))
    logger.info("모든 감지기 초기화 완료")
    yield
    # ── 종료 ──────────────────────────────────────────────────
    executor.shutdown(wait=False)


app = FastAPI(title="WatchOut AI", version="1.0.0", lifespan=lifespan)


# ── 이미지 업로드 & 통합 분석 ─────────────────────────────────
@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await image.read()

    # 이미지 저장 (최근 3개 유지)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ext = Path(image.filename).suffix or ".jpg"
    filename = f"{timestamp}{ext}"
    save_path = UPLOADS_DIR / filename
    save_path.write_bytes(contents)

    saved = sorted(UPLOADS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in saved[MAX_SAVED_IMAGES:]:
        old.unlink()

    # 감지기 병렬 실행
    loop = asyncio.get_event_loop()

    def _run_all():
        results = {}
        for name, detector in _detectors.items():
            results[name] = detector.detect(contents)
        return results

    detection_results = await loop.run_in_executor(executor, _run_all)

    # 결과 직렬화
    alerts = []
    summary = {}
    for name, result in detection_results.items():
        summary[name] = {
            "triggered": result.triggered,
            "message": result.message,
            "error": result.error,
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "bbox": d.bbox,
                    "metadata": d.metadata,
                }
                for d in result.detections
            ],
        }
        if result.triggered:
            alerts.append(name)

    # 이상 발생 시에만 JSON 저장 (트리거된 감지기만 포함)
    if alerts:
        alert_path = ALERTS_DIR / f"{timestamp}.json"
        alert_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "filename": filename,
                    "alerts": alerts,
                    "details": {name: summary[name] for name in alerts},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        logger.warning("[%s] 알림 발생: %s", timestamp, alerts)
        _print_alerts(timestamp, alerts, summary)
    else:
        logger.info("[%s] 이상 없음", timestamp)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "filename": filename,
            "alerts": alerts,
            "summary": summary,
        },
    )


# ── Virtual Fence 관리 API ────────────────────────────────────
class ZoneBody(BaseModel):
    polygon: list[list[int]]  # [[x,y], [x,y], ...]


@app.get("/zones")
async def list_zones():
    return get_zones()


@app.put("/zones/{zone_name}")
async def upsert_zone(zone_name: str, body: ZoneBody):
    if len(body.polygon) < 3:
        raise HTTPException(status_code=400, detail="폴리곤은 최소 3개 꼭짓점 필요")
    set_zone(zone_name, body.polygon)
    return {"status": "ok", "zone": zone_name, "polygon": body.polygon}


@app.delete("/zones/{zone_name}")
async def remove_zone(zone_name: str):
    if not delete_zone(zone_name):
        raise HTTPException(status_code=404, detail="구역을 찾을 수 없습니다.")
    return {"status": "ok", "deleted": zone_name}


# ── 이상 감지 레퍼런스 등록 API ──────────────────────────────
@app.post("/anomaly/reference")
async def register_reference(image: UploadFile = File(...)):
    """정상 설비 이미지를 레퍼런스로 등록합니다."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 가능합니다.")
    contents = await image.read()
    loop = asyncio.get_event_loop()
    msg = await loop.run_in_executor(executor, set_reference, contents)
    # 감지기 내부 레퍼런스도 갱신
    _detectors["anomaly"].reload()
    return {"status": "ok", "message": msg}


# ── 헬스체크 ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detectors": list(_detectors.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
