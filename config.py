from pathlib import Path

# ── 디렉터리 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
ALERTS_DIR = BASE_DIR / "alerts"

for _d in (MODELS_DIR, UPLOADS_DIR, ALERTS_DIR):
    _d.mkdir(exist_ok=True)

# ── 모델 가중치 경로 ──────────────────────────────────────
# 학습 완료 후 models/ 폴더에 가중치 파일을 배치하세요.
FIRE_SMOKE_MODEL = MODELS_DIR / "fire_smoke.pt"   # D-Fire 파인튜닝 YOLOv8
HARDHAT_MODEL = MODELS_DIR / "hardhat.pt"         # SHWD 학습 YOLOv8
PERSON_MODEL = MODELS_DIR / "yolov8n.pt"          # 기본 YOLOv8 (persons)
ANOMALY_MODEL_DIR = MODELS_DIR / "patchcore"      # PatchCore 저장 경로

# ── 임계값 ───────────────────────────────────────────────
FIRE_CONF = 0.45
HARDHAT_CONF = 0.50
PERSON_CONF = 0.45
FALL_FLOW_THRESHOLD = 8.0       # Optical flow 평균 크기 임계값
FALL_ASPECT_THRESHOLD = 2.0     # bbox 가로/세로 비율 (눕힌 상태 판단)
FALL_MAX_BBOX_RATIO = 0.65      # 이미지 너비 대비 bbox 너비 상한 (겹침 탐지 필터)
FALL_STILLNESS_FRAMES = 4       # N프레임 연속 미동 → 낙상 의심 (5초 간격 * 4 = 20초)
FALL_STILLNESS_FLOW_MIN = 2.0   # 과거 N프레임 중 한번이라도 이 이상 움직인 뒤 정지해야 의심
ANOMALY_DIFF_THRESHOLD = 25.0   # 레퍼런스 대비 픽셀 평균 차이 임계값 (0~255)
ANOMALY_SSIM_THRESHOLD = 0.85   # SSIM 유사도 임계값 (1.0=동일, 낮을수록 이상)

# ── 이미지 보존 개수 ──────────────────────────────────────
MAX_SAVED_IMAGES = 20
