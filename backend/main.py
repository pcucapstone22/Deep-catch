import io
import os
import sqlite3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from torchvision import models, transforms

# ==============================================================================
# [0] 추론 정합성 핵심 설정 스위치 [각주 1]
# ==============================================================================
# dataset.py의 DeepfakePreprocessingDataset 기준: {'real': 0, 'fake': 1}
# Softmax 확률 벡터 [P(real), P(fake)]에서 가짜(Fake) 확률을 추출할 인덱스 번호입니다.
FAKE_CLASS_INDEX = 1

# 가중치(.pth) 로드 시 모델 클래스의 레이어 변수명과 체크포인트 키가 정확히 일치하는지 검증합니다.
STRICT_WEIGHT_LOADING = True


# ==============================================================================
# [1] 경로 및 SQLite 데이터베이스 모듈 [각주 2]
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # 현재 실행 중인 main.py의 절대 경로
PARENT_DIR = os.path.dirname(BASE_DIR)                     # 상위 프로젝트 루트 경로 (backend/와 frontend/ 분리 시 탐색용)
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")   # 탐지 이력이 영구 기록될 SQLite 파일 경로
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")         # 학습 완료된 .pth 가중치 파일 기본 보관 폴더


def _resolve_frontend_dir() -> str:
    """
    index.html, style.css, app.js가 위치한 프론트엔드 폴더를 자동 탐색합니다.
    - 백엔드와 프론트엔드가 한 폴더에 있는 통합 배포와 분리된 디렉토리 구조 모두 대응합니다.
    """
    frontend_sibling = os.path.join(PARENT_DIR, "frontend")
    for candidate in (BASE_DIR, frontend_sibling, PARENT_DIR):
        if os.path.exists(os.path.join(candidate, "index.html")):
            return candidate
    return BASE_DIR


FRONTEND_DIR = _resolve_frontend_dir()


def init_db(db_path: str = DB_PATH) -> None:
    """
    탐지 이력을 저장할 SQLite 테이블(detection_logs)을 초기화합니다.
    - IF NOT EXISTS 구문으로 서버 재시작 시에도 기존 탐지 기록이 삭제되지 않도록 보호합니다.
    """
    abs_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    conn = sqlite3.connect(abs_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_probability REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        # 예외 발생 여부와 관계없이 SQLite 파일 잠금(Database Lock) 방지를 위해 연결 종료
        conn.close()


def save_result_to_db(
    filename: str,
    prediction: str,
    fake_probability: float,
    db_path: str = DB_PATH,
) -> None:
    """
    판별 결과를 detection_logs 테이블에 삽입합니다.
    - SQL 인젝션 공격을 예방하기 위해 문자열 포매팅 대신 파라미터 바인딩(?, ?, ?)을 사용합니다.
    """
    abs_path = os.path.abspath(db_path)
    init_db(abs_path)

    conn = sqlite3.connect(abs_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detection_logs (filename, prediction, fake_probability)
            VALUES (?, ?, ?)
            """,
            (filename, prediction, float(fake_probability)),
        )
        conn.commit()
    finally:
        conn.close()


# ==============================================================================
# [2] 학습 원본 전처리(dataset.py) 및 모델(composite_detector_models.py) 연동 [각주 3]
# ==============================================================================
# 1) dataset.py의 generate_dct_frequency_map 우선 임포트
#    - 학습 데이터셋 생성 시 사용한 함수 자체를 가져와 Training-Serving Skew(학습-추론 불일치)를 원천 차단합니다.
try:
    from dataset import generate_dct_frequency_map
    PREPROCESS_SOURCE = "dataset.py (학습 원본 DCT 전처리 연동)"
except ImportError:
    PREPROCESS_SOURCE = "main.py (dataset.py 동일 수식 폴백)"

    def generate_dct_frequency_map(
        img_np: np.ndarray, target_size: tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """
        dataset.py와 동일한 2D DCT 변환 및 0~1 Min-Max 정규화 폴백 함수: [각주 4]
        1. cv2.resize(256x256) -> 2. RGB2GRAY -> 3. cv2.dct -> 4. np.log1p -> 5. Min-Max(0~1)
        """
        img_resized = cv2.resize(img_np, target_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_float = np.float32(gray)

        # OpenCV 2D DCT 수행 (주파수 도메인 계수 행렬 추출)
        dct_coefficients = cv2.dct(gray_float)

        # log(1 + |x|) 변환으로 저주파-고주파 에너지 편차를 압축하고 음수 발산 방지
        dct_log = np.log1p(np.abs(dct_coefficients))

        # Min-Max 정규화로 모든 계수를 [0.0, 1.0] 구간으로 스케일링
        min_val, max_val = np.min(dct_log), np.max(dct_log)
        if max_val - min_val > 1e-6:
            dct_norm = (dct_log - min_val) / (max_val - min_val)
        else:
            dct_norm = np.zeros_like(dct_log)

        # [256, 256] NumPy 배열에 채널 차원을 추가해 [1, 256, 256] Float Tensor 반환
        return torch.tensor(dct_norm, dtype=torch.float32).unsqueeze(0)


# 2) composite_detector_models.py 우선 임포트 (원본 레이어 변수명 일치 보장)
try:
    from composite_detector_models import (
        SpatialCNNDetector,
        FrequencyNoiseDetector,
        DualStreamDeepfakeDetector,
    )
    MODEL_SOURCE = "composite_detector_models.py (학습 원본 아키텍처 연동)"
except ImportError:
    MODEL_SOURCE = "main.py (내장 폴백 아키텍처)"

    class SpatialCNNDetector(nn.Module):
        """[모델 1] RGB 공간 영역 탐지 모델 (EfficientNet-B0 기반 1280차원 특징 분류)"""

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            self.out_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.out_features, num_classes),
            )

        def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.backbone(rgb_image))

    class FrequencyNoiseDetector(nn.Module):
        """[모델 2] 2D DCT 주파수 노이즈 탐지 모델 (4단계 경량 CNN + 512차원 특징 분류)"""

        def __init__(self, in_channels: int = 1, num_classes: int = 2):
            super().__init__()
            self.lightweight_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        def forward(self, freq_image: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.lightweight_cnn(freq_image))

    class DualStreamDeepfakeDetector(nn.Module):
        """[모델 3] 공간(EfficientNet-B0, 1280) + 주파수(ResNet-18, 512) 특징 융합 모델"""

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.spatial_stream = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            in_features_eff = self.spatial_stream.classifier[1].in_features
            self.spatial_stream.classifier = nn.Identity()

            self.freq_stream = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
            # 1채널 DCT 입력을 받기 위해 ResNet-18의 첫 번째 Conv 레이어를 1채널 입력으로 재정의
            self.freq_stream.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            in_features_res = self.freq_stream.fc.in_features
            self.freq_stream.fc = nn.Identity()

            fusion_dim = in_features_eff + in_features_res
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes),
            )

        def forward(
            self, rgb_image: torch.Tensor, dct_image: torch.Tensor
        ) -> torch.Tensor:
            spatial_features = self.spatial_stream(rgb_image)
            freq_features = self.freq_stream(dct_image)
            fused_features = torch.cat((spatial_features, freq_features), dim=1)
            return self.classifier(fused_features)


# ==============================================================================
# [3] FastAPI 서버 및 모델 가중치 초기화 [각주 5]
# ==============================================================================
app = FastAPI(
    title="Deep-catch Backend API",
    description="공간(RGB) 및 2D DCT 주파수 3종 모델 앙상블 딥페이크 탐지 API",
    version="1.2.0",
)

# 외부 도메인(GitHub Codespaces, ngrok 등) 및 분리 포트에서 API 호출을 허용하기 위한 CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db(DB_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spatial_model = SpatialCNNDetector(num_classes=2).to(device)
freq_model = FrequencyNoiseDetector(in_channels=1, num_classes=2).to(device)
dual_model = DualStreamDeepfakeDetector(num_classes=2).to(device)


def load_weights_safely(model: nn.Module, filenames: list[str]) -> bool:
    """
    saved_models, 현재 폴더, 상위 폴더 순으로 .pth 가중치를 탐색해 로드합니다.
    - 1KB 미만 손상/포인터 파일 사전 차단, DataParallel('module.') 접두사 제거, strict=True 검증 수행
    """
    search_dirs = [MODEL_DIR, BASE_DIR, PARENT_DIR]
    for filename in filenames:
        for folder in search_dirs:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                if os.path.getsize(path) < 1024:
                    print(f"[오류] 가중치 파일({path}) 손상 의심 (크기: {os.path.getsize(path)}B)")
                    return False

                try:
                    ckpt = torch.load(path, map_location=device)
                    state_dict = (
                        ckpt["model_state_dict"]
                        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                        else ckpt
                    )
                    cleaned_state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }
                    model.load_state_dict(
                        cleaned_state_dict, strict=STRICT_WEIGHT_LOADING
                    )
                    print(f"[백엔드] 가중치 정밀 로드 완료: {path}")
                    return True
                except Exception as exc:
                    print(f"[오류] 가중치({path}) 로드 실패: {exc}")
                    return False

    print(f"[안내] {filenames[0]} 파일이 없어 초기 가중치 상태로 실행됩니다.")
    return False


WEIGHTS_STATUS = {
    "spatial": load_weights_safely(spatial_model, ["best_Spatial.pth", "spatial_model.pth"]),
    "frequency": load_weights_safely(freq_model, ["best_Frequency.pth", "frequency_model.pth"]),
    "dual_stream": load_weights_safely(dual_model, ["best_DualStream.pth", "dual_stream_model.pth"]),
}
print(
    f"[시스템] 모델: {MODEL_SOURCE} | 전처리: {PREPROCESS_SOURCE} | 가중치: {WEIGHTS_STATUS}"
)

# 추론 시 Dropout 비활성화 및 BatchNorm 통계량 고정을 위해 반드시 평가 모드(.eval()) 전환
spatial_model.eval()
freq_model.eval()
dual_model.eval()

# train_all.py 및 dataset.py와 동일한 RGB 정규화 파이프라인
rgb_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    업로드된 PIL 이미지를 공간(RGB) 텐서와 주파수(2D DCT) 텐서로 변환합니다. [각주 4]
    - dataset.py의 DeepfakePreprocessingDataset.__getitem__과 동일한 순서를 따릅니다.
    """
    # 1. 공간 스트림 입력: [3, 256, 256] -> 배치 차원 추가 [1, 3, 256, 256]
    rgb_tensor = rgb_transform(image).unsqueeze(0).to(device)

    # 2. 주파수 스트림 입력: PIL -> NumPy -> 0~1 정규화 DCT 맵 [1, 256, 256] -> 배치 차원 추가 [1, 1, 256, 256]
    img_np = np.array(image)
    dct_single = generate_dct_frequency_map(img_np, target_size=(256, 256))
    dct_tensor = dct_single.unsqueeze(0).to(device)

    return rgb_tensor, dct_tensor


# ==============================================================================
# [4] REST API 엔드포인트 및 프론트엔드 정적 파일 서빙 [각주 6, 각주 7]
# ==============================================================================
@app.get("/api/health")
def health_check():
    """서버 가동 상태, 모델/전처리 모듈 연동 출처, 가중치 로드 여부를 진단합니다."""
    return {
        "status": "online",
        "service": "Deep-catch Backend API",
        "device": str(device),
        "model_source": MODEL_SOURCE,
        "preprocess_source": PREPROCESS_SOURCE,
        "weights_loaded": WEIGHTS_STATUS,
        "fake_class_index": FAKE_CLASS_INDEX,
    }


@app.post("/api/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """업로드된 이미지를 3종 AI 모델로 앙상블 추론하고 탐지 결과를 DB에 저장합니다."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="이미지 파일(JPG, PNG, WEBP)만 업로드할 수 있습니다."
        )

    try:
        raw_bytes = await file.read()
        # 투명 채널(RGBA) 또는 흑백(L) 이미지를 3채널 RGB로 통일해 텐서 형태 불일치 방지
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail="이미지 파일을 읽을 수 없습니다. 파일 손상 여부를 확인하세요."
        )

    rgb_tensor, dct_tensor = preprocess_image(image)

    # 추론 시 기울기(Gradient) 계산 그래프 생성을 차단해 메모리 절약 및 추론 속도 향상
    with torch.no_grad():
        p_spatial = F.softmax(spatial_model(rgb_tensor), dim=1)[0][FAKE_CLASS_INDEX].item()
        p_freq = F.softmax(freq_model(dct_tensor), dim=1)[0][FAKE_CLASS_INDEX].item()
        p_dual = F.softmax(dual_model(rgb_tensor, dct_tensor), dim=1)[0][FAKE_CLASS_INDEX].item()

    spatial_pct = round(p_spatial * 100.0, 2)
    freq_pct = round(p_freq * 100.0, 2)
    dual_pct = round(p_dual * 100.0, 2)

    # 가중 소프트 보팅 (Spatial 0.3 : Frequency 0.3 : DualStream 0.4) [각주 6]
    ensemble_prob = round(
        ((p_spatial * 0.3) + (p_freq * 0.3) + (p_dual * 0.4)) * 100.0, 2
    )
    is_fake = bool(ensemble_prob >= 50.0)
    prediction = "FAKE (가짜)" if is_fake else "REAL (진짜)"
    safe_filename = file.filename or "uploaded_image.png"

    save_result_to_db(safe_filename, prediction, ensemble_prob, db_path=DB_PATH)

    # 프론트엔드(app.js)가 최상위 필드 또는 details 중첩 객체 어느 쪽을 읽어도 동작하도록 이중 제공
    return {
        "filename": safe_filename,
        "prediction": prediction,
        "is_fake": is_fake,
        "fake_probability": ensemble_prob,
        "spatial_probability": spatial_pct,
        "frequency_probability": freq_pct,
        "dual_probability": dual_pct,
        "details": {
            "spatial": spatial_pct,
            "frequency": freq_pct,
            "dual_stream": dual_pct,
        },
    }


@app.get("/api/logs")
def get_detection_logs(limit: int = 10):
    """SQLite에서 최근 탐지 이력을 최신순(id DESC)으로 조회해 프론트엔드 테이블에 전달합니다."""
    init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT id, filename, prediction, fake_probability, created_at
            FROM detection_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """루트 경로(/) 접속 시 프론트엔드 메인 화면(index.html)을 직접 서빙합니다."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html; charset=utf-8")
    return HTMLResponse(
        "<h3>Deep-catch Backend Running</h3><p>index.html 파일을 찾을 수 없습니다.</p>"
    )


@app.get("/style.css")
def serve_css():
    css_path = os.path.join(FRONTEND_DIR, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css; charset=utf-8")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js")
def serve_js():
    js_path = os.path.join(FRONTEND_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript; charset=utf-8")
    raise HTTPException(status_code=404, detail="app.js not found")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
