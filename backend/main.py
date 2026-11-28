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
# [0] 추론 정합성 및 앙상블 보정 설정 스위치 [각주 1]
# ==============================================================================
# dataset.py 기준: {'real': 0, 'fake': 1} -> 가짜(Fake) 확률을 추출할 인덱스
FAKE_CLASS_INDEX = 1

# 가중치(.pth) 로드 시 모델 레이어 변수명과 체크포인트 키의 1:1 일치를 엄격히 검증
STRICT_WEIGHT_LOADING = True

# 앙상블 가중치 및 고신뢰 보정(High-Confidence Boost) 하이퍼파라미터 [각주 6]
ENSEMBLE_WEIGHTS = {
    "spatial": 0.35,      # RGB 공간 영역 모델 기본 비중
    "frequency": 0.20,    # 2D DCT 주파수 모델 기본 비중 (JPEG 압축 오탐 억제를 위해 0.3 -> 0.2 하향)
    "dual_stream": 0.45,  # 공간+주파수 1792차원 융합 모델 기본 비중
}
HIGH_CONFIDENCE_THRESHOLD = 0.90  # 단일 모델이 "매우 확실한 가짜"로 판단하는 기준 확률 (90%)
MIN_SUPPORT_PROB = 0.10           # 다른 모델 중 최소 1개 이상이 뒷받침해야 하는 최소 확률 (10%) [각주 7]
DECISION_THRESHOLD = 45.0         # 최종 FAKE / REAL 판별 임계값 (%)


# ==============================================================================
# [1] 경로 및 SQLite 데이터베이스 모듈 [각주 2]
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # 현재 실행 중인 main.py의 절대 경로
PARENT_DIR = os.path.dirname(BASE_DIR)                     # 프로젝트 상위 루트 경로
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")   # 탐지 이력 저장용 SQLite 파일 경로
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")         # 학습된 .pth 가중치 파일 기본 디렉토리


def _resolve_frontend_dir() -> str:
    """
    index.html, style.css, app.js가 위치한 프론트엔드 폴더를 자동 탐색합니다.
    - 단일 폴더 배포 환경과 backend/·frontend/ 분리 구조를 모두 지원합니다.
    """
    frontend_sibling = os.path.join(PARENT_DIR, "frontend")
    for candidate in (BASE_DIR, frontend_sibling, PARENT_DIR):
        if os.path.exists(os.path.join(candidate, "index.html")):
            return candidate
    return BASE_DIR


FRONTEND_DIR = _resolve_frontend_dir()


def init_db(db_path: str = DB_PATH) -> None:
    """탐지 이력을 저장할 SQLite 테이블(detection_logs)을 초기화합니다."""
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
        # 예외 발생 여부와 무관하게 SQLite 파일 잠금(Lock)을 안전하게 해제
        conn.close()


def save_result_to_db(
    filename: str,
    prediction: str,
    fake_probability: float,
    db_path: str = DB_PATH,
) -> None:
    """판별 결과를 detection_logs 테이블에 안전하게 기록합니다 (파라미터 바인딩 적용)."""
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
        cv2.resize(256x256) -> RGB2GRAY -> cv2.dct -> np.log1p -> Min-Max(0~1)
        """
        img_resized = cv2.resize(img_np, target_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_float = np.float32(gray)
        dct_coefficients = cv2.dct(gray_float)
        dct_log = np.log1p(np.abs(dct_coefficients))

        min_val, max_val = np.min(dct_log), np.max(dct_log)
        if max_val - min_val > 1e-6:
            dct_norm = (dct_log - min_val) / (max_val - min_val)
        else:
            dct_norm = np.zeros_like(dct_log)

        return torch.tensor(dct_norm, dtype=torch.float32).unsqueeze(0)


# 2) composite_detector_models.py 우선 임포트
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
    version="1.3.0",
)

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
    """saved_models, 현재 폴더, 상위 폴더 순으로 .pth 가중치를 탐색해 로드합니다."""
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

# 추론 시 Dropout 비활성화 및 BatchNorm 러닝 통계량 고정을 위해 평가 모드(.eval()) 전환
spatial_model.eval()
freq_model.eval()
dual_model.eval()

rgb_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """PIL 이미지를 공간(RGB) 텐서와 2D DCT 주파수 텐서로 변환합니다."""
    rgb_tensor = rgb_transform(image).unsqueeze(0).to(device)

    img_np = np.array(image)
    dct_single = generate_dct_frequency_map
