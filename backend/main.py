import io
import os
import sqlite3
import numpy as np
import scipy.fftpack
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
# [1] 경로 및 SQLite 데이터베이스 모듈
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # 현재 실행 중인 main.py가 위치한 절대 디렉토리
PARENT_DIR = os.path.dirname(BASE_DIR)                     # 상위 루트 디렉토리 (backend/와 frontend/가 나란히 있을 때 탐색용)
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")   # 탐지 이력을 영구 저장할 SQLite 데이터베이스 경로
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")         # 학습 완료된 가중치(.pth) 파일이 보관되는 기본 디렉토리


def _resolve_frontend_dir() -> str:
    """
    index.html, style.css, app.js가 위치한 프론트엔드 폴더를 자동 탐색합니다.
    - 단일 폴더 배포 환경(BASE_DIR)과 분리된 저장소 구조(PARENT_DIR/frontend) 모두 대응할 수 있도록 설계되었습니다.
    """
    frontend_sibling = os.path.join(PARENT_DIR, "frontend")
    for candidate in (BASE_DIR, frontend_sibling, PARENT_DIR):
        if os.path.exists(os.path.join(candidate, "index.html")):
            return candidate
    return BASE_DIR


FRONTEND_DIR = _resolve_frontend_dir()


def init_db(db_path: str = DB_PATH) -> None:
    """
    탐지 이력을 저장할 SQLite 테이블(detection_logs)을 초기화합니다. [각주 1]
    - IF NOT EXISTS 구문을 사용하여 서버가 재시작되어도 기존 누적 이력을 보존합니다.
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
        # 예외 발생 여부와 무관하게 DB 커넥션 잠금(Lock)을 안전하게 해제
        conn.close()


def save_result_to_db(
    filename: str,
    prediction: str,
    fake_probability: float,
    db_path: str = DB_PATH,
) -> None:
    """
    앙상블 판별 결과를 detection_logs 테이블에 삽입합니다.
    - SQL 인젝션(SQL Injection) 방어를 위해 파라미터 바인딩(?, ?, ?)을 사용합니다.
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
# [2] 3종 딥페이크 탐지 모델 아키텍처 정의부
# ==============================================================================
def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    """
    공간 영역 특징 추출을 위한 EfficientNet-B0 백본 생성 헬퍼.
    - 오프라인/방화벽 환경에서 사전학습 가중치 다운로드가 차단될 경우를 대비해 weights=None 폴백을 지원합니다.
    """
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        net = models.efficientnet_b0(weights=weights)
    except Exception:
        net = models.efficientnet_b0(weights=None)

    # 기본 ImageNet 1,000 클래스 분류기를 이진 분류(Real=0, Fake=1) 헤드로 교체
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def _build_resnet18(num_classes: int) -> nn.Module:
    """듀얼 스트림 모델의 주파수 브랜치 백본으로 사용할 ResNet-18 생성 헬퍼."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(weights=None)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


class SpatialCNNDetector(nn.Module):
    """
    [모델 1] RGB 공간 영역 딥페이크 탐지 모델 (입력 규격: [Batch, 3, 256, 256]) [각주 2]
    - 인물 경계면의 부자연스러운 블렌딩, 피부 질감 붕괴 등 시각적 위변조 흔적을 포착합니다.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = _build_efficientnet_b0(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FrequencyNoiseDetector(nn.Module):
    """
    [모델 2] 2D DCT 주파수 로그 맵 노이즈 탐지 모델 (입력 규격: [Batch, 1, 256, 256]) [각주 3]
    - GAN 및 Diffusion 모델의 업샘플링 과정에서 고주파 대역에 남는 격자(Checkerboard) 아티팩트를 탐지합니다.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        # Stride=2 합성곱 4단계를 거쳐 256x256 -> 16x16으로 다운샘플링 후 전역 평균 풀링(GAP) 적용
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


class DualStreamDeepfakeDetector(nn.Module):
    """
    [모델 3] 공간(RGB) + 주파수(2D DCT) 듀얼 스트림 융합 탐지 모델 [각주 4]
    - 두 도메인의 특징 벡터를 채널 차원으로 결합(Concatenation)해 단일 도메인 탐지의 한계를 보완합니다.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # 공간 브랜치: 최종 분류 레이어를 Identity()로 치환하여 1280차원 임베딩 벡터 추출
        self.spatial_branch = _build_efficientnet_b0(num_classes=num_classes)
        self.spatial_branch.classifier[1] = nn.Identity()

        # 주파수 브랜치: 512차원 임베딩 벡터 추출
        self.freq_branch = _build_resnet18(num_classes=num_classes)
        self.freq_branch.fc = nn.Identity()

        # 1채널 DCT 맵을 3채널 입력 기반 ResNet-18에 호환시키기 위한 1x1 어댑터 레이어 (초기값 1/3 균등 분배)
        self.freq_conv1_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            nn.init.constant_(self.freq_conv1_adapt.weight, 1.0 / 3.0)

        # 1280(공간) + 512(주파수) = 1792차원 융합 특징을 받아 최종 판정 로짓(Logit) 출력
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, rgb_tensor: torch.Tensor, dct_tensor: torch.Tensor
    ) -> torch.Tensor:
        spatial_feat = self.spatial_branch(rgb_tensor)
        dct_rgb = self.freq_conv1_adapt(dct_tensor)
        freq_feat = self.freq_branch(dct_rgb)
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion_head(fused)


# ==============================================================================
# [3] FastAPI 서버 및 모델 초기화
# ==============================================================================
app = FastAPI(
    title="Deep-catch Backend API",
    description="공간(RGB) 및 2D DCT 주파수 3종 모델 앙상블 딥페이크 탐지 API",
    version="1.0.0",
)

# 분리된 프론트엔드 서버(예: localhost:5500) 및 외부 터널 도메인(ngrok 등)에서의 교차 출처 요청(CORS) 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db(DB_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 가속 우선 할당, 미지원 시 CPU 자동 전환

spatial_model = SpatialCNNDetector(num_classes=2).to(device)
freq_model = FrequencyNoiseDetector(in_channels=1, num_classes=2).to(device)
dual_model = DualStreamDeepfakeDetector(num_classes=2).to(device)


def load_weights_safely(model: nn.Module, filenames: list[str]) -> bool:
    """
    saved_models, 현재 폴더, 상위 폴더 순으로 .pth 체크포인트를 탐색해 안전하게 로드합니다. [각주 5]
    - state_dict 단독 저장 포맷과 {'model_state_dict': ...} 딕셔너리 체크포인트 포맷을 모두 지원합니다.
    """
    search_dirs = [MODEL_DIR, BASE_DIR, PARENT_DIR]
    for filename in filenames:
        for folder in search_dirs:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                try:
                    ckpt = torch.load(path, map_location=device)
                    state_dict = (
                        ckpt["model_state_dict"]
                        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                        else ckpt
                    )
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[백엔드] 가중치 로드 완료: {path}")
                    return True
                except Exception as exc:
                    print(f"[경고] 가중치({path}) 로드 실패, 초기 가중치로 진행합니다: {exc}")
                    return False

    print(f"[안내] {filenames[0]} 파일이 없어 초기 가중치 상태로 실행됩니다.")
    return False


load_weights_safely(spatial_model, ["best_Spatial.pth", "spatial_model.pth"])
load_weights_safely(freq_model, ["best_Frequency.pth", "frequency_model.pth"])
load_weights_safely(dual_model, ["best_DualStream.pth", "dual_stream_model.pth"])

# 추론 시 Dropout 비활성화 및 BatchNorm 실행 통계량 고정을 위해 반드시 평가 모드(.eval()) 전환
spatial_model.eval()
freq_model.eval()
dual_model.eval()

# ImageNet 표준 평균/표준편차 기반 RGB 정규화 파이프라인
rgb_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    업로드된 PIL 이미지를 공간(RGB) 텐서와 2차원 이산 코사인 변환(2D DCT) 주파수 텐서로 변환합니다. [각주 3]
    """
    rgb_tensor = rgb_transform(image).unsqueeze(0).to(device)  # [3, 256, 256] -> 배치 차원 추가 [1, 3, 256, 256]

    # 그레이스케일 변환 후 행(axis=0)과 열(axis=1) 방향으로 직교(ortho) 2D DCT 수행
    gray_256 = np.float32(image.convert("L").resize((256, 256)))
    dct_2d = scipy.fftpack.dct(
        scipy.fftpack.dct(gray_256, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    # 저주파 성분에 집중된 계수 동적 범위를 압축하고 0 로그 발산(log(0))을 막기 위해 1e-5 가산 후 로그 스케일링
    dct_log = np.log(np.abs(dct_2d) + 1e-5)
    dct_tensor = (
        torch.tensor(dct_log, dtype=torch.float32)
        .unsqueeze(0)  # 채널 차원 추가: [1, 256, 256]
        .unsqueeze(0)  # 배치 차원 추가: [1, 1, 256, 256]
        .to(device)
    )
    return rgb_tensor, dct_tensor


# ==============================================================================
# [4] REST API 엔드포인트 및 정적 프론트엔드 서빙
# ==============================================================================
@app.get("/api/health")
def health_check():
    """프론트엔드 연결 상태 표시등(Checking/Online) 및 연산 장치(CUDA/CPU) 확인용 엔드포인트."""
    return {
        "status": "online",
        "service": "Deep-catch Backend API",
        "device": str(device),
    }


@app.post("/api/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """
    이미지 파일을 수신하여 3종 AI 모델 앙상블 추론 및 SQLite 이력 저장을 수행합니다. [각주 6]
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="이미지 파일(JPG, PNG, WEBP)만 업로드할 수 있습니다."
        )

    try:
        raw_bytes = await file.read()
        # RGBA(투명 PNG)나 Grayscale 이미지가 들어와도 3채널 RGB로 통일해 텐서 차원 불일치 방지
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail="이미지 파일을 읽을 수 없습니다. 파일 손상 여부를 확인하세요."
        )

    rgb_tensor, dct_tensor = preprocess_image(image)

    # 추론 시 역전파(Autograd) 그래프 생성을 차단해 VRAM 사용량과 연산 지연 시간 최소화
    with torch.no_grad():
        # 클래스 인덱스 0: Real(진짜), 인덱스 1: Fake(가짜)의 Softmax 확률 추출
        p_spatial = F.softmax(spatial_model(rgb_tensor), dim=1)[0][1].item()
        p_freq = F.softmax(freq_model(dct_tensor), dim=1)[0][1].item()
        p_dual = F.softmax(dual_model(rgb_tensor, dct_tensor), dim=1)[0][1].item()

    spatial_pct = round(p_spatial * 100.0, 2)
    freq_pct = round(p_freq * 100.0, 2)
    dual_pct = round(p_dual * 100.0, 2)

    # 가중 소프트 보팅(Weighted Soft Voting): 단일 도메인(0.3, 0.3)보다 양방향 융합 모델(0.4)에 높은 신뢰도 부여
    ensemble_prob = round(
        ((p_spatial * 0.3) + (p_freq * 0.3) + (p_dual * 0.4)) * 100.0, 2
    )
    is_fake = bool(ensemble_prob >= 50.0)
    prediction = "FAKE (가짜)" if is_fake else "REAL (진짜)"
    safe_filename = file.filename or "uploaded_image.png"

    save_result_to_db(safe_filename, prediction, ensemble_prob, db_path=DB_PATH)

    # 프론트엔드(app.js)의 최상위 키 및 details 중첩 키 규격 모두와 호환되도록 응답 구성
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
    """최근 탐지 이력을 최신순(id DESC)으로 조회하여 프론트엔드 테이블에 전달합니다."""
    init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 딕셔너리 변환이 가능하도록 Row 팩토리 설정
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


# ==============================================================================
# [5] 통합 배포 시 정적 웹 파일(index.html, style.css, app.js) 직접 서빙 라우트
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def serve_index():
    """루트(/) 접속 시 프론트엔드 메인 대시보드 HTML을 반환합니다."""
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
    # 외부 기기 및 터널링 도구 접속을 허용하기 위해 0.0.0.0 인터페이스, 8000 포트로 바인딩
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
