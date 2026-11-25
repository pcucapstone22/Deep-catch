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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")


def _resolve_frontend_dir() -> str:
    """index.html, style.css, app.js가 위치한 프론트엔드 폴더를 탐색합니다."""
    for candidate in (BASE_DIR, PARENT_DIR):
        if os.path.exists(os.path.join(candidate, "index.html")):
            return candidate
    return BASE_DIR


FRONTEND_DIR = _resolve_frontend_dir()


def init_db(db_path: str = DB_PATH) -> None:
    """탐지 이력을 저장할 SQLite 테이블을 생성합니다."""
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
        conn.close()


def save_result_to_db(
    filename: str,
    prediction: str,
    fake_probability: float,
    db_path: str = DB_PATH,
) -> None:
    """판별 결과를 detection_logs 테이블에 기록합니다."""
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
# [2] 3종 딥페이크 탐지 모델 아키텍처
# ==============================================================================
def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        net = models.efficientnet_b0(weights=weights)
    except Exception:
        net = models.efficientnet_b0(weights=None)

    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def _build_resnet18(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(weights=None)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


class SpatialCNNDetector(nn.Module):
    """RGB 공간 영역 딥페이크 탐지 모델 (입력: [B, 3, 256, 256])"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = _build_efficientnet_b0(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FrequencyNoiseDetector(nn.Module):
    """2D DCT 주파수 로그 맵 노이즈 탐지 모델 (입력: [B, 1, 256, 256])"""

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
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
    """공간(RGB) + 주파수(2D DCT) 듀얼 스트림 융합 탐지 모델"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.spatial_branch = _build_efficientnet_b0(num_classes=num_classes)
        self.spatial_branch.classifier[1] = nn.Identity()

        self.freq_branch = _build_resnet18(num_classes=num_classes)
        self.freq_branch.fc = nn.Identity()
        self.freq_conv1_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            nn.init.constant_(self.freq_conv1_adapt.weight, 1.0 / 3.0)

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
    """saved_models 또는 현재 폴더에서 .pth 가중치를 탐색해 로드합니다."""
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
    """PIL 이미지를 공간(RGB) 텐서와 주파수(2D DCT) 텐서로 변환합니다."""
    rgb_tensor = rgb_transform(image).unsqueeze(0).to(device)

    gray_256 = np.float32(image.convert("L").resize((256, 256)))
    dct_2d = scipy.fftpack.dct(
        scipy.fftpack.dct(gray_256, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    dct_log = np.log(np.abs(dct_2d) + 1e-5)
    dct_tensor = (
        torch.tensor(dct_log, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    return rgb_tensor, dct_tensor


# ==============================================================================
# [4] REST API 엔드포인트 및 프론트엔드 파일 서빙
# ==============================================================================
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "Deep-catch Backend API",
        "device": str(device),
    }


@app.post("/api/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="이미지 파일(JPG, PNG, WEBP)만 업로드할 수 있습니다."
        )

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail="이미지 파일을 읽을 수 없습니다. 파일 손상 여부를 확인하세요."
        )

    rgb_tensor, dct_tensor = preprocess_image(image)

    with torch.no_grad():
        p_spatial = F.softmax(spatial_model(rgb_tensor), dim=1)[0][1].item()
        p_freq = F.softmax(freq_model(dct_tensor), dim=1)[0][1].item()
        p_dual = F.softmax(dual_model(rgb_tensor, dct_tensor), dim=1)[0][1].item()

    spatial_pct = round(p_spatial * 100.0, 2)
    freq_pct = round(p_freq * 100.0, 2)
    dual_pct = round(p_dual * 100.0, 2)

    ensemble_prob = round(
        ((p_spatial * 0.3) + (p_freq * 0.3) + (p_dual * 0.4)) * 100.0, 2
    )
    is_fake = bool(ensemble_prob >= 50.0)
    prediction = "FAKE (가짜)" if is_fake else "REAL (진짜)"
    safe_filename = file.filename or "uploaded_image.png"

    save_result_to_db(safe_filename, prediction, ensemble_prob, db_path=DB_PATH)

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
