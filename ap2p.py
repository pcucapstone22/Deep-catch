import os
import sys
import subprocess
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import scipy.fftpack
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")


class DualStreamDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.spatial_stream = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        spatial_dim = self.spatial_stream.classifier[1].in_features
        self.spatial_stream.classifier = nn.Identity()

        self.freq_stream = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.freq_stream.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        freq_dim = self.freq_stream.fc.in_features
        self.freq_stream.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb_img, freq_img):
        spatial_feat = self.spatial_stream(rgb_img)
        freq_feat = self.freq_stream(freq_img)
        combined = torch.cat((spatial_feat, freq_feat), dim=1)
        return self.classifier(combined)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result_to_db(filename: str, prediction: str, fake_prob: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detection_logs (filename, prediction, fake_probability, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, prediction, round(fake_prob, 2), now))
    conn.commit()
    conn.close()


def find_weights_path():
    candidates = [
        os.path.join(BASE_DIR, "dual_stream_model.pth"),
        os.path.join(BASE_DIR, "saved_models", "dual_stream_model.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_web_app():
    st.set_page_config(page_title="듀얼 스트림 딥페이크 탐지기", layout="centered")
    init_db()

    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualStreamDeepfakeDetector(num_classes=2).to(device)
        weights_path = find_weights_path()
        if weights_path is not None:
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )
            model.load_state_dict(state_dict)
        model.eval()
        return model, device

    model, device = load_model()

    def extract_dct_map(img_pil: Image.Image) -> torch.Tensor:
        img_gray = img_pil.convert("L").resize((256, 256))
        img_np = np.float32(img_gray)
        dct_np = scipy.fftpack.dct(
            scipy.fftpack.dct(img_np, axis=0, norm="ortho"), axis=1, norm="ortho"
        )
        dct_log = np.log(np.abs(dct_np) + 1e-5)
        return torch.tensor(dct_log, dtype=torch.float32).unsqueeze(0)

    def preprocess_for_inference(image: Image.Image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        rgb_tensor = transform(image).unsqueeze(0).to(device)
        dct_tensor = extract_dct_map(image).unsqueeze(0).to(device)
        return rgb_tensor, dct_tensor

    st.title("🔍 Dual-Stream 기반 딥페이크 탐지 시스템")
    st.caption("공간 영역(RGB) + 주파수 영역(DCT) 결합 아키텍처를 활용한 이미지 진위 판별 대시보드")

    uploaded_file = st.file_uploader(
        "분석할 얼굴 이미지를 업로드하세요 (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        rgb_tensor, dct_tensor = preprocess_for_inference(image)

        st.image(
            image,
            caption=f"업로드된 이미지: {uploaded_file.name}",
            use_container_width=True,
        )

        if st.button("🚀 딥페이크 판별 실행", type="primary", use_container_width=True):
            with torch.no_grad():
                logits = model(rgb_tensor, dct_tensor)
                probs = F.softmax(logits, dim=1)[0]
                fake_prob = probs[1].item() * 100
                prediction = "FAKE (가짜)" if fake_prob >= 50.0 else "REAL (진짜)"
                save_result_to_db(uploaded_file.name, prediction, fake_prob)

            st.divider()
            if fake_prob >= 50.0:
                st.error(f"🚨 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")
            else:
                st.success(f"✅ 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")

            st.progress(min(max(int(fake_prob), 0), 100))

    st.divider()
    st.subheader("📊 최근 탐지 이력 (SQLite DB 연동)")

    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql_query(
        "SELECT * FROM detection_logs ORDER BY id DESC LIMIT 10", conn
    )
    conn.close()

    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("아직 탐지 이력이 없습니다. 이미지를 업로드해 첫 분석을 실행해 보세요.")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        run_web_app()
    else:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
            check=False,
        )
