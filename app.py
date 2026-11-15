import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sqlite3
import pandas as pd

# 분류된 3개의 모듈 불러오기
from models import SpatialCNNDetector, FrequencyNoiseDetector, DualStreamDeepfakeDetector
from preprocess import generate_dct_frequency_map
from database import init_db, save_result_to_db

st.set_page_config(page_title="Dual-Stream 딥페이크 탐지기", layout="wide")
init_db()

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dual_model = DualStreamDeepfakeDetector().to(device)
    spatial_model = SpatialCNNDetector().to(device)
    freq_model = FrequencyNoiseDetector().to(device)
    
    # 학습 완료된 가중치(.pth) 파일이 생기면 아래 주석을 해제하고 경로를 맞춰주세요.
    # dual_model.load_state_dict(torch.load("dual_weights.pth", map_location=device))
    # spatial_model.load_state_dict(torch.load("spatial_weights.pth", map_location=device))
    # freq_model.load_state_dict(torch.load("freq_weights.pth", map_location=device))
    
    dual_model.eval()
    spatial_model.eval()
    freq_model.eval()
    return dual_model, spatial_model, freq_model, device

dual_model, spatial_model, freq_model, device = load_models()

def preprocess_for_inference(image: Image.Image):
    # 1. 공간(RGB) 전처리 (preprocess.py의 spatial_transform과 동일)
    rgb_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb_tensor = rgb_transform(image).unsqueeze(0).to(device)
    
    # 2. 주파수(DCT) 전처리 (preprocess.py의 함수 직접 사용 -> 학습과 100% 일치)
    img_np = np.array(image)
    dct_tensor_single = generate_dct_frequency_map(img_np, target_size=(256, 256))
    dct_tensor = dct_tensor_single.unsqueeze(0).to(device)
    
    # 3. 화면 출력용 시각화 이미지 (0~1 정규화된 텐서를 0~255 이미지로 변환)
    dct_vis = (dct_tensor_single.squeeze(0).numpy() * 255).astype(np.uint8)
    
    return rgb_tensor, dct_tensor, dct_vis

# ================= UI 레이아웃 구성 =================
st.title("🔍 공간·주파수 듀얼 스트림 딥페이크 탐지 시스템")

st.sidebar.header("⚙️ 탐지 모델 설정")
model_choice = st.sidebar.radio(
    "사용할 아키텍처를 선택하세요:",
    ("Dual-Stream (공간 + 주파수 결합)", "Spatial CNN (RGB 전용)", "Frequency CNN (DCT 전용)")
)

uploaded_file = st.file_uploader("분석할 얼굴 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    rgb_tensor, dct_tensor, dct_vis = preprocess_for_inference(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 공간 영역 입력 (RGB)")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("2. 주파수 영역 입력 (2D DCT 맵)")
        st.image(dct_vis, use_container_width=True, caption="고주파 노이즈 및 압축 격자 패턴")

    if st.button("🚀 딥페이크 판별 실행", type="primary", use_container_width=True):
        with torch.no_grad():
            if "Dual-Stream" in model_choice:
                logits = dual_model(rgb_tensor, dct_tensor)
            elif "Spatial" in model_choice:
                logits = spatial_model(rgb_tensor)
            else:
                logits = freq_model(dct_tensor)
                
            probs = F.softmax(logits, dim=1)[0]
            fake_prob = probs[1].item() * 100
            prediction = "FAKE (가짜)" if fake_prob >= 50.0 else "REAL (진짜)"
            
            save_result_to_db(uploaded_file.name, prediction, fake_prob)
            
        st.divider()
        if fake_prob >= 50.0:
            st.error(f"🚨 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")
        else:
            st.success(f"✅ 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")
        st.progress(int(fake_prob))

st.divider()
st.subheader("📊 최근 탐지 이력 (SQLite DB 연동)")
conn = sqlite3.connect("deepfake_detector.db")
df_logs = pd.read_sql_query("SELECT * FROM detection_logs ORDER BY id DESC LIMIT 10", conn)
conn.close()

if not df_logs.empty:
    st.dataframe(df_logs, use_container_width=True)
else:
    st.info("아직 탐지 이력이 없습니다. 이미지를 업로드해 첫 분석을 실행해 보세요.")