import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sqlite3
import pandas as pd

# ==============================================================================
# [0단계] 백엔드 모듈 연동 (모델 구조 & 데이터베이스)
# ==============================================================================
# - models.py에서 단일 CNN 모델 클래스(SpatialCNNDetector)를 가져옵니다.
# - database.py에서 DB 초기화(init_db) 및 결과 저장(save_result_to_db) 함수를 가져옵니다.
# - 기존 preprocess.py는 수정하지 않고 그대로 유지하기 위해 전처리는 이 파일 내부에서 처리합니다.
from models import SpatialCNNDetector
from database import init_db, save_result_to_db


# ==============================================================================
# [1단계] 웹 페이지 초기 설정 및 SQLite 데이터베이스 준비
# ==============================================================================
# - page_title: 브라우저 상단 탭에 표시될 이름
# - layout="centered": 단일 이미지 분석에 집중할 수 있도록 화면 중앙에 콘텐츠를 모아서 배치
st.set_page_config(page_title="CNN 딥페이크 탐지기", layout="centered")

# 웹 페이지가 처음 열릴 때 deepfake_detector.db 파일과 로그 테이블이 없으면 자동 생성
init_db()


# ==============================================================================
# [2단계] AI 모델 로드 및 메모리 캐싱 (속도 최적화 핵심 구간)
# ==============================================================================
# [왜 @st.cache_resource를 쓰나요?]
# Streamlit은 사용자가 버튼을 누를 때마다 코드 전체를 위에서 아래로 다시 실행합니다.
# 이 데코레이터를 붙여두면 무거운 PyTorch 모델을 최초 1회만 메모리에 올려두고 계속 재사용하므로
# 버튼을 누를 때마다 모델을 다시 불러오는 대기 시간(병목 현상)을 없앨 수 있습니다.
@st.cache_resource
def load_model():
    # 1. 연산 장치 자동 선택: NVIDIA GPU(cuda)가 있으면 GPU 사용, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. EfficientNet-B0 기반의 단일 CNN 모델 뼈대 생성 (0: 진짜, 1: 가짜 -> 총 2개 클래스)
    model = SpatialCNNDetector(num_classes=2).to(device)
    
    # 3. [백엔드 연동 지점]
    # 학습이 완료된 가중치 파일(cnn_weights.pth)이 준비되면 아래 주석(#)을 지워 활성화하세요.
    # 주석이 걸려 있는 동안은 모델이 아직 학습 전(초기 가중치 상태)이므로 테스트용 확률이 나옵니다.
    # model.load_state_dict(torch.load("cnn_weights.pth", map_location=device))
    
    # 4. 추론(평가) 모드 전환: 학습 시에만 작동하는 Dropout 등을 고정하여 일관된 예측 결과 보장
    model.eval()
    return model, device

# 위 함수를 호출하여 메모리에 올라간 모델과 장치 정보를 전역 변수로 받아둠
model, device = load_model()


# ==============================================================================
# [3단계] 이미지 전처리 파이프라인 (사용자 업로드 이미지 -> 모델 입력 텐서 변환)
# ==============================================================================
def preprocess_for_inference(image: Image.Image):
    """
    사용자가 업로드한 일반 이미지(PIL)를 AI 모델이 이해할 수 있는 4차원 숫자 행렬(Tensor)로 변환합니다.
    - 입력: PIL Image 객체 (해상도 자유)
    - 출력: [1, 3, 256, 256] 크기의 PyTorch Tensor (배치크기 1, RGB 3채널, 가로 256, 세로 256)
    """
    # 학습 때 사용한 전처리 규격과 동일하게 크기 조정 및 픽셀값 정규화 수행
    transform = transforms.Compose([
        transforms.Resize((256, 256)),                          # 모델 입력 크기에 맞춰 256x256으로 리사이즈
        transforms.ToTensor(),                                  # 0~255 픽셀값을 0.0~1.0 범위의 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406],        # ImageNet 사전학습 모델 표준 평균/표준편차로 정규화
                             std=[0.229, 0.224, 0.225])
    ])
    
    # PyTorch 모델은 여러 장(Batch)을 한 번에 처리하도록 설계되어 있으므로,
    # 1장의 이미지 맨 앞에 가짜 배치 차원(1)을 추가(.unsqueeze(0))한 뒤 GPU/CPU로 보냄
    rgb_tensor = transform(image).unsqueeze(0).to(device)
    return rgb_tensor


# ==============================================================================
# [4단계] 메인 프론트엔드 UI 구성 (제목 -> 업로드 -> 이미지 확인 -> 판별 실행)
# ==============================================================================
# 4-1. 상단 타이틀 및 부제목 안내
st.title("🔍 CNN 기반 딥페이크 탐지 시스템")
st.caption("EfficientNet-B0 단일 아키텍처를 활용한 이미지 진위 판별 대시보드")

# 4-2. 이미지 업로드 위젯 (jpg, jpeg, png 확장자만 허용)
uploaded_file = st.file_uploader("분석할 얼굴 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

# 사용자가 파일을 성공적으로 업로드했을 때만 아래 블록이 화면에 나타남
if uploaded_file is not None:
    # 투명 배경(RGBA)이나 흑백(L) 이미지가 들어와도 에러가 나지 않도록 무조건 3채널 RGB로 통일
    image = Image.open(uploaded_file).convert("RGB")
    
    # 전처리 함수를 거쳐 모델에 넣을 텐서 준비
    rgb_tensor = preprocess_for_inference(image)
    
    # 업로드된 원본 이미지를 화면에 출력하여 사용자가 제대로 올렸는지 확인시켜 줌
    st.image(image, caption=f"업로드된 이미지: {uploaded_file.name}", use_container_width=True)

    # 4-3. 판별 실행 버튼 (type="primary"로 강조 색상 적용, 가로 너비 꽉 채움)
    if st.button("🚀 딥페이크 판별 실행", type="primary", use_container_width=True):
        
        # torch.no_grad(): 추론 시에는 역전파(학습용 미분 계산)를 꺼서 속도를 높이고 메모리를 절약
        with torch.no_grad():
            # 모델에 이미지 텐서를 통과시켜 원시 점수(Logits: [진짜점수, 가짜점수]) 추출
            logits = model(rgb_tensor)
            
            # Softmax 함수를 적용해 원시 점수를 합이 1(100%)이 되는 확률값으로 변환
            probs = F.softmax(logits, dim=1)[0]
            
            # 1번 인덱스(가짜 클래스)의 확률을 꺼내 백분율(%)로 환산
            fake_prob = probs[1].item() * 100
            
            # 가짜 확률이 50.0% 이상이면 FAKE, 미만이면 REAL로 최종 라벨 결정
            prediction = "FAKE (가짜)" if fake_prob >= 50.0 else "REAL (진짜)"
            
            # 분석이 끝난 파일명, 판정 결과, 가짜 확률을 SQLite DB에 자동 기록
            save_result_to_db(uploaded_file.name, prediction, fake_prob)
            
        # 4-4. 판정 결과 시각적 피드백 출력 (구분선 + 색상 배너 + 확률 게이지 바)
        st.divider()
        if fake_prob >= 50.0:
            # 가짜일 경우 빨간색 경고 배너 출력
            st.error(f"🚨 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")
        else:
            # 진짜일 경우 초록색 성공 배너 출력
            st.success(f"✅ 판정 결과: **{prediction}** (가짜 확률: **{fake_prob:.2f}%**)")
            
        # 가짜 확률 수치를 직관적인 가로 막대 게이지(0~100)로 표시
        st.progress(int(fake_prob))


# ==============================================================================
# [5단계] 하단 대시보드: SQLite DB에 저장된 최근 탐지 이력 실시간 조회
# ==============================================================================
st.divider()
st.subheader("📊 최근 탐지 이력 (SQLite DB 연동)")

# SQLite DB에 접속하여 가장 최근에 판별한 기록 10개를 최신순(ORDER BY id DESC)으로 읽어옴
conn = sqlite3.connect("deepfake_detector.db")
df_logs = pd.read_sql_query("SELECT * FROM detection_logs ORDER BY id DESC LIMIT 10", conn)
conn.close()

# 저장된 이력이 1개라도 있으면 깔끔한 표(DataFrame) 형태로 화면에 출력하고, 없으면 안내 메시지 표시
if not df_logs.empty:
    st.dataframe(df_logs, use_container_width=True)
else:
    st.info("아직 탐지 이력이 없습니다. 이미지를 업로드해 첫 분석을 실행해 보세요.")
