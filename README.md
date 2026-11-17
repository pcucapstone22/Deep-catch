# 🛡️ Deepfake & AI-Generated Image Detection System

PyTorch 기반의 **공간(Spatial) 및 주파수(Frequency) 듀얼 스트림 앙상블 딥페이크 탐지 시스템**입니다.  
원본 이미지의 시각적 질감 오류뿐만 아니라, 2D DCT(이산 코사인 변환) 주파수 맵에 남는 미세 노이즈 아티팩트까지 동시에 분석하여 AI 생성 및 변조 이미지를 정밀하게 판별합니다.

---

## ✨ 주요 특징 (Key Features)

- **Multi-Stream 앙상블 아키텍처**:
  1. `SpatialCNNDetector` (EfficientNet-B0): RGB 컬러 영역에서 피부 질감, 경계 왜곡 등 시각적 단서 추출
  2. `FrequencyNoiseDetector` (4-Layer Custom CNN): 1채널 2D DCT 주파수 맵에서 고주파 격자 노이즈 패턴 분석
  3. `DualStreamDeepfakeDetector` (EfficientNet-B0 + ResNet-18): 공간 특징(1280D)과 주파수 특징(512D)을 결합(1792D)한 하이브리드 정밀 분석
- **Weighted Soft Voting Ensemble**: 3개 모델의 Softmax 확률을 가중 평균(`0.3 : 0.3 : 0.4`)으로 종합하여 최종 신뢰도 도출
- **End-to-End 자동화 파이프라인**: 필수 패키지 자동 점검(`setup_env.py`), 다양한 압축 포맷(`.zip`, `.tar.gz`, `.gz`) 자동 해제, Google Drive 연동 데이터셋 다운로드 지원
- **이력 관리 및 웹 대시보드**: SQLite 기반 탐지 로그 자동 기록 및 Streamlit 기반 실시간 이미지 판별 웹 UI 제공

---

## 🏗️ 시스템 아키텍처 (System Architecture)

```text
[Input Image] ──┬──> RGB Spatial Tensor [3x256x256] ──┬──> SpatialCNNDetector ───────┐
                │                                     └──> DualStreamDeepfakeDetector ┼──> Soft Voting ──> Final Prediction
                └──> 2D DCT Freq Map [1x256x256] ─────┬──> FrequencyNoiseDetector ───┘      Ensemble         (Real / Fake)
                                                      └──> DualStreamDeepfakeDetector ┘
```

---

## 📂 프로젝트 구조 (Project Structure)

```text
my_deepfake_project/
├── setup_env.py                   # 필수 라이브러리 자동 검사 및 설치 스크립트
├── download_dataset.py            # Google Drive 연동 데이터셋 자동 다운로드 스크립트
├── decompress.py                  # .zip, .tar.gz, .gz 압축 자동 해제 모듈
├── dataset.py                     # RGB 전처리 및 2D DCT 주파수 맵 생성 Custom Dataset
├── composite_detector_models.py   # 3종 탐지 모델 아키텍처 정의 (Spatial, Frequency, DualStream)
├── train_all.py                   # 3종 모델 독립 학습 및 최적 가중치(.pth) 저장 파이프라인
├── ensemble_detector.py           # 3종 모델 가중치 로드 및 앙상블 추론 엔진
├── database.py                    # SQLite 탐지 결과 기록 및 조회 모듈
└── app.py                         # Streamlit 기반 실시간 탐지 웹 대시보드
```

---

## 🛠️ 설치 및 환경 설정 (Installation)

Python **3.10 ~ 3.12** 환경을 권장합니다.

```bash
# 1. 저장소 클론
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# 2. 필수 패키지 자동 점검 및 설치
python setup_env.py
```

**주요 의존성 패키지:** `torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow`, `streamlit`, `pandas`

---

## 🚀 사용 방법 (Usage)

### 1. 데이터셋 준비
`./extracted_data` 경로 아래에 다음과 같이 `real`과 `fake` 폴더로 이미지를 배치합니다. (압축 파일이 있는 경우 `decompress.py` 실행 시 자동 해제됩니다.)

```text
extracted_data/
├── real/   # 진짜 원본 이미지 (.jpg, .png)
└── fake/   # 딥페이크 및 AI 생성 이미지 (.jpg, .png)
```

### 2. 모델 학습 (Training)
3가지 탐지 모델을 순차적으로 학습하고 최고 성능의 가중치 파일(`best_Spatial.pth`, `best_Frequency.pth`, `best_DualStream.pth`)을 저장합니다.

```bash
python train_all.py
```
> **Note**: 로컬 GPU 사양이나 저장공간이 부족한 경우 **Kaggle Notebooks** 또는 **Google Colab**에서 학습 후 생성된 `.pth` 가중치 파일 3개만 프로젝트 루트 폴더로 옮겨 추론에 사용할 수 있습니다.

### 3. 앙상블 추론 테스트 (Inference)
학습된 3개의 가중치를 결합하여 단일 이미지의 진위 여부를 판별하고 결과를 `deepfake_detector.db`에 저장합니다.

```bash
python ensemble_detector.py
```

### 4. 웹 대시보드 실행 (Web UI)
Streamlit 웹 인터페이스를 실행하여 브라우저에서 직접 이미지를 업로드하고 탐지 이력을 조회합니다.

```bash
streamlit run app.py
```

---

## 📊 학습 데이터셋 (Datasets)

### ✅ 실제 적용 데이터셋 (Applied Dataset)
- **[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)**
  - **구성**: CIFAR-10 원본 이미지(`REAL`) 60,000장 + Stable Diffusion v1.4 생성 이미지(`FAKE`) 60,000장 (총 120,000장)
  - **활용**: 본 프로젝트의 3종 탐지 모델(`Spatial`, `Frequency`, `DualStream`) 학습 및 앙상블 파이프라인 검증에 실제 적용됨

### 🔍 확장 학습용 추천 데이터셋 (Recommended Datasets)
- **얼굴 딥페이크 탐지**: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF (v2)](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- **최신 생성 AI 탐지**: [GenImage](https://github.com/GenImage-Dataset/GenImage), [Artifacts Dataset](https://www.kaggle.com/datasets/kazanova/artifacts-dataset)
