# 🛡️ Deep-catch : Deepfake & AI-Generated Image Detection System

PyTorch 기반의 **공간(Spatial) 및 주파수(Frequency) 듀얼 스트림 앙상블 딥페이크 탐지 시스템**입니다.  
원본 이미지의 시각적 질감 오류뿐만 아니라, 2D DCT(이산 코사인 변환) 주파수 맵에 남는 미세 노이즈 아티팩트까지 동시에 분석하여 AI 생성 및 변조 이미지를 정밀하게 판별합니다.

---

## 🎯 프로젝트 배경 및 목표 (Background & Goals)

### 📌 배경 및 필요성 (Background & Necessity)
최근 **Stable Diffusion, Midjourney** 등 생성형 AI 기술의 비약적인 발전으로 육안으로 구별하기 어려운 정교한 딥페이크 및 가짜 이미지가 손쉽게 생산되고 있습니다. 이에 따라 가짜 뉴스, 사기, 명예훼손 등 디지털 범죄와 보안 위협이 급증하고 있어 **조작된 이미지를 신속하고 정확하게 판별할 수 있는 탐지 기술의 확보가 시급한 상황**입니다.

### 🚀 프로젝트 목표 (Project Goals)
1. **고도화된 하이브리드 탐지 모델 구축**: 공간(RGB) 영역과 주파수(2D DCT) 영역을 동시에 분석하는 멀티 스트림 앙상블 모델을 개발하여 정밀한 AI 이미지 판별 성능 확보
2. **적대적 데이터 생성 및 취약점 검증**: 직접 가짜 AI 이미지를 생성·수집하여 탐지 모델의 사각지대와 취약점을 반복 검증하고 방어 성능을 고도화
3. **직관적인 통합 웹 서비스 제공**: 전문 지식이 없는 일반 사용자도 이미지를 업로드하는 것만으로 진위 여부와 신뢰도 확률을 직관적으로 확인할 수 있는 통합 웹 대시보드 구현

---

## 📋 프로젝트 주요 내용 (Project Details)

본 프로젝트는 사용자가 업로드한 이미지로부터 핵심 분석 대상을 추출하고, 다각도의 시각적·주파수적 요소를 종합하여 조작 위험도를 산출하는 **End-to-End 딥페이크 판별 플랫폼**을 개발합니다.

1. **분석 대상 추출 및 정밀 전처리 파이프라인**
   - 사용자가 업로드한 이미지에서 주요 분석 대상(얼굴 영역 등)을 추출하고, 이미지 내 미세 노이즈 성분을 정제·정규화하는 전처리 모듈 구현
2. **코어 AI 모델 통합 및 하이브리드 탐지 로직 설계 (`composite_detector_models.py`)**
   - **손현준**이 구현한 **단일 도메인 특화 탐지 모델 2종**(`SpatialCNNDetector`, `FrequencyNoiseDetector`)과 **박준서**가 구현한 **특징 결합형 듀얼 스트림 모델 1종**(`DualStreamDeepfakeDetector`)을 단일 통합 모듈(`composite_detector_models.py`)로 결합
   - 픽셀 단위의 시각적 변조 흔적(RGB 공간 영역)과 2D DCT 주파수 대역의 불일치 여부(주파수 영역)를 독립적·복합적으로 동시 분석하여 조작 확률 도출
3. **자체 생성 데이터 기반 적대적 취약점 검증 및 재학습**
   - 최신 생성형 AI 도구를 활용해 자체적인 고품질 가짜 이미지를 직접 생성하고, 이를 통해 탐지 모델의 오탐 및 취약점을 공격·검증한 뒤 추가 학습을 수행하여 방어 성능 고도화
4. **백엔드 API 및 데이터베이스 로그 체계 구축**
   - 백엔드 API와 SQLite 데이터베이스를 연동하여 실시간 탐지 이력, 모델별 산출 확률, 오탐 사례 데이터를 체계적으로 수집·관리
5. **직관적인 웹 대시보드 및 위험도 경고 시스템**
   - 분석 결과를 종합하여 이미지의 진위 여부를 수치화된 확률(%)로 시각화하는 웹 대시보드를 개발하고, 조작 위험도가 임계치를 초과할 경우 즉시 경고 알림을 제공하여 가짜 정보 및 피싱 피해 예방

---

## 💡 기대 효과 (Expected Effects)

- **가짜 뉴스 및 딥페이크 범죄 사전 예방**: 사용자가 이미지를 열람·활용하기 전 AI 생성 및 조작 위험성을 사전에 인지할 수 있도록 지원하여, 무분별한 허위 정보 수용과 딥페이크를 악용한 사기·피싱 피해를 예방합니다.
- **비판적 미디어 수용 및 정보 검증 문화 정착**: 시각적 정보를 무비판적으로 수용하던 기존의 미디어 소비 방식에서 벗어나, 이미지의 진위를 직접 분석하고 검증하는 과정을 거치게 함으로써 디지털 정보에 대한 신중한 접근을 유도합니다.
- **디지털 보안 인식 제고 및 자율적 신뢰성 판단 지원**: 일상적인 디지털 미디어 사용 환경에서 기본적인 보안 인식을 높이고, 일반 사용자도 직관적인 수치를 바탕으로 정보의 신뢰성을 스스로 판단할 수 있는 실용적 검증 수단으로 활용됩니다.

---

## 👥 팀원 소개 및 역할 (Team & Roles)

| 이름 | 구분 | GitHub | 담당 역할 (Role) |
| :---: | :---: | :--- | :--- |
| **박준서** | 조장 | [@AloneFihser](https://github.com/AloneFihser) | 시스템 전체 설계 및 프로젝트 총괄, AI 딥러닝 (`DualStreamDeepfakeDetector` 구현 및 코어 모델 통합) |
| **손현준** | 팀원 | [@hyenjune](https://github.com/hyenjune) | AI 메인 탐지 모델 구현 (`SpatialCNNDetector`, `FrequencyNoiseDetector`) 및 딥러닝 성능 최적화 |
| **최재영** | 팀원 | [@ggj0916-dotcom](https://github.com/ggj0916-dotcom) | 백엔드 서버 및 데이터베이스(DB) 구축 (백엔드 총괄) |
| **김경호** | 팀원 | [@KKH-930](https://github.com/KKH-930) | 백엔드 및 DB 구현, 검증용 AI 이미지 데이터 생성 |
| **이신석** | 팀원 | [@lesin4929-max](https://github.com/lesin4929-max) | AI 학습 데이터셋 수집 및 이미지 전처리 파이프라인 구축 |
| **송동화** | 팀원 | [@ssdh03](https://github.com/ssdh03) | 웹 프론트엔드 UI/UX 개발, 데이터 전처리 지원 |

---

## ✨ 주요 특징 및 아키텍처 통합 (Key Features & Model Integration)

- **AI 코어 모델 통합 아키텍처 (`composite_detector_models.py`)**:
  팀원별 핵심 AI 모듈(**손현준**: 단일 도메인 모델 2종 + **박준서**: 듀얼 스트림 결합 모델 1종)을 하나의 통합 파일로 병합하여, 단일 모델의 독립 판별과 다중 특징 융합 판별이 동시에 이루어지는 3-모델 앙상블 구조를 완성했습니다.
  1. **`SpatialCNNDetector` (EfficientNet-B0 / RGB 전용 — 구현: 손현준)**: 컬러 원본 이미지에서 피부 질감, 경계 왜곡 등 시각적 공간 단서(`1280D`) 추출 및 판정
  2. **`FrequencyNoiseDetector` (4-Layer Custom CNN / DCT 1채널 전용 — 구현: 손현준)**: 2D DCT 주파수 변환 맵에서 고주파 격자 노이즈 및 압축 아티팩트(`512D`) 추출 및 판정
  3. **`DualStreamDeepfakeDetector` (EfficientNet-B0 + ResNet-18 / 하이브리드 결합 — 구현: 박준서)**: 공간 특징(`1280D`)과 주파수 특징(`512D`)을 직렬 결합(`1792D` Concatenation) 후 심층 분류기(`512D` 병목 압축)를 거쳐 정밀 판정
- **Weighted Soft Voting Ensemble**: 통합된 3개 모델의 Softmax 확률을 가중 평균(`0.3 : 0.3 : 0.4`)으로 종합하여 최종 신뢰도 도출
- **End-to-End 자동화 파이프라인**: 필수 패키지 자동 점검(`setup_env.py`), 다양한 압축 포맷(`.zip`, `.tar.gz`, `.gz`) 자동 해제, Google Drive 연동 데이터셋 다운로드 지원
- **이력 관리 및 웹 대시보드**: SQLite 기반 탐지 로그 자동 기록 및 웹 UI(Streamlit / FastAPI 분리형 지원) 기반 실시간 이미지 판별 제공

---

## 🏗️ 시스템 아키텍처 (System Architecture)

```text
[Input Image] ──┬──> RGB Spatial Tensor [3x256x256] ──┬──> SpatialCNNDetector (1280D) ──────┐
                │                                     └──> DualStreamDeepfakeDetector ──────┼──> Soft Voting ──> Final Prediction
                └──> 2D DCT Freq Map [1x256x256] ─────┬──> FrequencyNoiseDetector (512D) ───┘      Ensemble         (Real / Fake)
                                                      └──> DualStreamDeepfakeDetector (1792D) ┘
```

---

## 📂 프로젝트 구조 (Project Structure)

```text
Deep-catch/
├── setup_env.py                   # 필수 라이브러리 자동 검사 및 설치 스크립트
├── download_dataset.py            # Google Drive 연동 데이터셋 자동 다운로드 스크립트
├── decompress.py                  # .zip, .tar.gz, .gz 압축 자동 해제 모듈
├── dataset.py                     # RGB 전처리 및 2D DCT 주파수 맵 생성 Custom Dataset
├── composite_detector_models.py   # [통합 모듈] 3종 탐지 모델 아키텍처 (손현준: Spatial/Freq + 박준서: DualStream)
├── train_all.py                   # 3종 모델 독립 학습 및 최적 가중치(.pth) 저장 파이프라인
├── ensemble_detector.py           # 3종 모델 가중치 로드 및 앙상블 추론 엔진
├── database.py                    # SQLite 탐지 결과 기록 및 조회 모듈
└── app.py                         # 실시간 탐지 웹 대시보드 및 API 서버
```

---

## 🛠️ 설치 및 환경 설정 (Installation)

Python **3.10 ~ 3.12** 환경을 권장합니다.

```bash
# 1. 저장소 클론
git clone [https://github.com/AloneFihser/Deep-catch.git](https://github.com/AloneFihser/Deep-catch.git)
cd Deep-catch

# 2. 필수 패키지 자동 점검 및 설치
python setup_env.py
```

**주요 의존성 패키지:** `torch`, `torchvision`, `opencv-python`, `numpy`, `scipy`, `Pillow`, `streamlit`, `pandas`

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
통합 모듈(`composite_detector_models.py`)의 3가지 탐지 모델을 순차적으로 학습하고 최고 성능의 가중치 파일(`best_Spatial.pth`, `best_Frequency.pth`, `best_DualStream.pth`)을 저장합니다.

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
웹 인터페이스를 실행하여 브라우저에서 직접 이미지를 업로드하고 탐지 이력을 조회합니다.

```bash
streamlit run app.py
```

---

## 📊 학습 데이터셋 (Datasets)

### ✅ 실제 적용 데이터셋 (Applied Dataset)
- **[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)**
  - **구성**: CIFAR-10 원본 이미지(`REAL`) 60,000장 + Stable Diffusion v1.4 생성 이미지(`FAKE`) 60,000장 (총 120,000장)
  - **활용**: 본 프로젝트의 3종 통합 탐지 모델(`Spatial`, `Frequency`, `DualStream`) 학습 및 앙상블 파이프라인 검증에 실제 적용됨

### 🔍 확장 학습용 추천 데이터셋 (Recommended Datasets)
- **얼굴 딥페이크 탐지**: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF (v2)](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- **최신 생성 AI 탐지**: [GenImage](https://github.com/GenImage-Dataset/GenImage), [Artifacts Dataset](https://www.kaggle.com/datasets/kazanova/artifacts-dataset)
