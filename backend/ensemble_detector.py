import torch
import torch.nn.functional as F

# 데이터베이스 조작 함수 및 composite_detector_models 클래스 불러오기
from database import init_db, save_result_to_db
from composite_detector_models import (
    SpatialCNNDetector,
    FrequencyNoiseDetector,
    DualStreamDeepfakeDetector
)

class EnsembleDeepfakeDetector:
    """
    3개 모델(공간, 주파수, 듀얼스트림)의 판정 확률을 로드하여 
    가중 평균(Soft Voting Ensemble)으로 종합 진단하는 추론 시스템
    """
    def __init__(self, spatial_path, freq_path, dual_path, device):
        """
        :param spatial_path: SpatialCNNDetector 가중치 파일 경로 (.pth)
        :param freq_path: FrequencyNoiseDetector 가중치 파일 경로 (.pth)
        :param dual_path: DualStreamDeepfakeDetector 가중치 파일 경로 (.pth)
        :param device: 연산 장치 (CPU or GPU)
        """
        self.device = device

        # [모델 1 생성 및 학습된 가중치 복원]
        self.spatial_model = SpatialCNNDetector(num_classes=2).to(device)
        self.spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
        self.spatial_model.eval() # 추론 모드로 설정

        # [모델 2 생성 및 학습된 가중치 복원]
        self.freq_model = FrequencyNoiseDetector(in_channels=1, num_classes=2).to(device)
        self.freq_model.load_state_dict(torch.load(freq_path, map_location=device))
        self.freq_model.eval()

        # [모델 3 생성 및 학습된 가중치 복원]
        self.dual_model = DualStreamDeepfakeDetector(num_classes=2).to(device)
        self.dual_model.load_state_dict(torch.load(dual_path, map_location=device))
        self.dual_model.eval()

    def predict_and_ensemble(self, filename, rgb_tensor, dct_tensor, weights=(0.3, 0.3, 0.4)):
        """
        3개의 탐지 모델 연산을 독립 수행한 뒤 소프트맥스 확률 종합
        
        :param filename: DB 보관용 파일명
        :param rgb_tensor: [1, 3, 256, 256] 공간 컬러 입력 텐서
        :param dct_tensor: [1, 1, 256, 256] 주파수 흑백 입력 텐서
        :param weights: (공간 비중, 주파수 비중, 듀얼스트림 비중) 가중치 튜플 (합계: 1.0)
        :return: final_pred ("Real" 또는 "Fake"), ensemble_fake_prob (최종 가짜 확률 %)
        """
        # 텐서 데이터를 연산 장치로 이동
        rgb_tensor = rgb_tensor.to(self.device)
        dct_tensor = dct_tensor.to(self.device)

        # 기울기 계산 안함
        with torch.no_grad():
            # 1. Spatial 모델 연산 -> Softmax 함수로 가짜 확률 계산
            spatial_logits = self.spatial_model(rgb_tensor)
            spatial_probs = F.softmax(spatial_logits, dim=1)
            spatial_fake_prob = spatial_probs[0][1].item() # index 1: 가짜(Fake) 확률

            # 2. Frequency 모델 연산 -> Softmax 가짜 확률 계산
            freq_logits = self.freq_model(dct_tensor)
            freq_probs = F.softmax(freq_logits, dim=1)
            freq_fake_prob = freq_probs[0][1].item()

            # 3. DualStream 모델 연산 -> Softmax 가짜 확률 계산
            dual_logits = self.dual_model(rgb_tensor, dct_tensor)
            dual_probs = F.softmax(dual_logits, dim=1)
            dual_fake_prob = dual_probs[0][1].item()

        # 4. 소프트 보팅(Soft Voting) 가중 평균 공식을 적용해 종합 가짜 확률 계산
        w_spatial, w_freq, w_dual = weights
        ensemble_fake_prob = (
            (spatial_fake_prob * w_spatial) +
            (freq_fake_prob * w_freq) +
            (dual_fake_prob * w_dual)
        ) * 100.0 # 퍼센트 단위 변환 (0~100%)

        # 5. 최종 기준값 50% 이상 시 Fake 확정
        final_pred = "Fake" if ensemble_fake_prob >= 50.0 else "Real"

        # 콘솔 모니터링 출력
        print(f"\n[단일 이미지 탐지 분석 결과 - 파일명: {filename}]")
        print(f" ├─ 공간 모델(Spatial) 가짜 확률     : {spatial_fake_prob * 100:.2f}%")
        print(f" ├─ 주파수 모델(Frequency) 가짜 확률 : {freq_fake_prob * 100:.2f}%")
        print(f" ├─ 듀얼 스트림(DualStream) 가짜 확률 : {dual_fake_prob * 100:.2f}%")
        print(f" └=> [종합 앙상블 판정] 최종 결과: {final_pred} (통합 가짜 확률: {ensemble_fake_prob:.2f}%)")

        # 6. 최종 판정 결과를 SQLite DB (deepfake_detector.db) 데이터베이스에 저장
        save_result_to_db(
            filename=filename,
            prediction=final_pred,
            fake_probability=ensemble_fake_prob,
            db_path="deepfake_detector.db"
        )

        return final_pred, ensemble_fake_prob


# ========================================================
# 독립 실행 및 테스트부
# ========================================================
if __name__ == "__main__":
    # 데이터베이스 준비
    init_db("deepfake_detector.db")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 앙상블 탐지 엔진 객체 초기화 (.pth 가중치 파일 로드)
    detector = EnsembleDeepfakeDetector(
        spatial_path="best_Spatial.pth",
        freq_path="best_Frequency.pth",
        dual_path="best_DualStream.pth",
        device=device
    )

    # 2. 테스트용 가상 임의 데이터 생성 (RGB 3채널, DCT 1채널)
    sample_rgb = torch.randn(1, 3, 256, 256)
    sample_dct = torch.randn(1, 1, 256, 256)

    # 3. 앙상블 종합 진단 및 DB 기록 실행
    detector.predict_and_ensemble("test_sample_01.jpg", sample_rgb, sample_dct)