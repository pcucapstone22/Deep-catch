import torch
import torch.nn as nn
from torchvision import models

# ========================================================
# [모델 1] 공간 영역 CNN 기반 모델 (Spatial Domain CNN)
# - 역할: 컬러(RGB) 원본 이미지에서 시각적 어색함 분석 (1280개 단서 추출)
# ========================================================
class SpatialCNNDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SpatialCNNDetector, self).__init__()
        
        # 1. 특징 추출기(Feature Extractor): 똑똑한 EfficientNet-B0 불러오기
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.out_features = self.backbone.classifier[1].in_features  # 1280차원
        
        # 기존 1000개 분류기를 지우고 순수 단서(1280개)만 나오도록 개조
        self.backbone.classifier = nn.Identity()

        # 2. 판별기(Classifier): 1280개 단서를 보고 진짜(0) / 가짜(1) 판정
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.out_features, num_classes)
        )

    def forward(self, rgb_image):
        features = self.backbone(rgb_image)      # 1) 1280개 단서 뽑기
        output = self.classifier(features)       # 2) 진짜/가짜 최종 판정
        return output


# ========================================================
# [모델 2] 주파수 도메인 & 노이즈 기반 모델 (Frequency & Noise Model)
# - 역할: 주파수(DCT/FFT)나 노이즈(SRM) 맵에서 미세 격자 무늬 분석 (512개 단서 추출)
# ========================================================
class FrequencyNoiseDetector(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(FrequencyNoiseDetector, self).__init__()
        
        # 1. 특징 추출기(Feature Extractor): 4단 가벼운 합성곱 신경망(Light-weight CNN)
        self.lightweight_cnn = nn.Sequential(
            # 1단 거름망 (단서 64개 추출)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 2단 거름망 (단서 128개 추출)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 3단 거름망 (단서 256개 추출)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 4단 거름망 (최종 단서 512개로 압축)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # [Batch, 512] 크기로 일렬 정돈
        )
        
        self.out_features = 512

        # 2. 판별기(Classifier): 512개 단서를 보고 진짜(0) / 가짜(1) 판정
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.out_features, num_classes)
        )

    def forward(self, freq_or_noise_image):
        features = self.lightweight_cnn(freq_or_noise_image) # 1) 512개 단서 뽑기
        output = self.classifier(features)                   # 2) 진짜/가짜 최종 판정
        return output


# ========================================================
# [실행 확인 스위치] 이 파일을 실행하면 아래 코드가 작동합니다!
# ========================================================
if __name__ == "__main__":
    print("1. 가상 테스트 이미지 생성 중...")
    # 전처리 조원이 보내줄 데이터라고 가정하고 가짜 이미지 4장씩 생성 (크기 256x256)
    sample_rgb = torch.randn(4, 3, 256, 256)   # 컬러 사진 4장 (채널 3개: R, G, B)
    sample_freq = torch.randn(4, 1, 256, 256)  # 흑백 주파수 맵 4장 (채널 1개)

    print("2. 공간 영역 CNN 모델 작동 테스트...")
    spatial_model = SpatialCNNDetector(num_classes=2)
    spatial_result = spatial_model(sample_rgb)
    print(f"   -> 공간 모델 출력 크기: {spatial_result.shape} (사진 4장 x [진짜점수, 가짜점수] 2개)")

    print("3. 주파수 & 노이즈 모델 작동 테스트...")
    freq_model = FrequencyNoiseDetector(in_channels=1, num_classes=2)
    freq_result = freq_model(sample_freq)
    print(f"   -> 주파수 모델 출력 크기: {freq_result.shape} (사진 4장 x [진짜점수, 가짜점수] 2개)")
    
    print("\n[성공] 두 모델 모두 에러 없이 완벽하게 작동합니다!") 
