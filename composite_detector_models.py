import torch
import torch.nn as nn
from torchvision import models

# ========================================================
# [모델 1] 공간 영역 CNN 모델 (RGB 전용)
# 역할: 원본 RGB 컬러 이미지의 픽셀 배치, 질감 어색함, 얼굴 경계 왜곡 분석
# ========================================================
class SpatialCNNDetector(nn.Module):
    def __init__(self, num_classes=2):
        """
        :param num_classes: 최종 분류할 클래스 개수 (진짜: 0, 가짜: 1 -> 총 2개)
        """
        super(SpatialCNNDetector, self).__init__()
        
        # 1. ImageNet 데이터셋으로 사전 학습된 EfficientNet-B0 백본(Backbone) 불러오기
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 2. EfficientNet-B0의 마지막 분류 직전 추출되는 특징(Feature) 벡터 차원 수 파악 (1280차원)
        self.out_features = self.backbone.classifier[1].in_features
        
        # 3. 기존 1000개 클래스용 분류기(Classifier)를 항등 연산(Identity)으로 교체하여 특징만 반환하도록 개조
        self.backbone.classifier = nn.Identity()

        # 4. 추출된 1280개의 공간 단서를 기반으로 2개 클래스(Real/Fake)로 판정할 전차층(Fully Connected Layer) 구성
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),                       # 과적합(Overfitting) 방지를 위해 30% 노드를 임의 끄기
            nn.Linear(self.out_features, num_classes) # 1280차원 -> 2차원(진짜점수, 가짜점수) 변환
        )

    def forward(self, rgb_image):
        """
        :param rgb_image: [Batch_Size, 3, 256, 256] 크기의 3채널 RGB 텐서
        :return: [Batch_Size, 2] 크기의 최종 로짓(Logits) 결과
        """
        # 1. 백본 망을 통과시켜 1280차원 공간 특징 벡터 추출 [Batch, 1280]
        features = self.backbone(rgb_image)
        
        # 2. 최종 분류 층을 통과시켜 로짓 출력 [Batch, 2]
        output = self.classifier(features)
        return output


# ========================================================
# [모델 2] 주파수 영역 CNN 모델 (DCT 1채널 전용)
# 역할: 이미지의 2D DCT 주파수 변환 맵에서 미세 격자 노이즈 및 고주파 압축 흔적 분석
# ========================================================
class FrequencyNoiseDetector(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        """
        :param in_channels: 입력 주파수 맵의 채널 수 (흑백 DCT 맵이므로 1채널)
        :param num_classes: 최종 분류 클래스 개수 (2개)
        """
        super(FrequencyNoiseDetector, self).__init__()
        
        # 주파수 맵 패턴 및 미세 노이즈를 계층적으로 압축 추출하는 4단계 가벼운 CNN 구조 정의
        self.lightweight_cnn = nn.Sequential(
            # [1단 블록] 입력 채널 1 -> 출력 채널 64 (해상도: 256x256 -> 128x128)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), # 학습 안정을 위한 배치 정규화
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 2x2 맥스 풀링으로 공간 해상도 절반으로 축소

            # [2단 블록] 입력 채널 64 -> 출력 채널 128 (해상도: 128x128 -> 64x64)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # [3단 블록] 입력 채널 128 -> 출력 채널 256 (해상도: 64x64 -> 32x32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # [4단 블록] 입력 채널 256 -> 출력 채널 512 (특징 요약 및 1D 직렬화)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # 커널 크기에 상관없이 최종 특징 맵을 [512, 1, 1] 크기로 평균 요약
            nn.Flatten()                   # 2D 평면을 1D 일렬 벡터 [Batch, 512] 형태로 펼침
        )
        
        # 최종 분류기 층
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes) # 512차원 주파수 특징 -> 2차원 점수로 분류
        )

    def forward(self, freq_image):
        """
        :param freq_image: [Batch_Size, 1, 256, 256] 크기의 1채널 DCT 주파수 맵 텐서
        :return: [Batch_Size, 2] 크기의 로짓 결과
        """
        features = self.lightweight_cnn(freq_image) # 주파수 단서 512개 추출 [Batch, 512]
        output = self.classifier(features)           # 분류 판정 [Batch, 2]
        return output


# ========================================================
# [모델 3] 공간 + 주파수 듀얼 스트림 결합 모델
# 역할: RGB 공간 분석 특징(1280) + DCT 주파수 특징(512) = 1792차원으로 직렬 결합하여 판정
# ========================================================
class DualStreamDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DualStreamDeepfakeDetector, self).__init__()
        
        # 1. 공간 영역 스트림 (EfficientNet-B0)
        self.spatial_stream = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features_eff = self.spatial_stream.classifier[1].in_features # 1280차원
        self.spatial_stream.classifier = nn.Identity()

        # 2. 주파수 영역 스트림 (ResNet-18)
        self.freq_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # ResNet18의 기본 입구(Conv1)인 3채널을 1채널(DCT 입력)을 받도록 수정
        self.freq_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features_res = self.freq_stream.fc.in_features # 512차원
        self.freq_stream.fc = nn.Identity()

        # 3. 듀얼 스트림 특징 결합 차원 계산 (1280 + 512 = 1792차원)
        fusion_dim = in_features_eff + in_features_res

        # 4. 결합된 1792개 단서를 통합 분석하는 심층 분류기
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), # 1792차원 -> 512차원으로 은닉 병목 압축
            nn.BatchNorm1d(512),        # 은닉층 배치 정규화
            nn.ReLU(),                  # 비선형 활성화 함수
            nn.Dropout(0.4),            # 과적합 방지 드롭아웃 (40%)
            nn.Linear(512, num_classes) # 최종 출력: 진짜(0) or 가짜(1)
        )

    def forward(self, rgb_image, dct_image):
        """
        :param rgb_image: [Batch, 3, 256, 256] 공간 입력
        :param dct_image: [Batch, 1, 256, 256] 주파수 입력
        """
        # 두 스트림 각각에서 특징 벡터 추출
        spatial_features = self.spatial_stream(rgb_image) # [Batch, 1280]
        freq_features = self.freq_stream(dct_image)       # [Batch, 512]
        
        # 두 특징 텐서를 가로 방향(dim=1)으로 연결 (Concatenation)
        fused_features = torch.cat((spatial_features, freq_features), dim=1) # [Batch, 1792]
        
        # 통합 분류기를 통과시켜 결과 출력
        output = self.classifier(fused_features) # [Batch, 2]
        return output