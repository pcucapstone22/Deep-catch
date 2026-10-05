import torch
import torch.nn as nn
from torchvision import models

class DualStreamDeepfakeDetector(nn.Module):
    """
    공간 영역(EfficientNet)과 주파수 영역(ResNet)을 결합한 듀얼 스트림 모델
    """
    def __init__(self, num_classes=2):
        super(DualStreamDeepfakeDetector, self).__init__()
        
        # ========================================================
        # 1. 공간 영역 (Spatial Stream) : EfficientNet-B0
        # 역할: 컬러(RGB) 원본 이미지를 받아 픽셀 형태와 시각적 오류 분석
        # ========================================================
        self.spatial_stream = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # EfficientNet이 뱉어내는 특징 데이터의 크기 (1280차원)
        in_features_eff = self.spatial_stream.classifier[1].in_features
        # 기존 1000개짜리 분류기를 없애고, 순수 '특징'만 뽑아내도록 개조
        self.spatial_stream.classifier = nn.Identity() 

        # ========================================================
        # 2. 주파수 영역 (Frequency Stream) : ResNet-18
        # 역할: 흑백(1채널) DCT 주파수 맵을 받아 미세 노이즈 격자 분석
        # ========================================================
        self.freq_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 흑백 주파수 맵(1채널)을 입력받을 수 있도록 첫 번째 레이어 입구 개조
        self.freq_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # ResNet-18이 뱉어내는 특징 데이터의 크기 (512차원)
        in_features_res = self.freq_stream.fc.in_features
        # 기존 분류기를 없애고 순수 '특징'만 뽑아내도록 개조
        self.freq_stream.fc = nn.Identity()

        # ========================================================
        # 3. 최종 결합 및 판별기 (Fusion & Classifier)
        # 역할: EfficientNet(1280) + ResNet18(512) 결과를 하나로 합쳐서 최종 판정
        # ========================================================
        fusion_dim = in_features_eff + in_features_res  # 총 1792차원
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4), # 과적합 방지용 드롭아웃
            nn.Linear(512, num_classes) # 최종 출력: 진짜(0) or 가짜(1)
        )

    def forward(self, rgb_image, dct_image):
        # 1. 각 스트림에 이미지를 통과시켜 특징(Feature) 추출
        spatial_features = self.spatial_stream(rgb_image)   # [Batch, 1280]
        freq_features = self.freq_stream(dct_image)         # [Batch, 512]
        
        # 2. 추출된 두 특징을 가로로 이어 붙이기 (Concatenation)
        fused_features = torch.cat((spatial_features, freq_features), dim=1) # [Batch, 1792]
        
        # 3. 결합된 데이터를 최종 판별기에 넣어 가짜 확률 계산
        output = self.classifier(fused_features)
        
        return output
