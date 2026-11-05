import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.fftpack

# ========================================================
# [1] 모델 정의부 (사용자 제공 코드)
# ========================================================
class SpatialCNNDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SpatialCNNDetector, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.out_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.out_features, num_classes)
        )

    def forward(self, rgb_image):
        features = self.backbone(rgb_image)
        output = self.classifier(features)
        return output


class FrequencyNoiseDetector(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(FrequencyNoiseDetector, self).__init__()
        self.lightweight_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, freq_image):
        features = self.lightweight_cnn(freq_image)
        output = self.classifier(features)
        return output


class DualStreamDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DualStreamDeepfakeDetector, self).__init__()
        self.spatial_stream = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features_eff = self.spatial_stream.classifier[1].in_features
        self.spatial_stream.classifier = nn.Identity()

        self.freq_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.freq_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features_res = self.freq_stream.fc.in_features
        self.freq_stream.fc = nn.Identity()

        fusion_dim = in_features_eff + in_features_res

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_image, dct_image):
        spatial_features = self.spatial_stream(rgb_image)
        freq_features = self.freq_stream(dct_image)
        fused_features = torch.cat((spatial_features, freq_features), dim=1)
        output = self.classifier(fused_features)
        return output


# ========================================================
# [2] 호환용 데이터셋 클래스 (RGB + DCT 자동 변환 지원)
# ========================================================
class DualStreamDataset(Dataset):
    """
    압축 해제된 이미지 폴더 구조(예: ./extracted_data/real, ./extracted_data/fake)를 
    읽어서 RGB 텐서와 DCT 주파수 맵 텐서를 동시 반환하는 데이터셋
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 폴더 구조 탐색 (real: 0, fake: 1 가정)
        class_mapping = {'real': 0, 'fake': 1}
        for label_name, label_idx in class_mapping.items():
            class_folder = os.path.join(data_dir, label_name)
            if os.path.exists(class_folder):
                for img_name in os.listdir(class_folder):
                    if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'webp')):
                        self.image_paths.append(os.path.join(class_folder, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def _get_dct_map(self, img_pil):
        """이미지를 그레이스케일로 변환 후 2D DCT 주파수 맵 추출"""
        img_gray = img_pil.convert('L').resize((256, 256))
        img_np = np.float32(img_gray)
        # 2D DCT 수행
        dct_np = scipy.fftpack.dct(scipy.fftpack.dct(img_np, axis=0, norm='ortho'), axis=1, norm='ortho')
        # 로그 스케일링으로 값의 범위 안정화
        dct_log = np.log(np.abs(dct_np) + 1e-5)
        dct_tensor = torch.tensor(dct_log, dtype=torch.float32).unsqueeze(0) # [1, 256, 256]
        return dct_tensor

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패 ({img_path}): {e}")
            # 예외 시 더미 데이터 반환
            img_pil = Image.new('RGB', (256, 256))

        # 1. 공간 영역 RGB 처리
        if self.transform:
            rgb_tensor = self.transform(img_pil)
        else:
            # 기본 정규화 및 텐서 변환
            from torchvision import transforms
            basic_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            rgb_tensor = basic_transform(img_pil)

        # 2. 주파수 영역 DCT 맵 처리
        dct_tensor = self._get_dct_map(img_pil)

        return rgb_tensor, dct_tensor, torch.tensor(label, dtype=torch.long)


# ========================================================
# [3] 앙상블 및 테스트 실행 예시 함수
# ========================================================
if __name__ == "__main__":
    print("=== 듀얼 스트림 모델 및 데이터셋 로드 테스트 ===")
    
    # 모델 인스턴스화 테스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamDeepfakeDetector(num_classes=2).to(device)
    print(f"모델이 성공적으로 {device}에 로드되었습니다.")

    # 더미 텐서 테스트 (Batch=2, RGB=[2,3,256,256], DCT=[2,1,256,256])
    dummy_rgb = torch.randn(2, 3, 256, 256).to(device)
    dummy_dct = torch.randn(2, 1, 256, 256).to(device)
    
    output = model(dummy_rgb, dummy_dct)
    print(f"출력 로짓(Logits) Shape: {output.shape}")  # 기대값: torch.Size([2, 2])
