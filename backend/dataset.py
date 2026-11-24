import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ========================================================
# 1. 주파수 영역 변환 함수 (DCT: Discrete Cosine Transform)
# ========================================================
def generate_dct_frequency_map(img_np, target_size=(256, 256)):
    """
    RGB 이미지 배열을 받아 2D DCT(이산 코사인 변환) 주파수 맵(1채널 텐서)으로 변환하는 함수
    
    :param img_np: (H, W, 3) 크기의 RGB NumPy 배열
    :param target_size: 주파수 맵 출력 크기 (가로, 세로)
    :return: (1, H, W) 크기의 32bit Float Tensor (0~1 정규화)
    """
    # [1단계] 이미지 크기를 고정 규격(256x256)으로 조절
    img_resized = cv2.resize(img_np, target_size)
    
    # [2단계] RGB 3채널 이미지를 흑백(Grayscale) 1채널 이미지로 변환
    # (주파수 분석은 단일 채널 노이즈/격자 패턴 분석에 주로 사용됨)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # [3단계] 2D DCT(이산 코사인 변환) 적용을 위해 float32 자료형으로 변환
    gray_float = np.float32(gray)
    
    # OpenCv의 cv2.dct 함수 적용: 공간 도메인 -> 주파수 도메인 변환
    dct_coefficients = cv2.dct(gray_float)
    
    # [4단계] 주파수 계수의 시각화 및 스케일 안정을 위해 로그 변환(Log Scale)
    # log1p(x) = log(1 + |x|) 함수로 고주파/저주파 값 간의 편차 줄임
    dct_log = np.log1p(np.abs(dct_coefficients))
    
    # [5단계] Min-Max Normalization (0 ~ 1 범위 정규화)
    min_val, max_val = np.min(dct_log), np.max(dct_log)
    if max_val - min_val > 1e-6: # 0으로 나누기 에러 방지
        dct_norm = (dct_log - min_val) / (max_val - min_val)
    else:
        dct_norm = np.zeros_like(dct_log)
        
    # [6단계] PyTorch Conv2d(1, C, kernel) 입력 형태에 맞춰 차원 변경
    # NumPy [256, 256] -> Tensor [1, 256, 256] (1채널 명시)
    dct_tensor = torch.tensor(dct_norm, dtype=torch.float32).unsqueeze(0)
    
    return dct_tensor


# ========================================================
# 2. PyTorch Custom Dataset 클래스
# ========================================================
class DeepfakePreprocessingDataset(Dataset):
    """
    압축 해제된 이미지 데이터셋 경로에서 RGB 이미지와 DCT 주파수 맵을 동시에 추출해주는 커스텀 데이터셋
    
    권장 디렉토리 구조:
    root_dir/
      ├── real/   (진짜 사진 폴더 -> 라벨 0)
      └── fake/   (가짜/AI 생성 사진 폴더 -> 라벨 1)
    """
    def __init__(self, root_dir, transform=None):
        super(DeepfakePreprocessingDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # [데이터 수집 1] real 폴더 내 모든 이미지 파일 수집 (Label: 0)
        real_files = glob.glob(os.path.join(root_dir, "real", "*.*"))
        for filepath in real_files:
            self.image_paths.append(filepath)
            self.labels.append(0)
            
        # [데이터 수집 2] fake 폴더 내 모든 이미지 파일 수집 (Label: 1)
        fake_files = glob.glob(os.path.join(root_dir, "fake", "*.*"))
        for filepath in fake_files:
            self.image_paths.append(filepath)
            self.labels.append(1)

    def __len__(self):
        # 전체 수집된 이미지 개수 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 대상 인덱스의 파일 경로 및 정답 라벨 가져오기
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 2. PIL 라이브러리로 RGB 이미지 읽기
        pil_image = Image.open(path).convert("RGB")
        
        # 3. 공간 영역(Spatial) RGB 이미지 전처리 적용
        if self.transform:
            rgb_tensor = self.transform(pil_image)
        else:
            # 기본 Transform 미지정 시 텐서 변환만 수행
            rgb_tensor = transforms.ToTensor()(pil_image)
            
        # 4. 주파수 영역(Frequency) DCT 맵 계산
        # PIL 이미지를 NumPy 배열로 변환 후 DCT 변환 함수 전달
        img_np = np.array(pil_image)
        dct_tensor = generate_dct_frequency_map(img_np, target_size=(256, 256))
        
        # 5. [RGB 텐서, DCT 텐서, 라벨] 최종 반환
        return rgb_tensor, dct_tensor, torch.tensor(label, dtype=torch.long)


# ========================================================
# 3. DataLoader 구축 및 모델 전달 데이터 호환 확인 예시
# ========================================================
if __name__ == "__main__":
    # 1. 공간(RGB) 이미지용 Transform 설정 (EfficientNet 규격 맞춤)
    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),               # 이미지 크기 256x256 통일
        transforms.ToTensor(),                        # 0~255 픽셀값을 0.0~1.0 텐서로 전환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 사전학습 가중치 표준화
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Dataset 및 DataLoader 생성
    # (압축해제된 데이터 경로 지정)
    dataset_path = "./extracted_data"
    
    if os.path.exists(dataset_path):
        dataset = DeepfakePreprocessingDataset(root_dir=dataset_path, transform=spatial_transform)
        
        # DataLoader: 배치 단위(배치크기 4)로 데이터를 묶어 모델에 공급
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"총 불러온 이미지 개수: {len(dataset)}장")
        
        # 3. 모델에 전달될 배치 데이터 형태 확인
        for batch_idx, (rgb_batch, dct_batch, labels_batch) in enumerate(dataloader):
            print(f"\n[Batch {batch_idx + 1}] 추출 데이터 텐서 모양:")
            print(f" - 공간(RGB) 텐서 크기     : {rgb_batch.shape}   -> [Batch, 3, 256, 256] (EfficientNet 입력)")
            print(f" - 주파수(DCT) 텐서 크기    : {dct_batch.shape}   -> [Batch, 1, 256, 256] (ResNet/CNN 입력)")
            print(f" - 정답 라벨 (0:Real, 1:Fake): {labels_batch}")
            
            # 첫 번째 배치 확인 후 루프 종료
            break