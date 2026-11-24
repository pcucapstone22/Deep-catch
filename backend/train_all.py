import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 1. Dataset 전처리 모듈 및 수정된 composite_detector_models 파일 import
from dataset import DeepfakePreprocessingDataset
from composite_detector_models import (
    SpatialCNNDetector,
    FrequencyNoiseDetector,
    DualStreamDeepfakeDetector
)

def train_single_model(model_name, model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    """
    모든 모델 구조에 구애받지 않고 학습 및 검증 루프를 공통으로 제어하는 처리 함수
    
    :param model_name: 모델 식별 이름 (가중치 파일명에 사용)
    :param model: 학습 대상 PyTorch 모델 인스턴스
    :param train_loader: 학습용 DataLoader
    :param val_loader: 검증용 DataLoader
    :param device: 연산 장치 (CPU 또는 GPU)
    :param epochs: 학습 총 횟수
    :param lr: 학습률 (Learning Rate)
    :return: 최적 가중치 파일 저장 경로
    """
    print(f"\n==========================================")
    print(f" ▶ [{model_name}] 모델 학습 프로세스 개시")
    print(f"==========================================")
    
    # 이진/다중 분류용 CrossEntropy 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # AdamW 최적화 알고리즘 (weight_decay로 과적합 제어)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_acc = 0.0
    save_path = f"best_{model_name}.pth"

    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------
        # (A) 학습(Training) 단계를 위한 루프
        # ----------------------------------------------------
        model.train() # 학습 모드 전환 (Dropout, BatchNorm 활성화)
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for rgb_batch, dct_batch, labels in train_loader:
            # 배치 데이터를 지정된 연산 장치(GPU/CPU)로 전송
            rgb_batch = rgb_batch.to(device)
            dct_batch = dct_batch.to(device)
            labels = labels.to(device)

            # 이전 루프의 역전파 기울기(Gradient) 초기화
            optimizer.zero_grad()

            # [모델 분기 처리] 모델 종류에 따라 요구되는 입력 형태에 맞게 통과
            if model_name == "Spatial":
                outputs = model(rgb_batch)           # RGB 텐서만 입력받음
            elif model_name == "Frequency":
                outputs = model(dct_batch)           # DCT 텐서만 입력받음
            elif model_name == "DualStream":
                outputs = model(rgb_batch, dct_batch) # 두 텐서 모두 입력받음

            # 손실값(Loss) 계산
            loss = criterion(outputs, labels)
            
            # 역전파 계산 및 가중치 업데이트
            loss.backward()
            optimizer.step()

            # 오차 및 정답 수집
            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1) # 가장 확률이 높은 클래스 결정
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        # ----------------------------------------------------
        # (B) 검증(Validation) 단계를 위한 루프
        # ----------------------------------------------------
        model.eval() # 평가 모드 전환 (Dropout 비활성화)
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        # 검증 연산 시에는 메모리 절약을 위해 기울기 계산을 끔
        with torch.no_grad():
            for rgb_batch, dct_batch, labels in val_loader:
                rgb_batch = rgb_batch.to(device)
                dct_batch = dct_batch.to(device)
                labels = labels.to(device)

                if model_name == "Spatial":
                    outputs = model(rgb_batch)
                elif model_name == "Frequency":
                    outputs = model(dct_batch)
                elif model_name == "DualStream":
                    outputs = model(rgb_batch, dct_batch)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")

        # 최고 검증 정확도를 경신하면 모델 가중치 파일 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  └─ 최적 가중치 갱신 완료 -> {save_path}")

    return save_path


if __name__ == "__main__":
    # GPU 장치 탐색
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습에 연산할 하드웨어 장치: {device}")
    
    DATA_DIR = "./extracted_data"

    # 이미지 전처리 통일 기준 설정
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset 생성 및 8:2 비율로 Train/Validation 분할
    dataset = DeepfakePreprocessingDataset(root_dir=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # DataLoader 구성
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # 1. SpatialCNNDetector 독립 학습 (best_Spatial.pth 저장)
    spatial_model = SpatialCNNDetector(num_classes=2).to(device)
    train_single_model("Spatial", spatial_model, train_loader, val_loader, device, epochs=5)

    # 2. FrequencyNoiseDetector 독립 학습 (best_Frequency.pth 저장)
    freq_model = FrequencyNoiseDetector(in_channels=1, num_classes=2).to(device)
    train_single_model("Frequency", freq_model, train_loader, val_loader, device, epochs=5)

    # 3. DualStreamDeepfakeDetector 독립 학습 (best_DualStream.pth 저장)
    dual_model = DualStreamDeepfakeDetector(num_classes=2).to(device)
    train_single_model("DualStream", dual_model, train_loader, val_loader, device, epochs=5)