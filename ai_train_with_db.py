import os
import sqlite3
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# ========================================================
# 1. 데이터베이스 초기화 및 저장 함수 (SQLite + Python)
# ========================================================
def init_db(db_path="deepfake_detector.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 분석 이력 및 학습 로그를 저장할 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[DB] SQLite 데이터베이스 및 테이블 준비 완료!")

def save_result_to_db(filename, prediction, fake_probability, db_path="deepfake_detector.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO detection_logs (filename, prediction, fake_probability, created_at)
        VALUES (?, ?, ?, ?)
    ''', (filename, prediction, fake_probability, current_time))
    
    conn.commit()
    conn.close()
    print(f"[DB 저장 완료] 파일: {filename} | 판정: {prediction} | 가짜 확률: {fake_probability:.2f}%")


# ========================================================
# 2. 듀얼 스트림 딥페이크 탐지 모델 (공간 + 주파수 결합)
# ========================================================
class DualStreamDeepfakeDetector(nn.Module):
    """
    공간 영역(EfficientNet)과 주파수 영역(ResNet)을 결합한 듀얼 스트림 모델
    """
    def __init__(self, num_classes=2):
        super(DualStreamDeepfakeDetector, self).__init__()
        
        # 1. 공간 영역 (Spatial Stream) : EfficientNet-B0
        self.spatial_stream = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features_eff = self.spatial_stream.classifier[1].in_features
        self.spatial_stream.classifier = nn.Identity() 

        # 2. 주파수 영역 (Frequency Stream) : ResNet-18
        self.freq_stream = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.freq_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features_res = self.freq_stream.fc.in_features
        self.freq_stream.fc = nn.Identity()

        # 3. 최종 결합 및 판별기 (Fusion & Classifier)
        fusion_dim = in_features_eff + in_features_res  # 총 1792차원
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_image, dct_image):
        spatial_features = self.spatial_stream(rgb_image)   # [Batch, 1280]
        freq_features = self.freq_stream(dct_image)         # [Batch, 512]
        
        # 두 특징 가로로 이어 붙이기 (Concatenation)
        fused_features = torch.cat((spatial_features, freq_features), dim=1) # [Batch, 1792]
        
        output = self.classifier(fused_features)
        return output


# ========================================================
# 3. 훈련용 커스텀 데이터셋 클래스
# ========================================================
class DualStreamDataset(Dataset):
    def __init__(self, rgb_data, dct_data, labels):
        self.rgb_data = rgb_data
        self.dct_data = dct_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.rgb_data[idx], self.dct_data[idx], self.labels[idx]


# ========================================================
# 4. 전체 실행 및 테스트 파이프라인 (훈련 + DB 연동)
# ========================================================
if __name__ == "__main__":
    # 1. 데이터베이스 구축(초기화)
    init_db()

    # 2. 디바이스 설정 (GPU 사용 가능 시 GPU 활용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 모델 초기화 및 이동
    model = DualStreamDeepfakeDetector(num_classes=2).to(device)

    # 4. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 가상의 훈련 데이터 생성 (샘플 16개, 이미지 크기 256x256)
    dummy_rgb = torch.randn(16, 3, 256, 256)
    dummy_dct = torch.randn(16, 1, 256, 256)
    dummy_labels = torch.randint(0, 2, (16,))

    dataset = DualStreamDataset(dummy_rgb, dummy_dct, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 6. 간단한 훈련(Training) 진행
    print("\n--- AI 모델 훈련 시작 ---")
    model.train()
    for epoch in range(2):  # 테스트용으로 2 에폭만 진행
        running_loss = 0.0
        for rgb_imgs, dct_imgs, labels in dataloader:
            rgb_imgs, dct_imgs, labels = rgb_imgs.to(device), dct_imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_imgs, dct_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/2], Loss: {running_loss/len(dataloader):.4f}")

    # 7. 훈련된 모델 저장
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/dual_stream_model.pth")
    print("모델 훈련 완료 및 저장 성공!")

    # 8. 추론(Inference) 테스트 및 데이터베이스 자동 저장 연동
    print("\n--- 모델 추론 및 DB 저장 테스트 ---")
    model.eval()
    
    # 테스트용 단일 이미지 시뮬레이션
    test_rgb = torch.randn(1, 3, 256, 256).to(device)
    test_dct = torch.randn(1, 1, 256, 256).to(device)
    test_filename = "sample_target_photo.jpg"

    with torch.no_grad():
        output = model(test_rgb, test_dct)
        probabilities = torch.softmax(output, dim=1)
        fake_prob = probabilities[0][1].item() * 100
        prediction = "가짜 (Fake)" if fake_prob > 50 else "진짜 (Real)"

    # 분석 결과를 SQLite 데이터베이스에 자동 저장
    save_result_to_db(filename=test_filename, prediction=prediction, fake_probability=fake_prob)

    # 9. DB에 저장된 내용 최종 확인
    conn = sqlite3.connect("deepfake_detector.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detection_logs")
    rows = cursor.fetchall()
    print("\n[현재 데이터베이스에 기록된 탐지 이력]")
    for row in rows:
        print(f"ID: {row[0]} | 파일명: {row[1]} | 판정: {row[2]} | 가짜 확률: {row[3]:.2f}% | 시간: {row[4]}")
    conn.close()
