import os
import glob
import cv2
import numpy as np
from scipy.fftpack import dct
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# =====================================================================
# 1. SRM(Steganography Rich Model) 필터 생성 함수
# =====================================================================
def get_srm_filters():
    """
    SRM 고주파 필터 커널 생성 함수.
    이미지 내부의 미세한 노이즈 레지듀얼(Residual)을 추출하기 위해
    1차, 2차, 3차 미분 기반 고주파 필터 3종을 생성합니다.
    """
    filter1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    filter2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    filter3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)

    # PyTorch Conv2d 형식인 (Out_Channels, In_Channels, H, W) -> (3, 1, 3, 3)으로 차원 확장
    srm_kernels = np.stack([filter1, filter2, filter3], axis=0)
    srm_kernels = np.expand_dims(srm_kernels, axis=1)
    return torch.from_numpy(srm_kernels)


# =====================================================================
# 2. Multi-Stream Custom Dataset 클래스
# =====================================================================
class DeepfakeMultiStreamDataset(Dataset):
    """
    RGB, DCT(주파수), SRM(노이즈) 3개 스트림 입력을 생성하는 PyTorch Dataset
    """
    def __init__(self, image_paths, labels, img_size=(256, 256)):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        
        # SRM 필터를 PyTorch 고정 텐서로 초기화
        self.srm_weights = get_srm_filters()
        
        # 기본 Transform (Resize 및 Tensor 변환)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        
        # ImageNet 정규화 파라미터 (RGB 입력용)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _extract_dct(self, img_gray):
        """
        [주파수 도메인 전처리]
        2D DCT 변환 후 Log-Scale 적용 및 Min-Max Normalization 수행
        """
        # 2D DCT 변환
        dct_feat = dct(dct(img_gray.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # Log Scale 적용 (epsilon 1e-5)
        dct_log = np.log(np.abs(dct_feat) + 1e-5)
        
        # Min-Max Normalization (0 ~ 1)
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8)
        
        # (1, H, W) 텐서 변환
        return torch.from_numpy(dct_norm).unsqueeze(0).float()

    def _extract_srm(self, rgb_tensor):
        """
        [노이즈 도메인 전처리]
        RGB 입력에 SRM 고주파 필터를 씌워 노이즈 레지듀얼 추출
        """
        # RGB -> Gray Scale 변환
        gray_tensor = 0.2989 * rgb_tensor[0:1] + 0.5870 * rgb_tensor[1:2] + 0.1140 * rgb_tensor[2:3]
        gray_tensor = gray_tensor.unsqueeze(0)  # (1, 1, H, W)
        
        # Conv2d 연산으로 SRM 필터 적용 (Gradient 비활성화)
        with torch.no_grad():
            srm_res = torch.nn.functional.conv2d(gray_tensor, self.srm_weights, padding=1)
            # -2 ~ 2 범위로 Truncation하여 노이즈 패턴 강조
            srm_res = torch.clamp(srm_res, min=-2.0, max=2.0)
            
        return srm_res.squeeze(0)  # (3, H, W)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. RGB Stream
        rgb_tensor = self.transform(img_rgb)
        rgb_normalized = self.normalize(rgb_tensor)

        # 2. DCT Frequency Stream
        img_gray_resized = cv2.resize(img_gray, self.img_size)
        dct_tensor = self._extract_dct(img_gray_resized)

        # 3. SRM Noise Stream
        srm_tensor = self._extract_srm(rgb_tensor)

        return {
            'rgb': rgb_normalized,
            'dct': dct_tensor,
            'srm': srm_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }


# =====================================================================
# 3. 백본 및 트리플 스트림 Fusion 신경망 구조
# =====================================================================
class StreamBackbone(nn.Module):
    """
    개별 스트림(RGB, DCT, SRM)의 Feature Extractor (ResNet-18 기반)
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super(StreamBackbone, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 1채널 입력(DCT) 대응을 위한 첫번째 Conv 레이어 수정
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    base_model.conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                )
        else:
            self.conv1 = base_model.conv1

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FeatureAttentionFusion(nn.Module):
    """
    Cross-Attention 기반 Stream Fusion 모듈
    """
    def __init__(self, feature_dim=256, num_streams=3):
        super(FeatureAttentionFusion, self).__init__()
        self.num_streams = num_streams
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_streams, 128),
            nn.ReLU(),
            nn.Linear(128, num_streams),
            nn.Softmax(dim=-1)
        )

    def forward(self, features):
        concat_feat = torch.cat(features, dim=1)
        attn_weights = self.attention(concat_feat)  # (B, 3)
        
        weighted_features = [features[i] * attn_weights[:, i:i+1] for i in range(self.num_streams)]
        fused = torch.cat(weighted_features, dim=1)
        return fused, attn_weights


class MultiStreamDeepfakeDetectorWithAux(nn.Module):
    """
    Auxiliary Loss를 지원하는 Triple Stream Detector
    """
    def __init__(self, feature_dim=256, num_classes=2):
        super(MultiStreamDeepfakeDetectorWithAux, self).__init__()
        
        # 3개 스트림 백본
        self.rgb_stream = StreamBackbone(in_channels=3, feature_dim=feature_dim)
        self.dct_stream = StreamBackbone(in_channels=1, feature_dim=feature_dim)
        self.srm_stream = StreamBackbone(in_channels=3, feature_dim=feature_dim)
        
        # Auxiliary Classifier (스트림 편향 방지용)
        self.aux_rgb = nn.Linear(feature_dim, num_classes)
        self.aux_dct = nn.Linear(feature_dim, num_classes)
        self.aux_srm = nn.Linear(feature_dim, num_classes)
        
        # Fusion & Main Classifier
        self.fusion_module = FeatureAttentionFusion(feature_dim=feature_dim, num_streams=3)
        self.main_classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, dct, srm):
        f_rgb = self.rgb_stream(rgb)
        f_dct = self.dct_stream(dct)
        f_srm = self.srm_stream(srm)
        
        # Auxiliary Logits
        l_rgb = self.aux_rgb(f_rgb)
        l_dct = self.aux_dct(f_dct)
        l_srm = self.aux_srm(f_srm)
        
        # Fusion & Final Logit
        fused_feat, attn_weights = self.fusion_module([f_rgb, f_dct, f_srm])
        l_final = self.main_classifier(fused_feat)
        
        return {
            'final': l_final,
            'rgb': l_rgb,
            'dct': l_dct,
            'srm': l_srm,
            'weights': attn_weights
        }


# =====================================================================
# 4. AUC / EER 성능 평가 함수
# =====================================================================
def calculate_metrics(y_true, y_scores):
    """
    ROC Curve 기반 AUC 및 EER(Equal Error Rate) 산출 함수
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    auc_score = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return auc_score, eer * 100.0, eer_threshold


def evaluate(model, dataloader, device, use_amp=True):
    """
    AMP가 적용된 평가 루프 함수
    """
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            dct = batch['dct'].to(device, non_blocking=True)
            srm = batch['srm'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # 평가 시에도 AMP 캐스팅으로 VRAM 소모 절감
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(rgb, dct, srm)
                probs = torch.softmax(outputs['final'], dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    auc_val, eer_val, thresh_val = calculate_metrics(y_true, y_scores)
    return auc_val, eer_val, thresh_val


# =====================================================================
# 5. RTX 3060 Laptop 최적화 AMP Training Loop 함수
# =====================================================================
def train_one_epoch_amp(model, dataloader, criterion, optimizer, scaler, device, alpha=0.3, use_amp=True):
    """
    torch.cuda.amp.autocast 및 GradScaler가 적용된 1 Epoch 학습 루프
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # non_blocking=True로 GPU 메모리 복사 병목 최소화
        rgb = batch['rgb'].to(device, non_blocking=True)
        dct = batch['dct'].to(device, non_blocking=True)
        srm = batch['srm'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # -------------------------------------------------------------
        # [핵심] AMP Autocast Context
        # FP32 연산 중 주요 연산을 FP16으로 자동 변환해 VRAM 40% 절감 및 Tensor Core 가속
        # -------------------------------------------------------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(rgb, dct, srm)

            loss_final = criterion(outputs['final'], labels)
            loss_rgb = criterion(outputs['rgb'], labels)
            loss_dct = criterion(outputs['dct'], labels)
            loss_srm = criterion(outputs['srm'], labels)

            total_loss = loss_final + alpha * (loss_rgb + loss_dct + loss_srm)

        # -------------------------------------------------------------
        # [핵심] GradScaler를 이용한 Gradient Scaling 및 Backward
        # FP16 연산 시 발생하는 Underflow 현상을 방지하며 가중치 업데이트
        # -------------------------------------------------------------
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item() * labels.size(0)

    return running_loss / len(dataloader.dataset)


# =====================================================================
# 6. 실제 데이터 경로 탐색 함수
# =====================================================================
def load_dataset_paths(dataset_root_dir):
    """
    루트 폴더 내 real/ 및 fake/ 하위 경로의 이미지 수집
    """
    real_dir = os.path.join(dataset_root_dir, "real")
    fake_dir = os.path.join(dataset_root_dir, "fake")

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG')
    real_paths, fake_paths = [], []

    for ext in extensions:
        real_paths.extend(glob.glob(os.path.join(real_dir, "**", ext), recursive=True))
        fake_paths.extend(glob.glob(os.path.join(fake_dir, "**", ext), recursive=True))

    image_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    print(f"[+] 데이터 수집 완료 | Real: {len(real_paths)}장 | Fake: {len(fake_paths)}장 | 총: {len(image_paths)}장")
    return image_paths, labels


# =====================================================================
# 7. Main Execution Flow (RTX 3060 Laptop Optimized)
# =====================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # [RTX 3060 Laptop VRAM 6GB 전용 하이퍼파라미터]
    # -----------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = "./data/FaceForensics"  # 본인의 데이터셋 경로로 설정
    
    IMAGE_SIZE = (256, 256)   # 해상도
    BATCH_SIZE = 16           # VRAM 6GB 안전선 (32 설정 시 OOM 발생 가능성 높음)
    EPOCHS = 20               # 에폭 수
    LEARNING_RATE = 1e-4      # 학습률
    NUM_WORKERS = 2           # 놋북 CPU 발열/병목 고려 2개 프로세스 사용
    USE_AMP = True            # Automatic Mixed Precision (FP16) 가속 여부
    
    print(f"[+] 실행 디바이스: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"[+] GPU 모델: {torch.cuda.get_device_name(0)}")
        # CUDNN Benchmark를 켜서 3060 Tensor Core 성능 최적화
        torch.backends.cudnn.benchmark = True

    # -----------------------------------------------------------------
    # 예시용 더미 데이터셋 자동 생성 (실제 데이터셋 없을 경우 파이프라인 검증용)
    # -----------------------------------------------------------------
    if not os.path.exists(DATASET_ROOT):
        print(f"[!] 지정한 데이터셋 경로가 없습니다: {DATASET_ROOT}")
        print("    [테스트용 더미 데이터셋을 생성하여 파이프라인을 구동합니다]")
        os.makedirs(os.path.join(DATASET_ROOT, "real"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, "fake"), exist_ok=True)
        
        for i in range(8):
            dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(DATASET_ROOT, f"real/img_{i}.jpg"), dummy_img)
            cv2.imwrite(os.path.join(DATASET_ROOT, f"fake/img_{i}.jpg"), dummy_img)

    # 1. 데이터 수집
    all_paths, all_labels = load_dataset_paths(DATASET_ROOT)

    # 2. Train / Val / Test Split (8:1:1)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    # 3. Dataset & DataLoader (pin_memory=True 및 non_blocking으로 VRAM 전송 최적화)
    train_dataset = DeepfakeMultiStreamDataset(train_paths, train_labels, img_size=IMAGE_SIZE)
    val_dataset = DeepfakeMultiStreamDataset(val_paths, val_labels, img_size=IMAGE_SIZE)
    test_dataset = DeepfakeMultiStreamDataset(test_paths, test_labels, img_size=IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda')
    )

    # 4. 모델 & 옵티마이저 & AMP GradScaler 초기화
    model = MultiStreamDeepfakeDetectorWithAux(feature_dim=256, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # [핵심] GradScaler 초기화 (AMP용)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 5. Training Loop 실행
    best_val_auc = 0.0
    print("\n[+] RTX 3060 Laptop 최적화 학습 시작...\n")

    for epoch in range(EPOCHS):
        # AMP 학습 루프 호출
        train_loss = train_one_epoch_amp(
            model, train_loader, criterion, optimizer, scaler, DEVICE, alpha=0.3, use_amp=USE_AMP
        )
        
        # Validation 평가
        val_auc, val_eer, val_thresh = evaluate(model, val_loader, DEVICE, use_amp=USE_AMP)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val AUC: {val_auc:.4f} "
              f"| Val EER: {val_eer:.2f}%")

        # Validation AUC 기준 Best Model 저장
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = "checkpoints/best_deepfake_detector_rtx3060.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_eer': val_eer
            }, checkpoint_path)
            print(f"  └─ [Saved] 최고 성능 모델 저장 -> {checkpoint_path}")

    # 6. Final Test Set Evaluation
    print("\n[+] 학습 완료. Test 데이터셋 평가 수행...")
    if os.path.exists("checkpoints/best_deepfake_detector_rtx3060.pth"):
        checkpoint = torch.load("checkpoints/best_deepfake_detector_rtx3060.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_auc, test_eer, test_thresh = evaluate(model, test_loader, DEVICE, use_amp=USE_AMP)
    
    print("\n========================================")
    print("      FINAL TEST EVALUATION METRICS     ")
    print("========================================")
    print(f" Test AUC Score     : {test_auc:.4f}")
    print(f" Test EER (오류율)  : {test_eer:.2f}%")
    print(f" Optimal Threshold  : {test_thresh:.4f}")
    print("========================================")