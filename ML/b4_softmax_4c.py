import os
import shutil
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import copy
from PIL import Image
import timm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 4개 클래스 정의
disease_classes = ['anthracnose', 'leaf_spot', 'mixed','virus']

# 데이터셋 경로
base_dir = r'C:\Users\dore4\Desktop\cap\cropdataset'
save_dir = r'C:\Users\dore4\Desktop\crop'
results_dir = os.path.join(save_dir, 'b4_results_0625_4class_softmax')
classification_dir = os.path.join(save_dir, 'dataset_crop_0625_4class')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'weights'), exist_ok=True)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    return device

# 데이터셋 클래스 
class HierarchicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(disease_classes)}
        
        # 데이터 로딩 (하위 폴더 제거)
        for disease in disease_classes:
            disease_path = os.path.join(data_dir, disease)
            if not os.path.exists(disease_path):
                continue
                
            for img_file in os.listdir(disease_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(disease_path, img_file)
                    disease_label = self.disease_to_idx[disease]
                    self.samples.append((img_path, disease_label, disease))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, disease_label, disease_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, disease_label, disease_name

# EfficientNet 모델 
class EfficientNetModel(nn.Module):
    def __init__(self, num_diseases=3, pretrained=True):
        super(EfficientNetModel, self).__init__()
        
        # EfficientNet-B4 백본
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        
        # 직접 분류기 연결 
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_diseases)  # n개 클래스 출력
        )
    
    def forward(self, x):
        features = self.backbone(x)
        disease_logits = self.classifier(features)
        return disease_logits  # 단일 출력으로 변경

# 데이터셋 준비 함수 
def prepare_dataset():
    os.makedirs(classification_dir, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(classification_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        
        for disease in disease_classes:
            disease_dir = os.path.join(subset_dir, disease)
            os.makedirs(disease_dir, exist_ok=True)
    
    class_counts = {"train": {}, "val": {}, "test": {}}
    
    for disease in disease_classes:
        src_disease_dir = os.path.join(base_dir, disease)
        if not os.path.exists(src_disease_dir):
            print(f"경로를 찾을 수 없음: {src_disease_dir}")
            continue
        
        all_img_files = []
        
        if disease in ['anthracnose', 'leaf_spot']:
            for severity in ['mild', 'severe']:
                severity_path = os.path.join(src_disease_dir, severity)
                if os.path.exists(severity_path):
                    severity_files = [os.path.join(severity_path, f) 
                                    for f in os.listdir(severity_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                    all_img_files.extend(severity_files)
        else:
            all_img_files = [os.path.join(src_disease_dir, f) 
                           for f in os.listdir(src_disease_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        random.shuffle(all_img_files)
        train_size = int(len(all_img_files) * 0.7)
        val_size = int(len(all_img_files) * 0.2)
        
        train_files = all_img_files[:train_size]
        val_files = all_img_files[train_size:train_size+val_size]
        test_files = all_img_files[train_size+val_size:]
        
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, files in splits.items():
            dst_dir = os.path.join(classification_dir, split_name, disease)
            class_counts[split_name][disease] = len(files)
            
            print(f"{disease} {split_name}: {len(files)}개 파일")
            
            for src_path in tqdm(files, desc=f'준비 중 {split_name}/{disease}'):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(dst_dir, filename)
                shutil.copy(src_path, dst_path)
    
    plot_dataset_distribution(class_counts)
    return classification_dir

# 시각화 함수
def plot_dataset_distribution(class_counts):
    all_classes = set()
    for split_data in class_counts.values():
        all_classes.update(split_data.keys())
    all_classes = sorted(list(all_classes))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    index = np.arange(len(all_classes))
    
    train_counts = [class_counts["train"].get(cls, 0) for cls in all_classes]
    val_counts = [class_counts["val"].get(cls, 0) for cls in all_classes]
    test_counts = [class_counts["test"].get(cls, 0) for cls in all_classes]
    
    ax.bar(index, train_counts, bar_width, label='Train')
    ax.bar(index + bar_width, val_counts, bar_width, label='Validation')
    ax.bar(index + 2*bar_width, test_counts, bar_width, label='Test')
    
    ax.set_xlabel('클래스')
    ax.set_ylabel('이미지 수')
    ax.set_title('데이터셋 분포')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dataset_distribution.png'))
    plt.close()

# 데이터 로더 생성
def create_data_loaders(data_dir, batch_size=12, img_size=380):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    datasets = {x: HierarchicalDataset(os.path.join(data_dir, x), data_transforms[x]) 
                for x in ['train', 'val', 'test']}
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, 
                                shuffle=(x == 'train'), num_workers=4, pin_memory=True) 
                   for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    
    return dataloaders, dataset_sizes

#손실 함수
def hierarchical_loss(disease_pred, disease_target):
    return nn.CrossEntropyLoss()(disease_pred, disease_target)

# 모델 학습 함수 
def train_model(model, dataloaders, dataset_sizes, device, num_epochs=150, patience=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = 0
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, disease_labels, _ in tqdm(dataloaders[phase], desc=f'{phase} 처리 중'):
                inputs = inputs.to(device)
                disease_labels = disease_labels.to(device)
                
                optimizer.zero_grad()
                
                if phase == 'train':
                    with torch.cuda.amp.autocast():
                        disease_pred = model(inputs)  # 단일 출력
                        loss = hierarchical_loss(disease_pred, disease_labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    with torch.no_grad():
                        disease_pred = model(inputs)
                        loss = hierarchical_loss(disease_pred, disease_labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # 정확도 계산
                _, preds = torch.max(disease_pred, 1)
                running_corrects += torch.sum(preds == disease_labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(results_dir, 'weights', 'best_model.pt'))
                    counter = 0
                else:
                    counter += 1
        
        # 주기적 시각화
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
        
        # 조기 종료
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print()
    
    time_elapsed = time.time() - since
    print(f'학습 완료: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')
    print(f'최고 검증 정확도: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

# 시각화 함수 
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_curves.png'))
    plt.close()

#테스트 함수
def test_model(model, test_loader, test_size, device):
    model.eval()
    
    # 질병 분류 정확도
    disease_corrects = 0
    all_disease_preds = []
    all_disease_labels = []
    
    with torch.no_grad():
        for inputs, disease_labels, _ in tqdm(test_loader, desc="테스트 중"):
            inputs = inputs.to(device)
            disease_labels = disease_labels.to(device)
            
            disease_pred = model(inputs)  # 단일 출력
            
            # 질병 분류 정확도
            _, disease_preds = torch.max(disease_pred, 1)
            disease_corrects += torch.sum(disease_preds == disease_labels.data)
            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
    
    # 결과 출력
    disease_accuracy = disease_corrects.double() / test_size
    print(f'테스트 정확도: {disease_accuracy:.4f}')
    
    # 혼동 행렬
    cm = confusion_matrix(all_disease_labels, all_disease_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=disease_classes, yticklabels=disease_classes)
    plt.title('Confusion Matrix - Disease Classification')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_disease.png'))
    plt.close()
    
    # 분류 보고서
    disease_report = classification_report(all_disease_labels, all_disease_preds, target_names=disease_classes)
    print("\n질병 분류 보고서:")
    print(disease_report)
    
    # 결과 파일 저장
    try:
        with open(os.path.join(results_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("단순 분류 모델 테스트 결과\n")
            f.write("="*60 + "\n\n")
            f.write(f'테스트 정확도: {disease_accuracy:.4f}\n\n')
            f.write('분류 보고서:\n')
            f.write('-' * 40 + '\n')
            f.write(disease_report)
            f.write('\n\n')
            
        print(f"\n결과 보고서 저장됨: {os.path.join(results_dir, 'test_report.txt')}")
    except Exception as e:
        print(f"파일 저장 오류: {e}")

#예측 함수
def predict_model(model, img_path, device):
    transform = transforms.Compose([
        transforms.Resize(412),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        disease_pred = model(img_tensor)  # 단일 출력
        
        # 질병 예측
        disease_probs = torch.nn.functional.softmax(disease_pred, dim=1)[0]
        _, disease_idx = torch.max(disease_pred, 1)
        predicted_disease = disease_classes[disease_idx.item()]
        
        print(f"\n예측 결과: {os.path.basename(img_path)}")
        print("클래스 확률:")
        for i, (disease, prob) in enumerate(zip(disease_classes, disease_probs)):
            print(f"  {disease}: {prob:.2%}")
        print(f"최종 예측: {predicted_disease}")
        
        # 시각화 
        plt.figure(figsize=(10, 12))
        
        # 이미지 표시
        plt.subplot(2, 1, 1)
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.title(f"Image: {os.path.basename(img_path)}", fontsize=12)
        
        # 예측 정보 표시
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        info_text = "예측 결과:\n"
        info_text += f"• 최종 예측: {predicted_disease}\n\n"
        info_text += "클래스 확률:\n"
        for disease, prob in zip(disease_classes, disease_probs):
            info_text += f"• {disease}: {prob:.2%}\n"
        
        plt.text(0.5, 0.5, info_text, 
                 fontsize=12, 
                 ha='center', 
                 va='center',
                 bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.8))
        
        plt.suptitle(f"질병 분류 결과: {predicted_disease}", fontsize=15, fontweight='bold')
        
        # 이미지 저장
        save_path = os.path.join(results_dir, f'prediction_{os.path.basename(img_path)}')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"\n예측 결과 이미지 저장 완료: {save_path}")
        
        return predicted_disease


if __name__ == "__main__":
    print("EfficientNet-B4 모델 학습 시작...")
    
    # 데이터셋 준비
    dataset_dir = classification_dir
    if not os.path.exists(dataset_dir):
        dataset_dir = prepare_dataset()
    print(f"데이터셋 경로: {dataset_dir}")
    
    # 디바이스 설정
    device = get_device()
    
    # 데이터 로더 생성
    dataloaders, dataset_sizes = create_data_loaders(dataset_dir, batch_size=8, img_size=380)
    print(f"데이터셋 크기: {dataset_sizes}")
    
    # 모델 생성
    model = EfficientNetModel(num_diseases=len(disease_classes), pretrained=True)
    model = model.to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 모델 학습
    model = train_model(
        model, 
        dataloaders, 
        dataset_sizes, 
        device, 
        num_epochs=250, 
        patience=35
    )
    
    # 최종 모델 저장
    model_path = os.path.join(results_dir, 'weights', 'final_model.pt')
    torch.save(model.state_dict(), model_path)
    
    # 테스트
    test_model(model, dataloaders['test'], dataset_sizes['test'], device)
    
    print("==================================================")
    print("EfficientNet 단순 분류 모델 학습 완료!")
    print(f"모델 저장 경로: {model_path}")
    print("==================================================")
    
    # 예측 예시
    test_dir = os.path.join(dataset_dir, 'test')
    for disease in disease_classes:
        disease_test_dir = os.path.join(test_dir, disease)
        if os.path.exists(disease_test_dir) and os.listdir(disease_test_dir):
            test_img = os.path.join(disease_test_dir, random.choice(os.listdir(disease_test_dir)))
            predict_model(model, test_img, device)
