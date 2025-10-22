#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
멀티클래스 음식 이미지 분류 학습 / 추론 스크립트

기능 요약:
1. 데이터셋(csv + 이미지 폴더) 로드
2. Train/Validation 분할 (Stratified)
3. 데이터 증강 (Albumentations 없이 torchvision 기본 사용)
4. 사전학습(pretrained)된 ResNet18 또는 사용자 선택 모델 파인튜닝
5. 혼동행렬 / 분류리포트 출력
6. 최종 best 모델(weight) 저장
7. test.csv 기반 추론 및 sample_submission 형식 제출 파일 생성

사용 예시 (PowerShell):

# (선택) 가상환경 생성
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python train_food_classifier.py --data_root . \
    --model resnet18 \
    --img_size 224 \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --freeze_epochs 1 \
    --output_dir outputs

python train_food_classifier.py --data_root . --predict --checkpoint outputs/best.pt

"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

# -------------------------------------------------------------
# 유틸
# -------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_available_models():
    return [
        'resnet18', 'resnet34', 'resnet50',
        'mobilenet_v3_small', 'mobilenet_v3_large',
        'efficientnet_b0', 'efficientnet_b1'
    ]

# -------------------------------------------------------------
# 데이터셋
# -------------------------------------------------------------

class FoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, label2idx: dict[str, int] | None, 
                 img_size: int = 224, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.is_train = is_train
        self.label2idx = label2idx
        self.img_size = img_size

        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_rel = row['image_id']
        img_path = self.root / 'train_images' / image_rel if 'label' in row else self.root / 'test_images' / image_rel
        if not img_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        with Image.open(img_path) as img:
            img = img.convert('RGB')
        image = self.transform(img)

        if 'label' in row and self.label2idx is not None:
            label_name = row['label']
            target = self.label2idx[label_name]
            return image, target
        else:
            return image, image_rel  # 추론 시 image_id 반환

# -------------------------------------------------------------
# 모델 빌더
# -------------------------------------------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")
    return model

# -------------------------------------------------------------
# 학습 루프
# -------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum: int = 1):
    """한 epoch 학습 수행

    grad_accum > 1이면 Gradient Accumulation 적용 (실효 배치 = batch_size * grad_accum)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(tqdm(loader, desc='Train', leave=False), start=1):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets) / grad_accum
        loss.backward()

        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0) * grad_accum  # 역정규화
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    # 남은 누적이 있으면 처리
    return running_loss / max(1, total), correct / max(1, total)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Valid', leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, np.array(all_targets), np.array(all_preds)

# -------------------------------------------------------------
# 메인 실행 로직
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='음식 이미지 분류 학습 / 추론 스크립트')
    parser.add_argument('--data_root', type=str, default='.', help='데이터 루트 폴더 (train.csv가 존재하는 위치)')
    parser.add_argument('--model', type=str, default='resnet18', choices=list_available_models())
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--freeze_epochs', type=int, default=0, help='초기 몇 epoch 동안 백본 동결 (0이면 사용 안함)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--predict', action='store_true', help='추론 모드 (테스트셋 예측)')
    parser.add_argument('--checkpoint', type=str, help='불러올 학습된 모델 경로 (predict 모드 또는 이어학습)')
    parser.add_argument('--no_pretrained', action='store_true', help='사전학습 weight 사용 안함')
    parser.add_argument('--save_all', action='store_true', help='각 epoch weight 모두 저장')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient Accumulation 스텝 (실효 배치 = batch_size * grad_accum)')
    parser.add_argument('--no_cuda', action='store_true', help='강제로 CPU 사용')
    parser.add_argument('--resume', action='store_true', help='--checkpoint 로드 후 이어서 학습 (predict 아님)')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='개선 없는 Epoch 연속 N회 발생 시 조기 종료 (0=비활성)')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f"[INFO] Device: {device}")

    train_csv = data_root / 'train.csv'
    test_csv = data_root / 'test.csv'
    sample_sub_csv = data_root / 'sample_submission.csv'

    if args.predict:
        # 추론 모드
        if not args.checkpoint:
            raise ValueError('--predict 모드에서는 --checkpoint 경로가 필요합니다.')
        print('[INFO] 추론 모드 실행')
        # label mapping 복원
        mapping_path = output_dir / 'label_mapping.json'
        if not mapping_path.is_file():
            raise FileNotFoundError(f'라벨 매핑 파일을 찾을 수 없습니다: {mapping_path}')
        label_map = json.loads(mapping_path.read_text(encoding='utf-8'))
        label2idx = {k: v for k, v in label_map['label2idx'].items()}
        idx2label = {int(k): v for k, v in label_map['idx2label'].items()}

        num_classes = len(label2idx)
        model = build_model(args.model, num_classes, pretrained=not args.no_pretrained)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()

        test_df = pd.read_csv(test_csv)
        ds = FoodDataset(test_df, data_root, label2idx=None, img_size=args.img_size, is_train=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_image_ids = []
        all_preds = []
        with torch.no_grad():
            for images, image_ids in tqdm(loader, desc='Predict'):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()
                all_image_ids.extend(image_ids)
                all_preds.extend([idx2label[int(p)] for p in preds])

        sub = pd.DataFrame({'image_id': all_image_ids, 'label': all_preds})
        save_path = output_dir / 'submission.csv'
        sub.to_csv(save_path, index=False)
        print(f'[INFO] 제출 파일 저장: {save_path}')
        return

    # 학습 모드
    df = pd.read_csv(train_csv)

    # 클래스 목록 및 매핑
    classes = sorted(df['label'].unique())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}

    # 저장
    mapping_path = output_dir / 'label_mapping.json'
    mapping_path.write_text(json.dumps({'label2idx': label2idx, 'idx2label': idx2label}, ensure_ascii=False, indent=2), encoding='utf-8')

    # Stratified Split
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed, stratify=df['label'])
    print(f"[INFO] Train: {len(train_df)} / Val: {len(val_df)}")

    train_dataset = FoodDataset(train_df, data_root, label2idx, img_size=args.img_size, is_train=True)
    val_dataset = FoodDataset(val_df, data_root, label2idx, img_size=args.img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(classes)
    if not args.no_pretrained:
        print('[INFO] 사전학습 가중치가 처음이라면 다운로드(수초~수분)로 인해 멈춘 것처럼 보일 수 있습니다...')
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    start_epoch = 1
    best_acc = 0.0
    best_state = None

    # 이어학습 로드 (predict 모드가 아니고 --resume + --checkpoint 지정된 경우)
    if (not args.predict) and args.resume and args.checkpoint:
        if Path(args.checkpoint).is_file():
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            if 'val_acc' in ckpt:
                best_acc = ckpt.get('val_acc', 0.0)
            print(f"[INFO] 체크포인트 로드: {args.checkpoint} (다음 epoch={start_epoch}, best_acc={best_acc:.4f})")
        else:
            print(f"[WARN] 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.freeze_epochs))

    # 백본 동결 (초기)
    if args.freeze_epochs > 0:
        print(f'[INFO] 초기 {args.freeze_epochs} epoch 동안 feature extractor 동결')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        frozen = True
    else:
        frozen = False

    # best_acc, best_state 는 위에서 초기화 또는 재설정됨

    if args.grad_accum > 1:
        print(f'[INFO] Gradient Accumulation 사용: grad_accum={args.grad_accum} (실효 배치 = {args.batch_size * args.grad_accum})')
    if os.name == 'nt' and args.num_workers > 0:
        print('[INFO] Windows 환경에서 num_workers>0 은 초기 로딩이 더 느릴 수 있습니다. 멈춤 현상 시 --num_workers 0 권장.')

    # Early Stopping 관련
    patience = args.early_stop_patience
    epochs_no_improve = 0

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n===== Epoch {epoch}/{args.epochs} =====")
            if frozen and epoch == args.freeze_epochs + 1:
                print('[INFO] 백본 파라미터 unfrozen')
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum=args.grad_accum)
            val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']
            if (not frozen) or (epoch > args.freeze_epochs):
                scheduler.step()  # step 후 다음 epoch 적용될 lr 업데이트
            next_lr = optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | LR(now->next): {current_lr:.6f}->{next_lr:.6f}")
            print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'args': vars(args),
                    'label2idx': label2idx
                }
                best_path = output_dir / 'best.pt'
                torch.save(best_state, best_path)
                print(f'[INFO] 새로운 best 모델 저장: {best_path}')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if patience > 0:
                    print(f"[INFO] 개선 없음 {epochs_no_improve}/{patience} (early stopping 모니터링)")
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"[INFO] Early Stopping 발동 (연속 {patience}회 개선 없음) 학습 종료")
                    break

            if args.save_all:
                ckpt_path = output_dir / f'epoch{epoch:02d}.pt'
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, ckpt_path)
    except KeyboardInterrupt:
        interrupt_path = output_dir / 'interrupt.pt'
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch if 'epoch' in locals() else 0,
            'val_acc': best_acc,
            'args': vars(args),
            'label2idx': label2idx
        }, interrupt_path)
        print(f"\n[WARN] 사용자가 학습을 중단했습니다 (Ctrl+C). 임시 체크포인트 저장: {interrupt_path}")
        # 중단 시 즉시 리포트 생략 가능 (best_state 있으면 그대로 유지)


    # 최종 평가 리포트 (best 기준)
    print(f'\n[RESULT] Best Val Acc: {best_acc:.4f}')
    print('[INFO] Validation Classification Report:')
    print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(num_classes)], digits=4))
    print('[INFO] Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    # 최종 submission 생성 (선택) - 바로 테스트 예측
    if test_csv.is_file():
        print('[INFO] Best 모델로 테스트셋 예측 수행')
        # best 모델 로드
        model.load_state_dict(best_state['model'])
        model.eval()
        test_df = pd.read_csv(test_csv)
        test_dataset = FoodDataset(test_df, data_root, label2idx=None, img_size=args.img_size, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        preds_all = []
        ids_all = []
        with torch.no_grad():
            for images, img_ids in tqdm(test_loader, desc='Test Predict'):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()
                preds_all.extend([idx2label[int(p)] for p in preds])
                ids_all.extend(img_ids)
        submission = pd.DataFrame({'image_id': ids_all, 'label': preds_all})
        sub_path = output_dir / 'submission.csv'
        submission.to_csv(sub_path, index=False)
        print(f'[INFO] submission.csv 저장: {sub_path}')

    print('[DONE]')


if __name__ == '__main__':
    main()
