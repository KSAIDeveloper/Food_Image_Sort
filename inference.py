#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
단일/배치 이미지 추론 스크립트

사용 예시:
(단일 이미지)
python inference.py --checkpoint outputs/best.pt --model resnet18 --image test_images/abcd1234.jpg

(폴더 전체 배치)
python inference.py --checkpoint outputs/best.pt --model resnet18 --input_dir test_images --batch_size 64

(출력 CSV 지정)
python inference.py --checkpoint outputs/best.pt --model resnet18 --input_dir test_images --output predictions.csv

(ONNX 내보내기)
python inference.py --checkpoint outputs/best.pt --model resnet18 --export_onnx model.onnx

주의: label_mapping.json 은 학습 시 생성된 outputs/label_mapping.json 을 사용.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------------------
# 모델 빌더 (train_food_classifier.py 와 동일 이름 유지)
# ----------------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool = False):
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
        raise ValueError(f"Unknown model: {model_name}")
    return model

# ----------------------------------------------------------------------------------
# 변환
# ----------------------------------------------------------------------------------

def build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ----------------------------------------------------------------------------------
# 추론
# ----------------------------------------------------------------------------------

def load_image(path: Path, transform):
    with Image.open(path) as img:
        img = img.convert('RGB')
    return transform(img)

def predict_batch(model, image_paths: List[Path], transform, device):
    tensors = [load_image(p, transform) for p in image_paths]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).cpu().tolist()
        conf = probs.max(dim=1).values.cpu().tolist()
    return pred_idx, conf

# ----------------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='음식 이미지 단일/배치 추론')
    ap.add_argument('--checkpoint', required=True, type=str, help='학습된 best.pt 경로')
    ap.add_argument('--model', default='resnet18', type=str)
    ap.add_argument('--image', type=str, help='단일 이미지 경로 (또는 --input_dir 사용)')
    ap.add_argument('--input_dir', type=str, help='배치 추론용 디렉토리 (jpg/png)')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--output', type=str, default='inference_results.csv')
    ap.add_argument('--label_mapping', type=str, default='outputs/label_mapping.json')
    ap.add_argument('--no_cuda', action='store_true')
    ap.add_argument('--export_onnx', type=str, help='ONNX 파일로 내보내기 경로')
    ap.add_argument('--onnx_opset', type=int, default=17)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'[INFO] Device: {device}')

    # 라벨 매핑 로드
    mapping_path = Path(args.label_mapping)
    if not mapping_path.is_file():
        print(f'[ERROR] label_mapping.json 을 찾을 수 없습니다: {mapping_path}')
        sys.exit(1)
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    idx2label = {int(k): v for k, v in mapping['idx2label'].items()}
    num_classes = len(idx2label)

    # 모델 로드
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = build_model(args.model, num_classes, pretrained=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f'[INFO] Checkpoint loaded: {args.checkpoint}')

    # ONNX 내보내기 (선택)
    if args.export_onnx:
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        onnx_path = Path(args.export_onnx)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=['input'], output_names=['logits'],
            opset_version=args.onnx_opset,
            dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        )
        print(f'[INFO] ONNX exported: {onnx_path}')

    # 입력 수집
    paths: List[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.input_dir:
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            paths.extend(Path(args.input_dir).glob(ext))
    if not paths:
        print('[ERROR] 추론할 이미지가 없습니다 (--image 또는 --input_dir 필요)')
        sys.exit(1)
    paths = sorted(set([p for p in paths if p.is_file()]))
    print(f'[INFO] Total images: {len(paths)}')

    transform = build_transform(args.img_size)

    results = []  # (path, label, confidence)
    batch_size = max(1, args.batch_size)
    for i in tqdm(range(0, len(paths), batch_size), desc='Infer'):
        batch_paths = paths[i:i+batch_size]
        pred_idx, conf = predict_batch(model, batch_paths, transform, device)
        for p, idx, cf in zip(batch_paths, pred_idx, conf):
            results.append((p.name, idx2label[idx], cf))

    # 콘솔 일부 출력
    for r in results[:5]:
        print(f'[SAMPLE] {r[0]} -> {r[1]} (conf={r[2]:.4f})')

    # CSV 저장
    import csv
    out_path = Path(args.output)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label', 'confidence'])
        writer.writerows(results)
    print(f'[INFO] 결과 저장: {out_path} (rows={len(results)})')

if __name__ == '__main__':
    main()
