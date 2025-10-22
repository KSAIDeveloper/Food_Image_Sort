#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
새로 추가한 클래스의 이미지가 너무 많거나 검증셋을 분리하고 싶을 때 사용.
- train_images/<class>/ 에 있는 이미지 중 일부를 val_images/<class>/ 로 이동하고,
- 별도 CSV(train_extra.csv, val_extra.csv)를 만들어 합쳐 쓸 수 있게 합니다.

기본 학습 스크립트는 train_images/ 만 사용하므로,
검증 분할을 직접 관리하고 싶을 때만 이 도구를 사용하세요.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import shutil
import csv

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def main():
    ap = argparse.ArgumentParser(description='새 클래스 수동 train/val 분할')
    ap.add_argument('--data_root', type=str, default='.')
    ap.add_argument('--class_name', required=True, type=str)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    cls = args.class_name

    src_dir = data_root / 'train_images' / cls
    val_dir = data_root / 'val_images' / cls
    val_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in src_dir.glob('*') if p.suffix.lower() in VALID_EXT]
    n_val = int(len(imgs) * args.val_ratio)
    random.shuffle(imgs)
    val_imgs = imgs[:n_val]

    moved = []
    for p in val_imgs:
        dst = val_dir / p.name
        shutil.move(str(p), str(dst))
        moved.append(dst.name)

    # CSV 기록
    with open(data_root / 'train_extra.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'label'])
        for p in src_dir.glob('*'):
            if p.suffix.lower() in VALID_EXT:
                w.writerow([f'{cls}/{p.name}', cls])

    with open(data_root / 'val_extra.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'label'])
        for name in moved:
            w.writerow([f'{cls}/{name}', cls])

    print(f'[INFO] moved {len(moved)} files to {val_dir}')
    print('[INFO] train_extra.csv / val_extra.csv 생성 완료')

if __name__ == '__main__':
    main()
