#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
새 클래스(예: tteokbokki, ramen)를 손쉽게 추가하는 도구.
- 소스 폴더(인터넷에서 수집한 이미지 모음 등)에서 jpg/png를 모아
  train_images/<class_name>/ 로 복사하고
- train.csv 에 해당 항목을 자동으로 append 합니다.

사용 예시 (PowerShell):
python dataset_tools/add_class_images.py --data_root . --class_name tteokbokki --src_dir C:/tmp/tteokbokki_imgs

여러 클래스 한 번에:
python dataset_tools/add_class_images.py --data_root . --bulk_json classes.json
(classes.json 예시)
{
  "tteokbokki": "C:/tmp/tteokbokki_imgs",
  "ramen": "D:/imgs/ramen"
}

주의: 이미지 파일명 충돌 시 자동으로 번호 suffix를 붙여 저장합니다.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

def safe_copy(src: Path, dst: Path) -> Path:
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    base = dst.stem
    ext = dst.suffix.lower()
    i = 0
    out = dst
    while out.exists():
        i += 1
        out = dst_parent / f"{base}_{i}{ext}"
    shutil.copy2(src, out)
    return out


def add_one_class(data_root: Path, class_name: str, src_dir: Path) -> int:
    train_dir = data_root / 'train_images' / class_name
    train_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in src_dir.rglob('*'):
        if p.suffix.lower() in VALID_EXT and p.is_file():
            dst = train_dir / p.name
            out = safe_copy(p, dst)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description='새 클래스 이미지 추가 및 train.csv 갱신')
    ap.add_argument('--data_root', type=str, default='.')
    ap.add_argument('--class_name', type=str, help='단일 클래스 이름 (예: tteokbokki)')
    ap.add_argument('--src_dir', type=str, help='단일 클래스 소스 이미지 폴더')
    ap.add_argument('--bulk_json', type=str, help='여러 클래스 일괄 추가용 JSON 경로')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_csv = data_root / 'train.csv'
    if not train_csv.is_file():
        raise FileNotFoundError(f'train.csv not found at {train_csv}')

    df = pd.read_csv(train_csv)
    added_rows = []

    if args.bulk_json:
        mapping = json.loads(Path(args.bulk_json).read_text(encoding='utf-8'))
        for cls, src in mapping.items():
            cnt = add_one_class(data_root, cls, Path(src))
            print(f"[INFO] {cls}: {cnt} images copied")
            # 새로 복사된 파일 목록을 다시 스캔하여 csv 항목 생성
            train_dir = data_root / 'train_images' / cls
            for p in train_dir.glob('*'):
                if p.suffix.lower() in VALID_EXT:
                    rel = f"{cls}/{p.name}"
                    if not ((df['image_id'] == rel) & (df['label'] == cls)).any():
                        added_rows.append({'image_id': rel, 'label': cls})
    else:
        if not args.class_name or not args.src_dir:
            ap.error('--class_name 과 --src_dir 또는 --bulk_json 중 하나는 반드시 지정해야 합니다')
        cls = args.class_name
        cnt = add_one_class(data_root, cls, Path(args.src_dir))
        print(f"[INFO] {cls}: {cnt} images copied")
        train_dir = data_root / 'train_images' / cls
        for p in train_dir.glob('*'):
            if p.suffix.lower() in VALID_EXT:
                rel = f"{cls}/{p.name}"
                if not ((df['image_id'] == rel) & (df['label'] == cls)).any():
                    added_rows.append({'image_id': rel, 'label': cls})

    if added_rows:
        df_new = pd.concat([df, pd.DataFrame(added_rows)], ignore_index=True)
        df_new.to_csv(train_csv, index=False)
        print(f"[INFO] train.csv updated: +{len(added_rows)} rows")
    else:
        print('[INFO] 추가된 새 항목이 없습니다 (이미 등록되었을 수 있음)')

if __name__ == '__main__':
    main()
