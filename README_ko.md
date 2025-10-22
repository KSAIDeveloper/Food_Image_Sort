# 멀티클래스 음식 이미지 분류

## 개요

6~10개 음식 카테고리(예: burger, pizza, salad, sushi, dessert, soup 등)에 대해 사진 이미지를 분류하는 PyTorch 기반 학습 스크립트입니다. `train.csv` / `test.csv` 와 이미지 폴더 구조를 이용해 모델을 학습하고 추론하여 `submission.csv` 를 생성합니다.

## 데이터 구조 예시

```
project_root/
  train.csv
  test.csv
  sample_submission.csv
  train_images/
    burger/0001_burger.jpg
    burger/...
    pizza/0012_pizza.jpg
    ...
  test_images/
    a1b2c3d4.jpg
    ...
```

- `train.csv`: 두 컬럼 (`image_id`, `label`) - `image_id` 는 `train_images/` 내의 상대 경로 (예: `burger/0001_burger.jpg`)
- `test.csv`: 하나의 컬럼 (`image_id`) - `test_images/` 내부 파일명
- `sample_submission.csv`: 제출 포맷 참고용

## 설치

(선택) 가상환경 생성 후 라이브러리 설치:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# 추가 모델 원하면
# pip install timm
```

## 학습 실행

```powershell
python train_food_classifier.py --data_root . ^
  --model resnet18 ^
  --img_size 224 ^
  --epochs 10 ^
  --batch_size 32 ^
  --lr 3e-4 ^
  --freeze_epochs 1 ^
  --output_dir outputs
```

PowerShell에서 여러 줄 명령은 `^` 를 사용합니다.

### 주요 인자 설명

- `--model`: resnet18 | resnet34 | resnet50 | mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | efficientnet_b1
- `--freeze_epochs`: 초기 feature extractor 동결 epoch 수
- `--val_split`: 검증 셋 비율 (기본 0.15)
- `--no_pretrained`: 사전학습 가중치 사용 안함
- `--save_all`: 모든 epoch 체크포인트 저장
- `--grad_accum`: 큰 배치를 흉내 내려는 Gradient Accumulation (현재 기본 loop에는 미적용, 필요 시 확장 가능)

## 추론(테스트 예측만) 실행

이미 학습된 체크포인트(`outputs/best.pt`)를 사용하여 test.csv 예측만 수행:

```powershell
python train_food_classifier.py --data_root . --predict --checkpoint outputs/best.pt --model resnet18
```

결과: `outputs/submission.csv`

## 출력물

- `outputs/label_mapping.json`: 클래스 인덱스-이름 매핑
- `outputs/best.pt`: 최고 검증 정확도 모델
- `outputs/submission.csv`: 테스트셋 예측 결과

## 추론 전용 스크립트 (`inference.py`)

학습 스크립트에 의존하지 않고 빠르게 단일/배치/폴더 추론 및 ONNX 내보내기를 할 수 있는 경량 스크립트.

예시(단일 이미지):

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --image test_images/0d335021.jpg
```

폴더 전체:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --input_dir test_images --batch_size 128 --output inference_results.csv
```

ONNX 변환:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --export_onnx food_model.onnx --img_size 224
```

ONNX 추론은 다른 환경(예: C#, Node.js, 모바일)에서 `onnxruntime` 으로 실행 가능.

## 간단한 서비스화 아이디어

1. ONNX 변환 후 `onnxruntime` 로 Python FastAPI 엔드포인트 구현
2. 배치 처리: 이미지를 폴더에 떨어뜨리면 크론(또는 스케줄러)이 `inference.py` 실행
3. Edge 디바이스: `efficientnet_b0` / `mobilenet_v3_small` 로 경량화 + ONNX + quantization (추후 확장)

### FastAPI 예시 스니펫 (추후 참고)

```python
from fastapi import FastAPI, UploadFile
import onnxruntime as ort, numpy as np
from PIL import Image

app = FastAPI()
sess = ort.InferenceSession('food_model.onnx', providers=['CPUExecutionProvider'])

def preprocess(img: Image.Image):
  img = img.resize((224, 224))
  arr = np.array(img.convert('RGB')).astype('float32') / 255.0
  mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
  arr = (arr - mean) / std
  arr = arr.transpose(2,0,1)[None]
  return arr

@app.post('/predict')
async def predict(file: UploadFile):
  img = Image.open(file.file)
  inp = preprocess(img)
  outputs = sess.run(None, {'input': inp})[0]
  idx = int(outputs.argmax())
  # idx2label.json 로드하여 라벨 변환 (생략)
  return {'pred_index': idx}
```

## 자주 겪는 문제(트러블슈팅)

| 증상                                  | 원인                                       | 해결                                                           |
| ------------------------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| 실행 초반 멈춘 것처럼 보임            | 첫 사전학습(Pretrained) weight 다운로드 중 | 인터넷 연결 확인, 혹은 `--no_pretrained` 사용                  |
| Windows에서 DataLoader 생성 직후 멈춤 | `num_workers` 프로세스 스폰 지연           | `--num_workers 0` 로 테스트 후 점진 증가                       |
| CUDA out of memory                    | GPU 메모리 부족                            | `--batch_size` 축소, `--img_size` 224→192, `--grad_accum` 활용 |
| 학습이 너무 느림                      | CPU 사용 중이거나 증강 과다                | CUDA 활성(`torch.cuda.is_available()`), 증강 줄이기            |
| Validation 정확도 안 올라감           | 과적합/학습률 부적절                       | `--lr` 조정(1e-3↔3e-4), `--freeze_epochs` 줄이기, 증강 강화    |

### Gradient Accumulation

`--grad_accum N` 을 주면 (실효배치 = `batch_size * N`). GPU 메모리가 작아 큰 배치를 못 넣을 때 유용.

예) 실제 128 배치 효과를 내려면:

```powershell
python train_food_classifier.py --batch_size 32 --grad_accum 4 ...
```

### 빠른 sanity check

```powershell
python train_food_classifier.py --epochs 1 --batch_size 8 --num_workers 0 --data_root . --model resnet18 --img_size 128
```

문제 없으면 파라미터를 점진적으로 키우세요.

## 커스텀

- 증강 강도를 줄이고 싶다면 `FoodDataset` 내 `ColorJitter`, `RandomRotation` 제거 가능
- 클래스 불균형이 심하면 `WeightedRandomSampler` 추가 고려
- F1-score 최적화를 위해 `classification_report` 참고

## 라이선스 / 주의

데이터의 사용 권한은 원본 출처를 따릅니다. 본 스크립트는 연구/학습용 예시입니다.

## 문의

필요한 추가 기능(Grad Accumulation 실제 적용, k-Fold, Mixup/Cutmix 등)이 있다면 요청해주세요.
