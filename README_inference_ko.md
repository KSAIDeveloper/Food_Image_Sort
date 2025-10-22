# 추론 & 배포 전용 가이드 (Inference / Deployment Guide)

이 문서는 학습 완료 후 생성된 `best.pt` 와 `label_mapping.json` 을 활용하여 **단일/배치 추론**, **ONNX 변환**, **간단 서비스화(FastAPI)**, **성능/최적화 팁** 등을 한 곳에 정리한 전용 가이드입니다.

---

## 1. 준비물

- 학습 완료 산출물:
  - `outputs/best.pt`
  - `outputs/label_mapping.json`
- 추론 스크립트: `inference.py`
- (선택) ONNXRuntime, FastAPI 등 추가 패키지

설치(선택 패키지 포함):

```powershell
pip install -r requirements.txt
pip install onnxruntime fastapi uvicorn
```

---

## 2. 빠른 시작 (퀵 스타트)

단일 이미지 추론:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --image test_images\0d335021.jpg
```

폴더 전체 배치 추론:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --input_dir test_images --batch_size 128 --output inference_results.csv
```

GPU 대신 CPU 강제:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --input_dir test_images --no_cuda
```

출력 CSV(`inference_results.csv`) 컬럼:

```
image_id,label,confidence
abc.jpg,burger,0.9123
...
```

---

## 3. 옵션 정리 (`inference.py`)

| 옵션            | 설명                         | 기본값                     |
| --------------- | ---------------------------- | -------------------------- |
| --checkpoint    | 학습된 `.pt` 경로(필수)      | -                          |
| --model         | 학습 시 사용한 모델 이름     | resnet18                   |
| --image         | 단일 이미지 경로             | -                          |
| --input_dir     | 폴더 내 다수 이미지          | -                          |
| --img_size      | 입력 리사이즈 크기           | 224                        |
| --batch_size    | 배치 크기                    | 64                         |
| --output        | 결과 CSV 경로                | inference_results.csv      |
| --label_mapping | 라벨 매핑 JSON               | outputs/label_mapping.json |
| --no_cuda       | GPU 비활성화                 | False                      |
| --export_onnx   | ONNX 파일 경로 지정시 내보냄 | -                          |
| --onnx_opset    | ONNX opset 버전              | 17                         |

단일 / 폴더는 둘 다 지정 가능(합쳐짐). 하나도 없으면 에러.

---

## 4. ONNX 변환 & 활용

ONNX 내보내기:

```powershell
python inference.py --checkpoint outputs/best.pt --model resnet18 --export_onnx food_model.onnx --img_size 224
```

ONNXRuntime 설치 & 테스트:

```powershell
pip install onnxruntime
```

파이썬 간단 테스트 스니펫:

```python
import onnxruntime as ort, numpy as np, json
from PIL import Image

sess = ort.InferenceSession('food_model.onnx', providers=['CPUExecutionProvider'])
with open('outputs/label_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
idx2label = {int(k): v for k, v in mapping['idx2label'].items()}

img = Image.open('test_images/0d335021.jpg').convert('RGB').resize((224,224))
import numpy as np
arr = np.array(img).astype('float32') / 255.0
mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
arr = (arr - mean) / std
arr = arr.transpose(2,0,1)[None]
outputs = sess.run(None, {'input': arr})[0]
print(idx2label[int(outputs.argmax())])
```

### (선택) 양자화 (정적 예시)

```powershell
pip install onnxruntime-tools
python -m onnxruntime.tools.convert_onnx_models_to_ort food_model.onnx
```

---

## 5. FastAPI 간단 서비스 예시

`serve.py` (예시):

```python
from fastapi import FastAPI, UploadFile
import onnxruntime as ort, numpy as np, json
from PIL import Image

app = FastAPI()

with open('outputs/label_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
idx2label = {int(k): v for k,v in mapping['idx2label'].items()}

sess = ort.InferenceSession('food_model.onnx', providers=['CPUExecutionProvider'])

mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])

def preprocess(img: Image.Image, size=224):
    img = img.convert('RGB').resize((size,size))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2,0,1)[None]
    return arr

@app.post('/predict')
async def predict(file: UploadFile):
    img = Image.open(file.file)
    inp = preprocess(img)
    outputs = sess.run(None, {'input': inp})[0]
    idx = int(outputs.argmax())
    conf = float(outputs.max())
    return {'label': idx2label[idx], 'confidence': conf}
```

실행:

```powershell
uvicorn serve:app --host 0.0.0.0 --port 8000
```

테스트:

```powershell
Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -InFile test_images\0d335021.jpg -ContentType "application/octet-stream"
```

(또는 curl / Postman 사용)

---

## 6. 성능 & 최적화 팁

| 목표           | 방법                    | 비고                     |
| -------------- | ----------------------- | ------------------------ |
| 추론 속도      | ONNX + CPU provider     | GPU 없는 서버에서도 빠름 |
| 메모리 절약    | mobilenet_v3_small 사용 | 정확도 약간 하락 가능    |
| 정확도 유지    | efficientnet_b1 사용    | 속도/메모리 비용 증가    |
| 대량 배치      | `--batch_size` 키우기   | GPU VRAM 확인            |
| 지연(lag) 감소 | 모델 warm-up            | 첫 더미 추론 1회         |

### Warm-up 예시

```python
with torch.no_grad():
    dummy = torch.randn(1,3,224,224, device=device)
    _ = model(dummy)
```

---

## 7. 문제 해결 (Troubleshooting)

| 증상             | 원인                   | 해결                                            |
| ---------------- | ---------------------- | ----------------------------------------------- |
| GPU 사용 안 됨   | CUDA 미설치 / 드라이버 | `torch.cuda.is_available()` 확인, 드라이버 설치 |
| 라벨 엇갈림      | 잘못된 mapping 사용    | 학습시 만든 `outputs/label_mapping.json` 재사용 |
| ONNX export 실패 | opset 호환성           | `--onnx_opset 16` 등 낮춰 재시도                |
| 속도 느림        | CPU single thread      | `batch_size` 증가, ONNXRuntime 최신 버전        |
| 메모리 부족      | 너무 큰 batch          | batch 축소 or 이미지 크기 축소                  |

---

## 8. 추가 고급 기능 (향후 확장 제안)

- Ensemble: 여러 모델 결과 softmax 평균 후 label 결정
- TTA(Test-Time Augmentation): Horizontal Flip 등 2~4회 추론 평균
- Quantization Aware Training (PyTorch) 후 ONNX 변환
- Triton Inference Server 배포 (gRPC/HTTP)
- Docker 이미지로 패키징

---

## 9. TTA 간단 스니펫 (예시)

```python
from PIL import ImageOps

def tta_predict(pil_img):
    imgs = [pil_img, ImageOps.mirror(pil_img)]
    probs_sum = 0
    for im in imgs:
        t = transform(im.convert('RGB'))[None].to(device)
        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1)
        probs_sum += probs
    probs_mean = probs_sum / len(imgs)
    pred = probs_mean.argmax(dim=1).item()
    conf = probs_mean.max().item()
    return pred, conf
```

---

## 10. 유지보수 체크리스트

- 새 데이터 분포 변화 감지: 예측 확률(confidence) 평균 하락 모니터링
- 주기적 재학습: 월/분기 단위 자동 파이프라인 구성(Cron + 스크립트)
- 버전 태깅: 모델 파일명에 날짜/커밋ID (예: `best_2025-09-25.pt`)
- 로그: 추론 요청/응답 JSON 구조 저장(PII 제거)

---

## 11. 빠른 요약 TL;DR

1. `inference.py` 로 단일/배치 실행
2. 필요하면 `--export_onnx` 후 ONNXRuntime 사용
3. 서비스는 FastAPI 샘플 참고
4. 고도화: TTA, Ensemble, Quantization
5. 운영: 모니터링 + 재학습 파이프라인

---

필요한 추가(Quantization 실제 적용, Dockerfile, Ensemble 구현 등)가 있으면 이 파일 기준으로 더 확장해드릴 수 있습니다. 요청해주세요!
