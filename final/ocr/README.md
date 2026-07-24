# OCR 파트 (`final/ocr/`)

Fine-tuning 없이 **PP-OCRv5 pretrained 모델**에 추론 단계 전처리·회전 보정 설계만 얹은 최종 채택
파이프라인. 전처리 → confidence 기반 multi-angle 회전 보정 → detection/recognition → (DB 매칭용
CER 기반 채점) 순서로 동작한다.

## 구성

다른 파트와 동일하게 **파일 하나**(`__init__.py`)로 정리했다. 내부는 5개 섹션으로 순서대로
나뉘어 있음(주석으로 구분):

| 섹션 | 역할 |
| --- | --- |
| 1. Preprocessing | CLAHE+Unsharp, 회전, 방향 정렬(`align_to_long_axis`), 업스케일 — OCR 입력 전용 이미지 전처리 |
| 2. Detection | `rotate_only` — PaddleOCR `TextDetection`으로 텍스트 poly를 찾아 회전각 미세보정(크롭 없음) |
| 3. Orientation | `resolve_180_flip` — PaddleOCR `TextLineOrientationClassification`으로 0/180도 방향 보정 |
| 4. Matching | `normalize_imprint`/`score_one`/`exact_match` — OCR 인식 결과와 DB 후보 각인 텍스트 간 CER 기반 채점. 최종 후보 선택(top-3 랭킹)은 fusion 모듈이 이 점수를 가져다 수행 — 여기는 채점만 담당 |
| 5. Pipeline | `OCRPipeline`, `resolve_rotation_by_confidence`, `create_ocr` — 위 조각들을 엮은 최종 추론 진입점(모델 로드/전처리/회전탐색/fallback/결과파싱) |

`from final.ocr import ...`로 가져오는 이름들은 파일을 합치기 전과 동일하다(인터페이스 변경 없음).

## 필요한 pip 패키지

```txt
paddlepaddle-gpu==3.1.0
paddleocr==3.7.0
opencv-python-headless
numpy
```

**설치 전 주의 (Colab 등 torch가 이미 깔린 환경)**: `paddleocr` → `paddlex` → `modelscope`로
이어지는 의존성 체인이 무조건 torch를 함께 끌고 오는데, 환경에 이미 깔린 torch의 CUDA/NCCL
버전과 충돌하면 `ImportError: undefined symbol: ncclCommShrink`가 발생한다. 아래 순서로 설치할 것:

```bash
pip uninstall -y torch torchvision torchaudio modelscope
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==3.7.0
pip install opencv-python-headless numpy
```

## 드라이브 파일 및 배치 경로

**없음.** OCR 파트는 pretrained 모델만 사용하므로 별도로 받아야 할 커스텀 가중치나 데이터가
없다 — `OCRPipeline()`을 처음 생성할 때 PaddleOCR가 필요한 사전학습 가중치
(`PP-OCRv5_server_det`, `PP-OCRv5_server_rec`, `PP-LCNet_x1_0_textline_ori`)를 자동으로
다운로드한다(최초 실행 시 인터넷 연결 필요, 이후는 로컬 캐시 사용).

## 시크릿

없음.

## 실행 순서

```python
from final.ocr import OCRPipeline, score_one

# 1) 모델 로드 (요청마다 새로 만들지 말고, 데모/서버 시작 시 한 번만 생성)
pipeline = OCRPipeline(device="gpu")  # GPU 없으면 device="cpu" (속도 크게 저하됨)

# 2) 알약 crop 이미지 1장 인식
result = pipeline.predict("path/to/pill_crop.png")
print(result.ocr_text_norm, result.ocr_conf)

# 3) DB 후보군과 매칭할 때 (다른 파트의 Late Fusion/DB 매칭 단계 입력값)
candidates = ["KL250", "MG"]  # 해당 약제의 DB상 앞/뒷면 인쇄 후보
score = score_one(result.ocr_text_norm, candidates)
```

- `OCRPipeline.predict()`는 이미지 경로(str/Path) 또는 이미 로드된 BGR `numpy.ndarray`(예: 실시간
  카메라 프레임)를 모두 받는다.
- `result.used_fallback=True`이면 12개 회전 후보 탐색이 모두 실패해 보정 없이 원본 전체 이미지로
  추론한 결과라는 뜻 — 그래도 완전히 포기하지 않고 최선의 결과를 반환한다.
