# Pill-agent

**투빅스 25기 컨퍼런스 프로젝트** | 유주형 · 한수영 · 김지우 · 조윤수

AI 기반 다중 의약품 인식 및 복약 지도 자동 생성 시스템

---

## 프로젝트 소개

여러 약을 함께 복용하는 다약제 복용(polypharmacy) 환경에서 약물 간 상호작용은 조합이 늘어날수록 기하급수적으로 복잡해진다. 스마트폰으로 알약 사진을 촬영하면, AI가 약물을 식별하고 DUR 데이터를 기반으로 상호작용을 분석하여 보호자·어르신이 이해하기 쉬운 맞춤형 복약 지도서를 자동 생성한다.

---

## 시스템 파이프라인

```
알약 사진 입력
      ↓
[1] Detection — YOLOv11n으로 알약 bbox 검출 → margin crop 생성
      ↓
[2] ID (병렬)
      ├─ 속성 분류기 (EfficientNet-B3, color/shape 2-head)
      └─ OCR (EasyOCR, 각인 텍스트)
            ↓ 앙상블 → 후보 약품명 추론
      ↓
[3] Knowledge Retrieval — DUR DB에서 DDI·병용금기·중복효능 검색
      ↓
[4] Context Assembly — 검색 결과 + 사용자 메타데이터(기저질환 등)
      ↓
[5] Report Generation — LLM이 맞춤형 복약 지도 리포트 생성
```

---

## 모델

### Detection — YOLOv11n

| 항목 | 값 |
|---|---|
| 모델 | YOLOv11n (COCO pretrained) |
| 태스크 | 1-class detection (알약 위치 검출) |
| 학습 데이터 | AI Hub 단일 경구약제 (TS_1, TS_2 / val: VS_1) |
| 입력 크기 | 640px |
| 하이퍼파라미터 | AdamW lr=0.002, 30 epochs, batch=16 |

bbox 검출 후 20% margin을 추가해 정사각형으로 확장한 crop을 분류 및 OCR 입력으로 사용한다.

### 속성 분류기 — EfficientNet-B3 (2-head)

| 항목 | 값 |
|---|---|
| 모델 | EfficientNet-B3 (timm, ImageNet pretrained) |
| 입력 크기 | 300px |
| 출력 | color head (30 classes) + shape head (11 classes) |
| 색상 라벨 | 하양·노랑·분홍·주황 등 30종 |
| 형태 라벨 | 원형·장방형·타원형·팔각형 등 11종 |
| Loss | CE_color + CE_shape |
| Checkpoint 기준 | mean(val macro-F1_color, val macro-F1_shape) |

### OCR — EasyOCR (예정)

각인 텍스트를 인식하여 속성 분류기 점수와 앙상블, 최종 약품 식별에 활용한다.

---

## 데이터 및 전처리

### 데이터셋 — AI Hub 경구약제 이미지 (과제번호 576)

| 구분 | 수량 |
|---|---|
| 단일 경구약제 train | 81 zip |
| 단일 경구약제 validation | 10 zip |
| 조합 경구약제 train | 8 zip |
| 조합 경구약제 validation | 1 zip |
| Detection manifest 전체 | 2,663,439행 |
| Classification manifest (train) | 2,451,927행 |
| 약품 품목 수 | 4,522종 |

AI Hub 라벨 JSON을 파싱하여 MySQL DB에 적재했다.

| 테이블 | 단위 | 행 수 | 설명 |
|---|---|---|---|
| `drug_master` | 품목(item_seq) | 4,522 | 약품 기본정보 (색상·형태·각인 등) |
| `aihub_images` | 이미지 | 2,663,619 | 파일명, split, 촬영조건, 약품 연결 |
| `aihub_annotations` | bbox | 2,663,439 | bbox 좌표 및 category_id |

### Split 전략

**Detection:** AI Hub 기본 split(`split_type` 컬럼)을 그대로 승계한다.

**Classification:** crop manifest를 `StratifiedGroupKFold(n_splits=5, 80/20)`으로 내부 분할한다.

- **group key:** `image_file` — 같은 원본 이미지의 crop이 train/val에 섞이지 않도록 보장
- **stratify key:** `color_class1 + '_' + drug_shape` 조합 — 두 속성의 클래스 분포를 동시에 유지

---

## 진행 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | YOLO Detection 베이스라인 | 노트북 완료, 학습 실행 대기 |
| 2 | Crop 파이프라인 | 노트북 완료, 학습 실행 대기 |
| 3 | 속성 분류기 EfficientNet-B3 2-head | 노트북 완료, 학습 실행 대기 |
| 4 | EasyOCR 프로토타입 | 미완료 |
| 5 | 속성 분류기 + OCR 앙상블 | 미완료 |
