# PILLAR

**투빅스 25기 컨퍼런스 프로젝트** | 유주형 · 한수영 · 김지우 · 조윤수

AI 기반 다중 의약품 인식 및 복약 지도 자동 생성 시스템

---

## 프로젝트 소개

여러 약을 함께 복용하는 다약제 복용(polypharmacy) 환경에서 약물 간 상호작용은 조합이 늘어날수록 복잡해진다. 특히 조제된 약이 약봉투(정보)를 떠나 보관되는 순간, "무슨 약인지·무엇을 주의해야 하는지"를 확인할 방법이 사라진다.

PILLAR는 스마트폰으로 알약 사진을 촬영하면 AI가 약물을 식별하고, DUR 데이터를 기반으로 상호작용을 분석해 보호자·어르신이 이해하기 쉬운 맞춤형 복약 지도서를 자동 생성한다.

---

## 시스템 파이프라인

```
알약 사진 입력
      ↓
[1] Detection — YOLOv11n 1-class 검출 → margin crop
      ↓
[2] ID (병렬)
      ├─ 속성 분류기 (ConvNeXt-Tiny, color/shape 2-head)
      └─ OCR (PaddleOCR PP-OCRv5, 각인 텍스트)
            ↓
[3] Learned Late Fusion — 조건부 로짓으로 학습된 가중 결합 → 후보 Top-3
      ↓
[4] Human-in-the-loop — 사용자가 후보 중 직접 확인
      ↓
[5] Knowledge Retrieval — DUR DB에서 병용금기·용량·기간 검색
      ↓
[6] Report Generation — LLM(Solar)이 맞춤형 복약 지도서 생성
```

---

## 저장소 구조

```
final/                  # 최종 코드
├── detection/          # YOLO 검출 학습
├── classification/     # 색·모양 2-head 분류기 학습
├── ocr/                # 각인 인식 파이프라인
├── matching/           # 융합·랭킹·E2E 오케스트레이션
└── demo/               # 웹 데모 (Colab + Gradio)  ※ 커밋 예정
notebooks/              # 실험·학습 노트북 (출력 제거본)
```

데이터·모델 가중치·데모 자산은 **git에 포함하지 않으며 Google Drive에 보관**한다.

---

## 실행 준비 (Setup)

### 1) 패키지 설치
```bash
pip install -r requirements.txt
```

### 2) Drive 자산 배치
Colab에서 Drive를 mount한 뒤, 아래 파일이 지정 경로에 있어야 한다.

**데모 실행에 필요 (`fusion_test/`)**

| 파일 | 용도 |
|---|---|
| `best.pt` | YOLOv11n 검출 가중치 |
| `best_20k_v3_5_nosampler_ep16_v3.pth` | 색·모양 분류기 |
| `temperature_20k_v3_5_nosampler_ep16_v3.pkl` | 분류기 보정(TS) |
| `label_encoders_20k_11cls.pkl` | 색/모양 라벨 인코더 |
| `drug_master.csv` | 약품 기본정보 (색·형태·각인) |
| `candidate_images/` | 후보 알약 참조 사진 |
| `demo_images/` | 데모 입력 사진 |

> 융합 가중치 `fw_final.json`은 `final/matching/`에 포함되어 있다.

**학습·평가 재현에 추가로 필요**

| 파일 | 용도 |
|---|---|
| `manifest_clean_20k_33340.csv` | YOLO 학습 / Fusion val manifest |
| `images_train.zip`, `images_val.zip` | YOLO 학습 이미지 |
| `test_filtered.zip` | Test crop |
| `final_test_manifest.csv` | Test 정답 라벨 |
| `ocr_result_final_v1_test.csv` | Test OCR 결과 |

### 3) 시크릿
값은 저장소에 두지 않는다. Colab Secrets(🔑) 또는 환경변수로 등록한다.

| 이름 | 용도 |
|---|---|
| `UPSTAGE_API_KEY` | 복약지도서 생성 (Solar LLM) |
| `DB_USER`, `DB_PASSWORD` | DUR DB 접속 |
| `PILLIOT_DB_HOST`, `PILLIOT_DB_USER`, `PILLIOT_PW` | Fusion의 DB 직접 조회 시 |
| `NALAL_API_KEY` (선택) | 식약처 낱알식별 이미지 조회 |

---

## 실행 방법

### 웹 데모 (End-to-End)
Colab에서 `final/demo/`의 데모 노트북을 열고(런타임: T4 GPU) 셀 순서대로 실행한다.
설치 → 시크릿 → 자산 검증 → 모델 로드 → 데모 기동.

### YOLO 검출 학습 — `final/detection/yolo_detect_train.py`
1-class 축정렬 bbox YOLOv11n 학습.
```bash
python final/detection/yolo_detect_train.py --mode all \
  --data-root <데이터 루트> --name yolo11n_detect_v1
```
manifest 파일명이 다르면 `--manifest <파일명>`으로 지정한다.

### Fusion 평가 — `final/matching/pill_fusion.py`
색·모양 + 각인 융합으로 후보 약품 Top-k 랭킹.
```bash
python final/matching/pill_fusion.py --split test \
  --drug-master-csv drug_master.csv --encoders label_encoders_20k_11cls.pkl \
  --ckpt best_20k_v3_5_nosampler_ep16_v3.pth --ts temperature_...pkl \
  --labels-csv final_test_manifest.csv --crops test_filtered.zip \
  --ocr-csv ocr_result_final_v1_test.csv --weights final/matching/fw_final.json
```

---

## 데이터

**AI Hub 경구약제 이미지** (과제번호 576). 라벨 JSON을 파싱해 MySQL DB에 적재했다.

| 테이블 | 단위 | 행 수 |
|---|---|---|
| `drug_master` | 품목(item_seq) | 4,522 |
| `aihub_images` | 이미지 | 2,663,619 |
| `aihub_annotations` | bbox | 2,663,439 |

Split은 데이터 누수를 막기 위해 품목 코드 단위로 분리하거나 AI Hub 기본 split을 승계한다.

---

## 자산 접근 문의

데이터셋·모델 가중치·DB 접근 권한이 필요한 경우 담당자에게 요청한다.

- Drive 자산(모델 가중치·학습 데이터·데모 이미지): **유주형**
- DB 자산(DUR DB·manifest·export 이미지): **한수영**
