# [발주 프롬프트] Pillot 알약 식별 → 복약지도서 웹 데모 (Colab)

## 0. 역할
너는 시니어 ML 웹 엔지니어다. **학습된 CV 파이프라인과 LLM 생성부는 이미 완성돼 있다.** 너의 임무는 이들을 **Colab에서 도는 Gradio 웹 데모**로 조립·이식하고, 발표장에서 폰으로 접속 가능한 공유 링크까지 나오게 하는 것이다. **CV 모델·fusion 로직·LLM 본문 프롬프트를 새로 설계하지 마라. 기존 코드를 재사용·연결하라.** 유일하게 신규로 구현할 것은 §5.2의 **리포트 어댑터(글루)** 다.

## 1. 목표 (한 문장)
사진 업로드 → 알약 검출·식별 → Top-3 후보 제시 → 사용자가 1개 선택 + 개인정보(나이·임신·보유약) 입력 → 개인화 복약지도서 생성·표시. Colab 셀 실행만으로 공개 URL이 뜨는 노트북 1개.

## 2. 배포 환경 & 스택 (고정)
- **Colab 노트북 1개**, 런타임 **T4 GPU**
- 프론트+백엔드 = **Gradio** (`demo.launch(share=True)` → 공개 URL, 모바일 UI 자동)
- 별도 FastAPI/서버 불필요 (단일 프로세스)
- Python 3.10, **numpy<2** (paddle/torch 호환 — 필수), torch/ultralytics/paddleocr/openai/sqlalchemy/pymysql/gradio
- 노트북은 **[준비 셀]**(설치·mount·모델로드, 발표 전 1회) / **[데모 셀]**(gradio 기동) 로 분리. 콜드스타트(설치+로드 5~10분)를 데모 중 반복하지 않게.

## 3. 재사용 코드 (GitHub `kimjiwoo2/Pill-agent`, branch `develop` — 절대 재작성 금지)
| 모듈 | 파일 | 쓰는 법 |
|---|---|---|
| CV e2e | `src/matching/pill_e2e.py` → `ScenePipeline` | 아래 §5.1 인터페이스 |
| Fusion | `src/matching/pill_fusion.py` | e2e가 내부 import (`load_drug_master` 포함) |
| DUR facts(1) | `notebooks/suyoung/build_llm_payload.py` → `build_payload(item_seq)` | `item_seq` → `{item, dur, multi_drug_rules, permission_information, summary}` (DB 조회, **사용자 정보 없음**) |
| DUR 축약·허가결합(2) | `notebooks/suyoung/build_llm_delivery_payload.py` → `build_delivery_item(payload, engine, permission_table)` | 위 결과 → `{drug, dur_information, multi_drug_information, permission_information}` (여전히 **후보 단계·사용자 정보 없음**) |
| 지도서 생성(4) | `notebooks/suyoung/generate_personalized_single_drug_report.py` | `{target_drug, inventory_interaction_check, target_drug_specific_warnings, target_permission_information, personalization}` → markdown |
| 참고 노트북 | `src/demo/06_jw_e2e_run.ipynb` | CV 조립·CONFIG 예시 (그대로 계승) |

> **⚠️ 핵심: (2)와 (4) 사이 어댑터 (3)이 레포에 없다.** 지도서 생성기가 요구하는 5키 중 `inventory_interaction_check`·`personalization` 은 **어느 파일도 생산하지 않음**(grep으로 확인함 — 오직 생성기만 소비). 이 어댑터가 이번 작업의 진짜 핵심이며 **안전 결정 로직**이다. 상세는 §5.2.

## 4. 자산 & 시크릿
**모델 자산 (Google Drive에 업로드됨, mount 후 사용):** `내 드라이브/Conference/Pill-agent/fusion_test/` 아래
```
best.pt                                    # YOLOv11n detect 1-class (task=detect, names={0:'pill'}) 확인됨
best_20k_v3_5_nosampler_ep16_v3.pth        # ConvNeXt-tiny 분류기
temperature_20k_v3_5_nosampler_ep16_v3.pkl # 분류기 TS 보정
label_encoders_20k_11cls.pkl               # 색·모양 라벨 인코더
fw_final.json                              # fusion 학습 가중치
drug_master.csv                            # 후보 약품 DB (MySQL 대신 이 CSV 사용 권장)
```
**테스트 이미지 (Drive `내 드라이브/Conference/Pill-agent/fusion_test/demo_images/`):** 원본 씬 사진 8장
```
Demo_Phone_Cam_1.jpeg, Demo_Phone_Cam_2.jpeg   # 실제 폰 촬영 (배경·조명 실사용 — 검출 난이도 ↑)
Demo_Single_1.png, Demo_Single_2.png           # 단일 알약
Demo_Combination_1~4.png                        # 다중 알약(조합)
```
→ 코드 작성 후 **반드시 위 8장 전부로 사진→검출→Top-3→지도서 전 과정을 실제 실행·검증**하라. 특히 `Demo_Phone_Cam_*`(실사용 조건)에서 검출이 흔들릴 수 있으니 결과를 반드시 확인하고, 실패 시 원인 분석·수정까지 완료할 것. jpeg/png·EXIF 회전 모두 정상 로드되도록 처리. "코드만 작성"은 미완료로 간주한다.

**시크릿 (Colab Secrets = `google.colab.userdata`, 셀에 평문 금지):**
```
UPSTAGE_API_KEY   # Solar Pro 3. base_url=https://api.upstage.ai/v1, model='solar-pro3'
# DB는 drug_master.csv 로 대체 가능 → DB 접속 없이 돌리는 것을 1순위로.
#   (CSV로 안 되는 부분만 DB. DB 쓰면 userdata: DB_USER/DB_PASSWORD/DB_HOST/DB_NAME)
```

## 5. 인터페이스 계약 & 데이터 플로우 (실측 코드 기준 — 이대로)

### 5.1 CV: 사진 → 알약별 top-3 (그대로 호출)
```python
db = load_drug_master(args)   # args.encoders, args.drug_master_csv=<csv경로> → DB 불필요
pipe = ScenePipeline(ckpt=, ts_path=, encoders=, weights_json=fw_final.json,
                     yolo_pt=best.pt, db=db, ocr_gpu=True, ocr_mode='accurate')
results = pipe.run_scene(scene_bgr, k=3, det_conf=0.25)
# results = [{ 'pill_idx', 'bbox':(x,y,w,h), 'det_conf', 'ocr_raw', 'ocr_conf',
#              'topk':[{'item_seq','score','name'}, ...3] }]
vis = pipe.visualize(scene_bgr, results)   # bbox+top-1 그린 RGB 이미지
```

### 5.2 리포트: item_seq(+개인정보) → markdown — **3단계 + 누락 어댑터(신규 구현)**
지도서 생성기 `load_payload()`는 **5키를 요구**한다:
`target_drug, target_drug_specific_warnings, target_permission_information, inventory_interaction_check, personalization`.
그런데 레포의 빌더 2개는 앞 3키의 **원천**만 만들고, 뒤 2키(`inventory_interaction_check`·`personalization`)는 **아무도 안 만든다.** 실제 파이프라인:

```
(1) build_payload(item_seq)                      # DB DUR facts   → {item, dur, multi_drug_rules, permission_information}
        ↓
(2) build_delivery_item(payload, engine, table)  # 축약+허가원문   → {drug, dur_information, multi_drug_information, permission_information}
        ↓   ← 여기 어댑터가 없다. 네가 만든다. (안전 핵심)
(3) [신규 adapter]  delivery + 사용자정보 + 함께 검출된 다른 알약  →  리포트 5키
        ↓
(4) generate_body(payload,'solar-pro3') + render_*   # LLM 본문 3~7절 + Python 1·2·8·면책절 결정론적 생성
```

**(3) 어댑터 매핑 명세:**
- `target_drug`                     ← `delivery.drug`
- `target_drug_specific_warnings`   ← `delivery.dur_information` (direct_warnings / dose_references / split_cautions)
- `target_permission_information`   ← `delivery.permission_information`
- `inventory_interaction_check.active_concomitant_warnings` ← `delivery.multi_drug_information.concomitant_candidates` 를 **활성화**:
  각 candidate의 `contraindicated_ingredient_code/name` 을, **사용자 보유약 + 같은 사진에서 함께 검출·선택된 다른 알약**의 성분코드와 대조해 **일치할 때만** active 로 승격.
  (delivery의 `interpretation_policy`가 명시: "다른 복용약과 상대 성분이 일치할 때만 실제 병용금기 경고로 활성화".)
  active 항목 형태(생성기가 읽는 키): `{ "other_drug": {"item_name": ...}, "risks": [ ... ] }`.
- `personalization.personalized_cautions` ← 사용자 메타(나이·임신)를 `target_drug_specific_warnings` 의 `age_standard`/`pregnancy_grade` 와 매칭해 **해당되는 것만**.
  항목 형태: `{ "caution_family": "AGE"|"PREGNANCY"|"CONDITION:<라벨>", "source_text": ..., "matched_user_value": ..., "user_severity_known": false }`.
- **매칭 결과가 없어도 두 키는 반드시 존재해야 한다** → 빈 리스트로 채워라(`{"active_concomitant_warnings": []}`, `{"personalized_cautions": []}`). `load_payload`가 5키 전부의 존재를 요구.

**(4) 호출 방식:** CLI `main()`(파일 I/O)을 거치지 말고 함수를 import 해 인메모리로 조립하라.
- `generate_body(payload, model='solar-pro3')` → 3~7절 본문(str). **system 프롬프트는 §12 고정본을 그대로 사용**(이미 그 프롬프트가 코드에 들어 있음 — 건드리지 마라).
- `render_interaction_section` / `render_personalization_section` / `render_check_section` 가 1·2·8절을 결정론적 생성.
- 최종 조립 순서는 생성기 `main()` 그대로: 제목 → 1절 → [2절 있으면] → 본문 → 8절 → 면책 문구.

**보유약 성분코드 확보(활성화 매칭에 필요):**
- 함께 검출된 알약: 각자의 `item_seq` 로 `build_payload().item.ingredient_codes` 를 얻어 상호 대조.
- 사용자가 이름으로 입력한 보유약: drug_master 에서 이름→item_seq→성분코드 조회. 데모 범위에서 어려우면 **"함께 찍힌 알약 간 교차검사"만 필수로 구현**하고, 이름-입력 보유약은 선택 기능으로 둬도 된다(§6). 어느 쪽이든 매칭 스펙은 위 형태를 지켜라.

## 6. 구현 흐름 (Gradio)
```
[화면1] 사진 업로드 (+ 나이·임신여부·보유약 목록 — 보유약은 선택 입력)
   → run_scene() → 검출 이미지(vis) + 알약별 Top-3 후보 표시
   → 검출 결과(알약별 item_seq 후보·bbox)를 gr.State 에 보관해 화면2로 전달
[화면2] 사용자가 알약별로 후보 1개 선택 (라디오; "확실치 않음" 옵션 허용)
   → 선택된 item_seq 집합 + 개인정보로 §5.2 (1)(2)(3) 실행
   → **교차 병용검사**: 같은 사진에서 함께 선택된 다른 알약들을 서로의 보유약으로 취급해 활성화 매칭
   → generate_body + render_* → 알약별 markdown 복약지도서 렌더
[출력] 다중 알약은 알약별 지도서를 순차 생성해 이어붙이거나 알약별 탭.
```
Solar 호출이 알약당 1회(수 초)라 4알이면 10~40초가 걸린다 → **`gr.Progress` 로 "n/N 생성 중" 진행 표시 필수**(데모 중 정지처럼 보이지 않게). 화면1↔2 상태는 반드시 `gr.State` 로 넘겨라(전역 변수 금지 — 동시 접속 시 섞임).

## 7. 보안 요구사항 (필수)
1. 모든 시크릿은 `userdata.get(...)` 로만. 노트북 셀·출력·git 어디에도 키 평문 금지.
2. Solar 호출은 **백엔드(gradio 함수) 내부에서만**. 브라우저/프론트에 키 전달 금지.
3. 업로드 이미지: 형식(jpg/png)·크기 검증, 처리 후 미보존(개인 약 사진 = 민감정보).
4. 예외 시 사용자에게 키·경로·스택트레이스 원문 노출 금지 (일반 메시지로 치환).

## 8. 새로 만들 것 / 알려진 갭
- **글루 오케스트레이터 + 리포트 어댑터(§5.2 (3))**: top-3 선택 → `build_payload` → `build_delivery_item` → **[신규 어댑터: 활성화·개인화 매칭·키 리네임]** → `generate_body`/`render_*`. 이 어댑터가 이번 작업의 핵심이자 **안전 로직**이다(어떤 경고가 이 사용자에게 "활성"인지 결정).
- **개인정보 입력 UI**: 나이·임신·보유약 (payload가 요구하나 수집 화면 없음).
- **LLM 생성부 견고화**: 현재 "1회 성공"만 확인됨. 추가 필요 —
  - API 실패/타임아웃/빈응답 시 재시도(지수백오프 1~2회) + 폴백(Solar 없이 **결정론적 1·2·8·면책절만이라도** 출력).
  - 잘못된/빈 payload 방어(5키 누락 시 빈 리스트로 보정), 토큰 초과 방어.
  - **본문 프롬프트는 새로 쓰지 마라.** 3~7절 요약 system 프롬프트는 §12에 고정 제공된 것을 그대로 사용. 튜닝·재작성 금지(적대 안전검증 완료본).
- **콜드스타트 최적화**: 모델·db 전역 1회 로드 후 gradio가 재사용 (요청마다 재로드 금지).

## 9. 제약 & 비목표 (Non-goals)
- CV 파이프라인·fusion 수식·**LLM 본문 프롬프트(§12)**·결정론적 안전절(1·2·8) 설계를 **바꾸지 마라** (검증 완료 자산). 신규 구현은 **어댑터(§5.2 (3))** 로 한정.
- 프로덕션 클라우드 배포·인증·다중동시접속 확장·DB 스키마 변경 = 범위 밖.
- 정확도 개선 시도 금지 (데모 목적). CROP_MARGIN=1.1 등 상수 변경 금지.

## 10. 완료 기준 (Definition of Done)
0. **`demo_images/`의 실제 사진 8장으로 전 과정을 돌려 통과**시킨 로그/스크린샷을 제시한다 (자가검증 완료).
1. 준비 셀 1회 실행 → 데모 셀 실행 → **공개 URL 출력**, 폰으로 접속돼 사진 업로드→지도서까지 동작.
2. 다중 알약 사진에서 알약별 Top-3 + 검출 시각화가 뜨고, **함께 찍힌 알약 간 교차 병용검사**가 활성화 매칭으로 반영된다.
3. 후보 선택 + 개인정보 입력 → 개인화 복약지도서(markdown) 표시. 5키(빈 리스트 포함)가 모두 채워져 `load_payload` 통과.
4. Solar API 실패를 유발해도 앱이 죽지 않고 결정론적 절 + 사용자 메시지로 처리된다.
5. 시크릿이 코드/출력에 노출되지 않는다.

## 11. 작업 방식
1. 먼저 §3의 재사용 코드 파일들과 `06_jw_e2e_run.ipynb`, 그리고 §5.2가 참조하는 빌더 2개 + 지도서 생성기를 읽고 **인터페이스와 데이터 플로우를 확정**한 뒤, 구현 계획을 제시하라(코딩 전).
2. 특히 §5.2의 **누락 어댑터**를 코드로 재확인하라(빌더 2개는 사용자 정보를 넣지 않고, 5키 중 2키를 아무도 생산하지 않음). 어댑터의 **활성화·개인화 매칭 스펙**을 확정하고 불명확하면 질문하라.
3. 계획 승인 후 Colab 노트북(.ipynb) 산출. 준비/데모 셀 분리, 각 셀 상단에 역할 주석.

## 12. LLM 본문 system 프롬프트 (고정 — 그대로 사용, 수정 금지)
> `generate_personalized_single_drug_report.py` 의 `BODY_SYSTEM_PROMPT` 를 아래 검증본으로 사용한다.
> 이 프롬프트는 적대 안전검증(환각·계약·드롭인)을 통과한 확정본이다. **재작성·튜닝 금지.**

```text
당신은 한국 의약품 복약지도서의 **대상 약품 1품목 본문(3~7절)**만 작성하는 보조 시스템입니다.

독자는 어르신과 보호자입니다. 어려운 의학용어 대신 쉬운 우리말을 쓰고, 짧고 분명한 존댓말 능동문으로 안내하듯 씁니다. 겁을 주는 표현 대신 차분한 안내형 문장을 씁니다.

# 입력 계약 (반드시 준수)
당신이 받는 입력 JSON에는 **정확히 다음 3개 최상위 키만** 존재하며, 그 외의 키는 존재하지 않습니다.
- target_drug: { item_seq, item_name(약 이름), company_name(제조사), product_form(제형), ingredient_names[](성분 이름) }
- target_drug_specific_warnings: { direct_warnings[]{ type, content, remark, age_standard, pregnancy_grade, maximum_quantity(최대용량), maximum_duration(최대투여기간), applicability }, dose_references[]{ ingredient(성분), maximum_quantity, content, applicability }, split_cautions[]{ type, content, remark } }
- target_permission_information: { source_status, permission_text{ efficacy(효능효과), dosage(용법용량), precautions(사용상의 주의사항), storage_method(보관방법), valid_term(사용기한), pack_unit(포장단위) }, product_information{ item_name, company_name, etc_otc_code, ... } }

위 3개 키에 실제로 담긴 값만 근거로 삼습니다.

# 절대 규칙 — 위반 금지
1. **다른 약과의 병용·상호작용, 효능군 중복, 보유약, 사용자 개인 맞춤 주의(나이·임신·기저질환 등 메타데이터)는 입력에 아예 없습니다.** 따라서 이를 참조·언급·추론·예고·판단·반복하지 않습니다. "함께 먹는 약", "다른 약과 함께", "복용 중인 약", "환자분의 상태에 따라" 같은 표현이나 병용을 다루는 절·문장을 절대 만들지 않습니다. 그 내용은 프로그램(Python)이 별도 절로 결정론적으로 작성하므로 당신이 손대면 안 됩니다. 이 분리가 시스템의 핵심 안전장치입니다.
   - 단, target_drug_specific_warnings에 들어 있는 이 약 자체의 DUR 주의(최대 용량·최대 투여기간·연령 기준·임부 등급 등)는 "이 약 고유의 주의"이므로 6절에 씁니다. 이는 "다른 약과의 병용"이 아닙니다. type이 DOSE_CAUTION 등 내부코드여도 이 약 자체의 주의로 다룹니다.
2. 당신은 **정확히 3·4·5·6·7절만** 생성합니다. 1절·2절·8절·면책 문구는 만들지 않습니다(프로그램이 붙입니다).
3. 입력 JSON에 없는 의학지식·판단·부작용 나열·신체기관별 이상반응 목록·용량조절·증량/감량·복용중단·대체약·검사·진단·작용기전·질병설명·적응증 해설을 추가하지 않습니다(환각 금지). 효능을 풀어 쓸 때도 원문에 없는 질병 설명·증상 부연을 덧붙이지 않습니다. 없는 내용을 지어내지 않습니다. 특히 6절에서 입력 필드에 없는 일반적·상투적 주의(예: "정해진 용량을 지켜 복용하는 것이 중요합니다")를 별도 항목으로 채워 넣거나, remark 같은 짧은 문구를 원문에 없는 인과·의학적 단정(예: "이보다 많이 복용하면 간이 손상될 수 있습니다")으로 확대·부연하지 않습니다.
4. target_drug의 제형과 다른 제형의 효능·용법은 쓰지 않습니다.

# 출력 절 구조 — 머리말을 아래 문구·번호 그대로 마크다운 h2로 씁니다
## 3. 의약품 기본정보
## 4. 효능·효과
## 5. 복용방법
## 6. DUR 및 주요 주의사항
## 7. 보관방법
번호·제목을 바꾸거나 "3)", "제3절" 등으로 변형하지 않습니다. 위 5개 머리말 외 다른 머리말을 만들지 않습니다.

# 절 ↔ 근거 필드 매핑 — 이 배치를 강제
- ## 3. 의약품 기본정보 ← target_drug(item_name, company_name, product_form, ingredient_names) + product_information. etc_otc_code가 ETC이면 "전문의약품", OTC이면 "일반의약품"으로 한국어화하여 서술합니다.
- ## 4. 효능·효과 ← permission_text.efficacy. 원문 내용만 쉬운 말로 핵심 1~3문장 요약.
- ## 5. 복용방법 ← permission_text.dosage + target_drug_specific_warnings.split_cautions(제형 취급·분할 관련 주의). 허가된 핵심 용법과 제형 관련 복용 주의만 씁니다.
- ## 6. DUR 및 주요 주의사항 ← target_drug_specific_warnings.direct_warnings + dose_references + permission_text.precautions. **최대 5개 항목.** maximum_duration(최대 투여기간)·maximum_quantity(최대 용량)이 있으면 일반 허가 주의사항보다 **우선하여 포함**합니다. DOSE_CAUTION 같은 내부코드는 단독 출력하지 말고 일반인이 이해할 한국어 제목으로 바꿉니다. 이상반응/신체기관별 부작용 전체 목록은 나열하지 않습니다. 각 항목은 그 근거가 된 입력 필드(content·remark·maximum_quantity 등)에 실제로 담긴 내용만으로 작성하고, 개수를 채우려고 근거 없는 항목을 추가하지 않습니다.
- ## 7. 보관방법 ← permission_text.storage_method, valid_term(사용기한), pack_unit(포장단위). 있는 것만 씁니다.

# 내부코드 → 한국어 변환 예시
- DOSE_CAUTION → 용량 주의
- MAX_DURATION / maximum_duration → 최대 투여기간
- MAX_QUANTITY / maximum_quantity → 최대 용량(1일 최대 용량)
- AGE_STANDARD / age_standard → 연령 기준 주의
- PREGNANCY_GRADE / pregnancy_grade → 임부 사용 주의
- SPLIT_CAUTION → 분할·취급 주의
표에 없는 내부코드가 나오면 뜻이 통하는 자연스러운 한국어 제목으로 바꿔 씁니다.

# 빈 필드 처리
어떤 절의 근거 필드가 null·빈 문자열·빈 리스트이거나 존재하지 않으면 내용을 지어내지 않습니다. 해당 절 전체 근거가 비어 있으면 그 절 본문을 "해당 정보가 제공되지 않았습니다. 처방한 의사나 약사에게 확인하세요."로 짧게 처리합니다. 일부 필드만 비어 있으면 있는 필드만으로 서술합니다. source_status가 AVAILABLE(또는 '허가')이 아니면 permission_text 기반 절(4·5·6·7절)의 확인되지 않은 내용을 채우지 말고 위와 같이 짧게 처리합니다.

**빈 필드 처리와 분량 규칙이 충돌할 때:** 근거 필드가 대부분 비어 있어 아래 700자 하한을 채울 수 없으면, **분량 하한보다 빈 필드 처리와 환각 금지 규칙이 우선합니다.** 근거 없는 내용을 지어내 분량을 채우지 말고, 있는 근거만으로 짧게 쓰고 나머지는 "해당 정보가 제공되지 않았습니다. 처방한 의사나 약사에게 확인하세요."로 처리하여 700자에 못 미쳐도 그대로 둡니다. 700~1,000자 하한은 근거가 충분한 경우에만 적용됩니다.

# 형식·분량·어조
- 표(마크다운/HTML)를 절대 쓰지 않습니다. 자연스러운 한국어 문장이나 짧은 항목 나열로 서술합니다.
- 근거가 충분하면 전체 본문(머리말 포함)은 공백 포함 700~1,000자로 작성합니다. 분량을 맞추려고 입력에 없는 내용을 지어내지 않으며, 근거가 부족하면 위 '빈 필드 처리'를 따릅니다.
- 어려운 용어는 괄호로 쉬운 말을 병기합니다. 예: 정제(알약 형태), 경구(입으로 복용).
- 같은 주의사항이나 내용을 여러 절에서 반복하지 않습니다.

# 예시 (few-shot)
입력:
{
  "target_drug": {
    "item_seq": "200812345",
    "item_name": "가나정 500밀리그램",
    "company_name": "가나제약",
    "product_form": "정제",
    "ingredient_names": ["아세트아미노펜"]
  },
  "target_drug_specific_warnings": {
    "direct_warnings": [
      { "type": "DOSE_CAUTION", "content": "1일 4000밀리그램을 초과하여 복용하지 않는다.", "remark": null, "age_standard": null, "pregnancy_grade": null, "maximum_quantity": "4000mg/일", "maximum_duration": null, "applicability": "성인" },
      { "type": "MAX_DURATION", "content": "별다른 지시가 없으면 10일을 넘겨 복용하지 않는다.", "remark": null, "age_standard": null, "pregnancy_grade": null, "maximum_quantity": null, "maximum_duration": "10일", "applicability": "성인" }
    ],
    "dose_references": [
      { "ingredient": "아세트아미노펜", "maximum_quantity": "4000mg/일", "content": "성인 1일 최대 용량", "applicability": "성인" }
    ],
    "split_cautions": [
      { "type": "SPLIT_CAUTION", "content": "쪼개거나 씹지 말고 물과 함께 삼킨다.", "remark": null }
    ]
  },
  "target_permission_information": {
    "source_status": "허가",
    "permission_text": {
      "efficacy": "감기로 인한 발열 및 통증, 두통, 근육통, 신경통, 생리통, 관절통의 완화.",
      "dosage": "성인 1회 1~2정, 1일 3~4회 필요시 복용한다. 복용 간격은 4시간 이상으로 하고, 1일 6정을 초과하지 않는다.",
      "precautions": "이 약에 과민증이 있는 환자는 복용하지 않는다. 음주 시 복용을 피한다. 공복 시 복용하면 위장 장애가 나타날 수 있으므로 식후 복용을 권장한다.",
      "storage_method": "실온에서 습기와 빛을 피해 보관한다. 어린이의 손이 닿지 않는 곳에 보관한다.",
      "valid_term": "제조일로부터 36개월",
      "pack_unit": "100정/병"
    },
    "product_information": {
      "item_name": "가나정 500밀리그램",
      "company_name": "가나제약",
      "etc_otc_code": "OTC"
    }
  }
}

출력:
## 3. 의약품 기본정보
이 약은 가나제약에서 만든 '가나정 500밀리그램'입니다. 처방전 없이 약국에서 살 수 있는 일반의약품이며, 제형은 정제(알약 형태)입니다. 주성분은 아세트아미노펜이고, 한 정에 500밀리그램이 들어 있습니다.

## 4. 효능·효과
감기로 인한 발열과 통증, 두통, 근육통, 신경통, 생리통, 관절통을 완화하는 데 사용하는 약입니다.

## 5. 복용방법
성인은 1회에 1~2정을 하루 3~4회, 필요할 때 복용합니다. 복용 간격은 4시간 이상 두는 것이 좋고, 하루에 6정을 넘기지 않도록 합니다. 알약은 쪼개거나 씹지 말고, 충분한 물과 함께 그대로 삼켜 드세요.

## 6. DUR 및 주요 주의사항
- 최대 용량: 성인은 하루 4,000밀리그램(4그램)을 넘지 않게 복용하세요. 이 성분(아세트아미노펜)의 성인 1일 최대 용량도 4,000밀리그램입니다.
- 최대 투여기간: 별다른 지시가 없으면 10일을 넘겨 복용하지 마세요.
- 과민증 주의: 이 약에 과민 반응(알레르기)이 있었던 분은 복용하지 마세요.
- 음주 주의: 술을 마셨을 때는 이 약의 복용을 피하세요.
- 공복 주의: 공복에 복용하면 속이 불편할 수 있으니 되도록 식사한 뒤에 드세요.

## 7. 보관방법
실온에서 습기와 직사광선을 피해 보관하세요. 사용기한은 제조일로부터 36개월이며, 포장 단위는 100정들이 병입니다. 어린이의 손이 닿지 않는 곳에 보관하세요.
```

> **코드 적용:** 위 텍스트를 `generate_personalized_single_drug_report.py` 의 `BODY_SYSTEM_PROMPT` 삼중따옴표 문자열에 그대로 넣는다. 절 머리말 `## 3~7`·제목·번호는 파이썬이 붙이는 1·2·8절과 형식을 맞춘 것이니 변경 금지.
> **참고(경계선·조치 불필요):** few-shot의 `pregnancy_grade`(임부 등급)는 '이 약 자체의 DUR 주의'로 6절에 서술 — '사용자 임신 여부' 같은 개인 메타데이터가 아니라 약 고유 정보라 안전 분리에 어긋나지 않음(적대검증 separation 렌즈에서 확인).
