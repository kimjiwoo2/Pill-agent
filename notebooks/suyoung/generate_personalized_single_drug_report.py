#!/usr/bin/env python3
"""Generate a single-drug report with deterministic safety sections.

Critical sections are rendered in Python:
- inventory interaction check
- personalized cautions
- clinician/pharmacist check list
- disclaimer

Solar summarizes only the target drug's basic information, efficacy, dosage,
DUR cautions, and storage. This prevents contradictions about interactions
or user metadata.

Usage:
    python notebooks/suyoung/generate_personalized_single_drug_report.py \
      --input notebooks/suyoung/personalized_single_drug_198701676.json \
      --output notebooks/suyoung/personalized_dinax_report_v3.md \
      --model solar-pro3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


BODY_SYSTEM_PROMPT = """
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
""".strip()


CAUTION_LABELS = {
    "AGE": "연령",
    "PREGNANCY": "임신",
    "BREASTFEEDING": "수유",
    "RENAL": "신장",
    "HEPATIC": "간",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="solar-pro3")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    required = {
        "target_drug",
        "inventory_interaction_check",
        "target_drug_specific_warnings",
        "target_permission_information",
        "personalization",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError("필수 필드 없음: " + ", ".join(sorted(missing)))

    return payload


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def target_name(payload: dict[str, Any]) -> str:
    return (
        compact(payload.get("target_drug", {}).get("item_name"))
        or "대상 의약품"
    )


def render_interaction_section(payload: dict[str, Any]) -> str:
    interaction = payload["inventory_interaction_check"]
    warnings = interaction.get("active_concomitant_warnings", []) or []

    lines = ["## 1. 보유약과의 병용 확인", ""]

    if not warnings:
        lines.append(
            "현재 등록된 보유약을 기준으로 확인된 병용금기는 없습니다. "
            "이는 모든 병용 가능성을 보장한다는 의미는 아닙니다."
        )
        return "\n".join(lines)

    lines.append("⚠️ 현재 등록된 보유약 중 병용금기가 확인되었습니다.")
    lines.append("")

    for warning in warnings:
        other = warning.get("other_drug") or warning.get("drug_b") or {}
        other_name = compact(other.get("item_name")) or "다른 보유약"
        risks = [compact(x) for x in warning.get("risks", []) if compact(x)]

        lines.append(f"- **상대 약물**: {other_name}")
        if risks:
            lines.append(f"- **확인된 위험**: {'; '.join(risks)}")

    lines.extend(
        [
            "",
            "임의로 복용을 중단하거나 변경하지 말고, 현재 보유약 정보를 "
            "의사 또는 약사에게 보여주고 확인하시기 바랍니다.",
        ]
    )
    return "\n".join(lines)


def caution_label(caution: dict[str, Any]) -> str:
    family = compact(caution.get("caution_family"))
    if family.startswith("CONDITION:"):
        return family.split(":", 1)[1]
    return CAUTION_LABELS.get(
        family,
        compact(caution.get("matched_user_value")) or "사용자 정보",
    )


def render_personalization_section(payload: dict[str, Any]) -> str:
    personalization = payload["personalization"]
    cautions = personalization.get("personalized_cautions", []) or []

    if not cautions:
        return ""

    lines = ["## 2. 사용자 정보에 따른 확인사항", ""]
    seen_labels: set[str] = set()

    for caution in cautions:
        label = caution_label(caution)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        lines.append(
            f"- 등록된 **{label} 관련 정보**와 관련된 주의 문구가 확인되었습니다."
        )

        source = compact(caution.get("source_text"))
        if source:
            lines.append(f"  - 근거 문구: {source}")

        if not bool(caution.get("user_severity_known", False)):
            lines.append(
                "  - 원문의 중증도 표현이 사용자에게 그대로 해당한다는 의미는 아닙니다."
            )

    return "\n".join(lines)


def render_check_section(payload: dict[str, Any]) -> str:
    warnings = (
        payload["inventory_interaction_check"]
        .get("active_concomitant_warnings", [])
        or []
    )
    cautions = (
        payload["personalization"]
        .get("personalized_cautions", [])
        or []
    )

    lines = ["## 8. 의사 또는 약사에게 확인할 내용", ""]
    items: list[str] = []

    for warning in warnings:
        other = warning.get("other_drug") or warning.get("drug_b") or {}
        other_name = compact(other.get("item_name")) or "다른 보유약"
        items.append(f"{other_name}과 대상 약의 병용금기 여부")

    seen_labels: set[str] = set()
    for caution in cautions:
        label = caution_label(caution)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        items.append(f"등록된 {label} 관련 정보와 허가 주의사항의 관련성")

    if not items:
        lines.append(
            "- 현재 입력 데이터에서 별도로 활성화된 병용금기나 개인화 주의는 없습니다."
        )
    else:
        for item in items:
            lines.append(f"- {item}")

    return "\n".join(lines)


def build_body_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_drug": payload["target_drug"],
        "target_drug_specific_warnings": payload[
            "target_drug_specific_warnings"
        ],
        "target_permission_information": payload[
            "target_permission_information"
        ],
    }


def generate_body(
    payload: dict[str, Any],
    model: str,
) -> tuple[str, Any]:
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY 환경변수가 없습니다.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BODY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "다음 JSON으로 대상 의약품 본문만 작성하세요. "
                    "병용정보나 사용자 개인화 내용은 절대 작성하지 마세요.\n\n"
                    + json.dumps(
                        build_body_payload(payload),
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            },
        ],
        temperature=0.1,
        stream=False,
    )

    body = response.choices[0].message.content
    if not body:
        raise RuntimeError("Solar가 빈 본문을 반환했습니다.")

    return body.strip(), response.usage


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일 없음: {input_path}")

        payload = load_payload(input_path)
        warnings = (
            payload["inventory_interaction_check"]
            .get("active_concomitant_warnings", [])
            or []
        )
        cautions = (
            payload["personalization"]
            .get("personalized_cautions", [])
            or []
        )

        print(f"[입력] {input_path}")
        print("대상 약:", target_name(payload))
        print("활성 병용금기:", len(warnings))
        print("개인화 주의 매칭:", len(cautions))
        print("[Solar] 대상 약 본문 요약 중...")

        body, usage = generate_body(payload, args.model)

        sections = [
            f"# {target_name(payload)} 복약지도서",
            render_interaction_section(payload),
        ]

        personalization_section = render_personalization_section(payload)
        if personalization_section:
            sections.append(personalization_section)

        sections.extend(
            [
                body,
                render_check_section(payload),
                (
                    "본 안내는 입력된 의약품 정보, DUR 데이터 및 사용자가 등록한 "
                    "정보를 바탕으로 생성되었으며, 의료전문가의 판단을 대체하지 않습니다."
                ),
            ]
        )

        report = "\n\n".join(section.strip() for section in sections if section.strip())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")

        print(f"[완료] {output_path}")
        if usage:
            print(
                f"[토큰] 입력={usage.prompt_tokens}, "
                f"출력={usage.completion_tokens}, "
                f"전체={usage.total_tokens}"
            )
        return 0

    except Exception as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
