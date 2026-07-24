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
    python final/llm/generate_personalized_single_drug_report.py \
      --input outputs/personalized_single_drug_198701676.json \
      --output outputs/personalized_single_drug_198701676.md \
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
당신은 의약품 복약지도서 작성 보조 시스템이다.

입력 JSON에서 대상 의약품 한 품목의 정보만 요약한다.
보유약 병용정보, 사용자 메타데이터, 개인화 주의사항은 절대 작성하지 않는다.
해당 내용은 프로그램이 별도로 작성하므로 본문에서 반복하거나 판단하지 않는다.

반드시 다음 필드만 사용한다.
- target_drug
- target_drug_specific_warnings
- target_permission_information

[작성 규칙]
- 입력 JSON에 없는 의학 지식이나 판단을 추가하지 않는다.
- 효능은 핵심 1~3문장으로 요약한다.
- 복용방법은 허가된 핵심 용법과 제형 관련 복용 주의만 요약한다.
- DUR 및 주요 주의사항은 직접 적용되는 핵심 항목을 최대 5개만 작성한다.
- DOSE_CAUTION 같은 내부 코드만 단독으로 출력하지 말고 일반인이 이해할 수 있는 한국어 제목으로 바꾼다.
- 이상반응 전체 목록이나 신체기관별 부작용 목록을 나열하지 않는다.
- 원문에 없는 약 중단, 대체약, 검사, 용량 조절을 제안하지 않는다.
- target_drug의 product_form_raw 또는 product_form_family와 다른 제형의
  효능·용법은 작성하지 않는다.
- target_drug_specific_warnings에 최대 투여기간과 최대 용량 정보가 있으면
  일반 허가 주의사항보다 우선하여 포함한다.
- 표를 사용하지 않는다.
- 전체 본문은 약 700~1,000자 내외로 작성한다.

[출력 구조]
## 3. 의약품 기본정보

## 4. 효능·효과

## 5. 복용방법

## 6. DUR 및 주요 주의사항
- 최대 5개

## 7. 보관방법

다른 번호의 절은 작성하지 않는다.
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
