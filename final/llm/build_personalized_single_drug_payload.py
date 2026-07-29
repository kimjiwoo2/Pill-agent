#!/usr/bin/env python3
"""Build a personalized single-drug report payload.

Current personalization scope:
- age
- pregnancy
- breastfeeding
- renal impairment
- hepatic impairment
- registered conditions
- active personal-medication interactions

Allergy matching is intentionally excluded from this version.

Example:
    python final/llm/build_personalized_single_drug_payload.py \
      --user-id test_user \
      --item-seq 198701676 \
      --output outputs/personalized_single_drug_198701676.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pymysql

from build_personal_medication_payload import get_connection, json_default
from build_single_drug_report_payload import build_payload as build_single_payload


CONDITION_ALIASES: dict[str, list[str]] = {
    "고혈압": ["고혈압"],
    "심부전": ["심부전", "심기능부전"],
    "심혈관질환": ["심혈관", "심근경색", "뇌졸중"],
    "신장질환": ["신장애", "신부전", "신기능", "신장"],
    "간질환": ["간장애", "간부전", "간기능", "간염", "간장"],
    "위궤양": ["소화성 궤양", "위장관 궤양", "궤양"],
    "위장관출혈": ["위장관 출혈", "출혈"],
    "천식": ["천식"],
    "크론병": ["크론병"],
    "궤양성대장염": ["궤양성 대장염"],
}

# Structured flags and free-text conditions that represent the same concept.
CONDITION_FAMILY: dict[str, str] = {
    "신장질환": "RENAL",
    "간질환": "HEPATIC",
}


def calculate_age(birth_date: date | None) -> int | None:
    if not birth_date:
        return None
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def compact_text(*values: Any) -> str:
    return "\n".join(clean(value) for value in values if clean(value))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_context_units(text: str) -> list[str]:
    """Split long permission text into conservative sentence/line-sized units."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。])\s+|\n+|(?<=다\.)\s*", text)
    return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]


def find_context_excerpt(
    text: str,
    keywords: list[str],
    max_length: int = 280,
) -> tuple[str, str] | None:
    """
    Return the first sentence/line containing one of the keywords.
    Falls back to a short local excerpt when sentence splitting is insufficient.
    """
    for unit in split_context_units(text):
        for keyword in keywords:
            if keyword.lower() in unit.lower():
                return keyword, unit[:max_length]

    lowered = text.lower()
    for keyword in keywords:
        idx = lowered.find(keyword.lower())
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(text), idx + len(keyword) + 160)
            excerpt = normalize_whitespace(text[start:end])
            if start > 0:
                excerpt = "…" + excerpt
            if end < len(text):
                excerpt += "…"
            return keyword, excerpt[:max_length]

    return None


def fetch_user_metadata(cursor, user_id: str) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT
            user_id,
            birth_date,
            sex,
            pregnancy_status,
            breastfeeding_status,
            renal_impairment,
            hepatic_impairment
        FROM user_medical_profile
        WHERE user_id = %s
        LIMIT 1
        """,
        (user_id,),
    )
    profile = cursor.fetchone()

    cursor.execute(
        """
        SELECT condition_code, condition_name
        FROM user_condition
        WHERE user_id = %s
          AND is_active = 1
        ORDER BY user_condition_id
        """,
        (user_id,),
    )
    conditions = list(cursor.fetchall())

    profile_dict = dict(profile) if profile else {
        "user_id": user_id,
        "birth_date": None,
        "sex": "UNKNOWN",
        "pregnancy_status": "UNKNOWN",
        "breastfeeding_status": "UNKNOWN",
        "renal_impairment": None,
        "hepatic_impairment": None,
    }
    profile_dict["age"] = calculate_age(profile_dict.get("birth_date"))
    profile_dict["conditions"] = [dict(row) for row in conditions]

    # Allergy personalization is intentionally excluded in this version.
    profile_dict["allergy_personalization_enabled"] = False
    return profile_dict


def add_caution(
    cautions: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    caution_type: str,
    caution_family: str,
    matched_value: Any,
    source_section: str,
    source_text: str,
    reason: str,
    user_severity_known: bool = False,
) -> None:
    source_text = normalize_whitespace(source_text)
    if not source_text:
        return

    # Deduplicate by clinical concept family and normalized source text.
    key = (caution_family, source_text.lower())
    if key in seen:
        return
    seen.add(key)

    cautions.append(
        {
            "type": caution_type,
            "caution_family": caution_family,
            "matched_user_value": matched_value,
            "match_reason": reason,
            "source_section": source_section,
            "source_text": source_text,
            "user_severity_known": user_severity_known,
            "safe_expression": (
                "등록된 사용자 정보와 관련된 주의 문구가 확인되었습니다. "
                "원문에 중증도 표현이 있어도 사용자 중증도는 별도로 확인되지 않았습니다."
                if not user_severity_known
                else "등록된 사용자 정보와 관련된 주의 문구가 확인되었습니다."
            ),
            "interpretation": (
                "사용자 메타데이터와 대상 의약품 원문을 보수적으로 매칭한 결과이며, "
                "진단·중증도 판정·처방 변경의 근거가 아니다."
            ),
        }
    )


def match_direct_warnings(
    profile: dict[str, Any],
    warnings: list[dict[str, Any]],
    cautions: list[dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    age = profile.get("age")
    pregnancy = profile.get("pregnancy_status")
    breastfeeding = profile.get("breastfeeding_status")

    for warning in warnings:
        text = compact_text(
            warning.get("dur_type"),
            warning.get("prohibition_content"),
            warning.get("remark"),
            warning.get("age_base_raw"),
            warning.get("pregnancy_grade"),
        )
        if not text:
            continue

        if age is not None and age >= 65 and any(
            keyword in text for keyword in ("고령자", "노인", "65세")
        ):
            add_caution(
                cautions,
                seen,
                caution_type="AGE_CAUTION",
                caution_family="AGE",
                matched_value=age,
                source_section="target_drug_specific_warnings",
                source_text=text,
                reason="등록 나이가 고령자 관련 DUR 문구와 일치",
                user_severity_known=True,
            )

        if pregnancy == "PREGNANT" and any(
            keyword in text for keyword in ("임부", "임신", "태아")
        ):
            add_caution(
                cautions,
                seen,
                caution_type="PREGNANCY_CAUTION",
                caution_family="PREGNANCY",
                matched_value=pregnancy,
                source_section="target_drug_specific_warnings",
                source_text=text,
                reason="등록된 임신 상태가 임신 관련 DUR 문구와 일치",
                user_severity_known=True,
            )

        if breastfeeding == "YES" and any(
            keyword in text for keyword in ("수유", "수유부")
        ):
            add_caution(
                cautions,
                seen,
                caution_type="BREASTFEEDING_CAUTION",
                caution_family="BREASTFEEDING",
                matched_value=breastfeeding,
                source_section="target_drug_specific_warnings",
                source_text=text,
                reason="등록된 수유 상태가 수유 관련 DUR 문구와 일치",
                user_severity_known=True,
            )


def match_permission_text(
    profile: dict[str, Any],
    precautions: str,
    cautions: list[dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    if not precautions:
        return

    # Structured renal/hepatic flags are processed first so equivalent
    # free-text conditions are automatically deduplicated by caution_family.
    structured_checks = [
        (
            bool(profile.get("renal_impairment")),
            "RENAL_CAUTION",
            "RENAL",
            "renal_impairment=true",
            ["신장애", "신부전", "신기능", "신장"],
            "등록된 신장 관련 정보와 허가 주의사항 문구가 일치",
        ),
        (
            bool(profile.get("hepatic_impairment")),
            "HEPATIC_CAUTION",
            "HEPATIC",
            "hepatic_impairment=true",
            ["간장애", "간부전", "간기능", "간염", "간장"],
            "등록된 간 관련 정보와 허가 주의사항 문구가 일치",
        ),
    ]

    for enabled, caution_type, family, value, keywords, reason in structured_checks:
        if not enabled:
            continue
        match = find_context_excerpt(precautions, keywords)
        if match:
            _, source = match
            add_caution(
                cautions,
                seen,
                caution_type=caution_type,
                caution_family=family,
                matched_value=value,
                source_section=(
                    "target_permission_information.permission_text.precautions"
                ),
                source_text=source,
                reason=reason,
                user_severity_known=False,
            )

    for condition in profile.get("conditions", []):
        name = clean(condition.get("condition_name"))
        if not name:
            continue

        family = CONDITION_FAMILY.get(name, f"CONDITION:{name}")
        keywords = CONDITION_ALIASES.get(name, [name])
        match = find_context_excerpt(precautions, keywords)
        if not match:
            continue

        keyword, source = match
        add_caution(
            cautions,
            seen,
            caution_type="CONDITION_CAUTION",
            caution_family=family,
            matched_value=name,
            source_section=(
                "target_permission_information.permission_text.precautions"
            ),
            source_text=source,
            reason=f"등록 질환 '{name}'이 허가 주의사항의 '{keyword}' 문구와 일치",
            user_severity_known=False,
        )


def build_relevant_context(
    profile: dict[str, Any],
    cautions: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant_types = {item["type"] for item in cautions}
    context: dict[str, Any] = {}

    if "AGE_CAUTION" in relevant_types:
        context["age"] = profile.get("age")
    if "PREGNANCY_CAUTION" in relevant_types:
        context["pregnancy_status"] = profile.get("pregnancy_status")
    if "BREASTFEEDING_CAUTION" in relevant_types:
        context["breastfeeding_status"] = profile.get("breastfeeding_status")
    if "RENAL_CAUTION" in relevant_types:
        context["renal_impairment"] = True
    if "HEPATIC_CAUTION" in relevant_types:
        context["hepatic_impairment"] = True

    matched_conditions = sorted(
        {
            str(item["matched_user_value"])
            for item in cautions
            if item["type"] == "CONDITION_CAUTION"
        }
    )
    if matched_conditions:
        context["matched_conditions"] = matched_conditions

    return context


def build_personalized_payload(
    user_id: str,
    target_item_seq: str,
) -> dict[str, Any]:
    payload = build_single_payload(user_id, target_item_seq)

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            profile = fetch_user_metadata(cursor, user_id)
    finally:
        connection.close()

    warnings = payload.get("target_drug_specific_warnings", [])
    permission = payload.get("target_permission_information", {})
    permission_text = permission.get("permission_text", {}) or {}
    precautions = clean(permission_text.get("precautions"))

    cautions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    match_direct_warnings(profile, warnings, cautions, seen)
    match_permission_text(profile, precautions, cautions, seen)

    payload["schema"] = "personalized-single-drug-report-v2"
    payload["personalization"] = {
        "metadata_found": any(
            [
                profile.get("birth_date"),
                profile.get("sex") not in (None, "UNKNOWN"),
                profile.get("pregnancy_status") not in (None, "UNKNOWN"),
                profile.get("breastfeeding_status") not in (None, "UNKNOWN"),
                profile.get("renal_impairment") is not None,
                profile.get("hepatic_impairment") is not None,
                profile.get("conditions"),
            ]
        ),
        "allergy_personalization_enabled": False,
        "relevant_user_context": build_relevant_context(profile, cautions),
        "personalized_cautions": cautions,
        "match_count": len(cautions),
        "policy": {
            "only_source_grounded_matches": True,
            "no_diagnosis_or_dose_decision": True,
            "do_not_infer_user_severity": True,
            "deduplicate_same_clinical_family": True,
            "unmatched_metadata_not_sent_to_report_model": True,
            "allergy_matching_excluded": True,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사용자 메타데이터를 반영한 단일 의약품 payload를 생성합니다."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--item-seq", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    try:
        payload = build_personalized_payload(args.user_id, str(args.item_seq))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )

        personalization = payload["personalization"]
        print(f"[완료] {output_path}")
        print("대상 약:", payload["target_drug"].get("item_name"))
        print(
            "활성 병용금기:",
            len(
                payload["inventory_interaction_check"].get(
                    "active_concomitant_warnings",
                    [],
                )
            ),
        )
        print("사용자 메타데이터 존재:", personalization["metadata_found"])
        print("개인화 주의 매칭:", personalization["match_count"])
        print("알레르기 개인화:", "제외")
        return 0

    except pymysql.MySQLError as exc:
        print(f"[DB 오류] {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[실행 오류] {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
