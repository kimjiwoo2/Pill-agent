#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor


def json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def split_multi_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_values = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            raw_values = re.split(r"\s*(?:,|\||;|\n)\s*", text)

    result, seen = [], set()
    for raw in raw_values:
        item = str(raw).strip().strip('"').strip("'")
        if item and item.lower() not in {"null", "none"} and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_connection() -> pymysql.Connection:
    required = ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("필수 DB 환경변수가 없습니다: " + ", ".join(missing))
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


def placeholders(n: int) -> str:
    return ", ".join(["%s"] * n)


def fetch_active_medications(cursor: DictCursor, user_id: str) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT personal_medication_id, user_id, item_seq, item_name, company_name,
               medication_status, identification_source, confirmed_by_user,
               started_at, ended_at, dose_text, frequency_text,
               administration_time_text, created_at, updated_at
        FROM personal_medication
        WHERE user_id = %s
          AND medication_status = 'ACTIVE'
          AND confirmed_by_user = 1
          AND (started_at IS NULL OR started_at <= CURDATE())
          AND (ended_at IS NULL OR ended_at >= CURDATE())
        ORDER BY personal_medication_id
        """,
        (user_id,),
    )
    return list(cursor.fetchall())


def fetch_direct_rows(cursor: DictCursor, item_seqs: list[str]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT item_seq, item_name, company_name, product_form_raw,
               product_form_family, dur_type, rule_id,
               primary_ingredient_code, product_ingredient_codes,
               product_ingredient_names, applicability, prohibition_content,
               remark, max_quantity_raw, max_duration_raw, age_base_raw,
               pregnancy_grade, warning_level
        FROM svc_dur_item_caution_v3_cache
        WHERE item_seq IN ({placeholders(len(item_seqs))})
          AND COALESCE(is_active, 1) = 1
          AND applicability IN ('DIRECT', 'DIRECT_APPLICABLE')
        ORDER BY item_seq, dur_type, rule_id
        """,
        item_seqs,
    )
    return list(cursor.fetchall())


def fetch_concomitant_rows(cursor: DictCursor, item_seqs: list[str]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT item_seq, item_name, company_name, form_name, dur_type, rule_id,
               ingredient_code, ingredient_name, ingredient_name_en,
               mixture_item_seq, mixture_item_name,
               mixture_ingredient_code, mixture_ingredient_name,
               canonical_pair_key, prohibition_content, remark,
               notification_date, mapping_strategy, mapping_direction,
               ingredient_mapping_status, mixture_mapping_status,
               evidence_row_count
        FROM svc_dur_concomitant_effective
        WHERE item_seq IN ({placeholders(len(item_seqs))})
           OR mixture_item_seq IN ({placeholders(len(item_seqs))})
        ORDER BY rule_id, item_seq, mixture_item_seq
        """,
        item_seqs + item_seqs,
    )
    return list(cursor.fetchall())


def fetch_therapeutic_rows(cursor: DictCursor, item_seqs: list[str]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT item_seq, item_name, company_name, form_name, dur_type, rule_id,
               ingredient_code, ingredient_name, ingredient_name_en,
               mixture_item_seq, mixture_item_name,
               mixture_ingredient_code, mixture_ingredient_name,
               canonical_pair_key, prohibition_content, remark,
               effect_code, series_name, notification_date,
               mapping_strategy, mapping_direction,
               ingredient_mapping_status, mixture_mapping_status,
               evidence_row_count
        FROM svc_dur_therapeutic_duplication_v3
        WHERE COALESCE(is_active, 1) = 1
          AND (
                item_seq IN ({placeholders(len(item_seqs))})
             OR mixture_item_seq IN ({placeholders(len(item_seqs))})
          )
        ORDER BY rule_id, item_seq, mixture_item_seq
        """,
        item_seqs + item_seqs,
    )
    return list(cursor.fetchall())


def fetch_permission_rows(cursor: DictCursor, item_seqs: list[str]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT item_seq, item_name, company_name, permit_date, etc_otc_code,
               main_item_ingredient, ingredient_name, material_name,
               efficacy_text, dosage_text, precautions_text, storage_method,
               valid_term, pack_unit, edi_code, atc_code, fetch_status
        FROM drug_permission_info
        WHERE item_seq IN ({placeholders(len(item_seqs))})
        ORDER BY item_seq
        """,
        item_seqs,
    )
    return list(cursor.fetchall())


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def build_indices(medications, direct_rows):
    medication_by_item = {str(m["item_seq"]): dict(m) for m in medications}
    ingredient_codes_by_item = defaultdict(set)
    ingredient_names_by_item = defaultdict(set)

    for row in direct_rows:
        item_seq = str(row["item_seq"])
        if clean_text(row.get("primary_ingredient_code")):
            ingredient_codes_by_item[item_seq].add(clean_text(row["primary_ingredient_code"]))
        ingredient_codes_by_item[item_seq].update(split_multi_value(row.get("product_ingredient_codes")))
        ingredient_names_by_item[item_seq].update(split_multi_value(row.get("product_ingredient_names")))

        med = medication_by_item.get(item_seq)
        if med:
            med["item_name"] = med.get("item_name") or row.get("item_name")
            med["company_name"] = med.get("company_name") or row.get("company_name")
            med.setdefault("product_form_raw", row.get("product_form_raw"))
            med.setdefault("product_form_family", row.get("product_form_family"))

    return medication_by_item, ingredient_codes_by_item, ingredient_names_by_item


def build_drug_specific_warnings(item_seqs, direct_rows):
    grouped = defaultdict(list)
    seen = set()
    for row in direct_rows:
        key = (
            str(row["item_seq"]),
            str(row.get("dur_type") or ""),
            str(row.get("rule_id") or ""),
            str(row.get("prohibition_content") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        grouped[str(row["item_seq"])].append(
            {
                "dur_type": row.get("dur_type"),
                "rule_id": row.get("rule_id"),
                "applicability": row.get("applicability"),
                "warning_level": row.get("warning_level"),
                "prohibition_content": row.get("prohibition_content"),
                "remark": row.get("remark"),
                "max_quantity_raw": row.get("max_quantity_raw"),
                "max_duration_raw": row.get("max_duration_raw"),
                "age_base_raw": row.get("age_base_raw"),
                "pregnancy_grade": row.get("pregnancy_grade"),
            }
        )
    return [{"item_seq": item_seq, "direct_warnings": grouped.get(item_seq, [])} for item_seq in item_seqs]


def match_pairs(row, medication_by_item, ingredient_codes_by_item):
    active_items = set(medication_by_item)
    current_item = clean_text(row.get("item_seq"))
    mixture_item = clean_text(row.get("mixture_item_seq"))
    current_code = clean_text(row.get("ingredient_code"))
    mixture_code = clean_text(row.get("mixture_ingredient_code"))
    pairs = set()

    if current_item in active_items and mixture_item in active_items and current_item != mixture_item:
        pairs.add(normalize_pair(current_item, mixture_item))

    if current_item in active_items and mixture_code:
        for other_item, codes in ingredient_codes_by_item.items():
            if other_item != current_item and mixture_code in codes:
                pairs.add(normalize_pair(current_item, other_item))

    if mixture_item in active_items and current_code:
        for other_item, codes in ingredient_codes_by_item.items():
            if other_item != mixture_item and current_code in codes:
                pairs.add(normalize_pair(mixture_item, other_item))

    return pairs


def build_active_concomitant(rows, medication_by_item, ingredient_codes_by_item):
    grouped = {}
    for row in rows:
        for item_a, item_b in match_pairs(row, medication_by_item, ingredient_codes_by_item):
            key = (item_a, item_b, str(row.get("canonical_pair_key") or row.get("rule_id") or ""))
            if key not in grouped:
                grouped[key] = {
                    "status": "ACTIVE",
                    "type": "CONCOMITANT_CONTRAINDICATION",
                    "drug_a": {
                        "item_seq": item_a,
                        "item_name": medication_by_item[item_a].get("item_name"),
                        "company_name": medication_by_item[item_a].get("company_name"),
                    },
                    "drug_b": {
                        "item_seq": item_b,
                        "item_name": medication_by_item[item_b].get("item_name"),
                        "company_name": medication_by_item[item_b].get("company_name"),
                    },
                    "rule_ids": [],
                    "ingredient_pairs": [],
                    "risks": [],
                    "remarks": [],
                }
            entry = grouped[key]
            rule_id = clean_text(row.get("rule_id"))
            if rule_id and rule_id not in entry["rule_ids"]:
                entry["rule_ids"].append(rule_id)
            ingredient_pair = {
                "ingredient_code": clean_text(row.get("ingredient_code")),
                "ingredient_name": clean_text(row.get("ingredient_name")),
                "mixture_ingredient_code": clean_text(row.get("mixture_ingredient_code")),
                "mixture_ingredient_name": clean_text(row.get("mixture_ingredient_name")),
            }
            if ingredient_pair not in entry["ingredient_pairs"]:
                entry["ingredient_pairs"].append(ingredient_pair)
            risk = clean_text(row.get("prohibition_content"))
            if risk and risk not in entry["risks"]:
                entry["risks"].append(risk)
            remark = clean_text(row.get("remark"))
            if remark and remark not in entry["remarks"]:
                entry["remarks"].append(remark)
    return list(grouped.values())


def build_active_therapeutic(rows, medication_by_item, ingredient_codes_by_item):
    grouped = {}
    for row in rows:
        for item_a, item_b in match_pairs(row, medication_by_item, ingredient_codes_by_item):
            effect_code = clean_text(row.get("effect_code")) or ""
            series_name = clean_text(row.get("series_name")) or ""
            key = (item_a, item_b, effect_code, series_name)
            if key not in grouped:
                grouped[key] = {
                    "status": "ACTIVE",
                    "type": "THERAPEUTIC_DUPLICATION",
                    "drug_a": {
                        "item_seq": item_a,
                        "item_name": medication_by_item[item_a].get("item_name"),
                    },
                    "drug_b": {
                        "item_seq": item_b,
                        "item_name": medication_by_item[item_b].get("item_name"),
                    },
                    "effect_code": effect_code or None,
                    "series_name": series_name or None,
                    "rule_ids": [],
                    "risks": [],
                    "remarks": [],
                }
            entry = grouped[key]
            rule_id = clean_text(row.get("rule_id"))
            if rule_id and rule_id not in entry["rule_ids"]:
                entry["rule_ids"].append(rule_id)
            risk = clean_text(row.get("prohibition_content"))
            if risk and risk not in entry["risks"]:
                entry["risks"].append(risk)
            remark = clean_text(row.get("remark"))
            if remark and remark not in entry["remarks"]:
                entry["remarks"].append(remark)
    return list(grouped.values())


def build_permission_payload(item_seqs, rows):
    by_item = {str(row["item_seq"]): row for row in rows}
    result = []
    for item_seq in item_seqs:
        row = by_item.get(item_seq)
        if not row:
            result.append({"item_seq": item_seq, "source_status": "NOT_FOUND", "permission_text": {}})
            continue
        result.append(
            {
                "item_seq": item_seq,
                "source_status": row.get("fetch_status"),
                "product_information": {
                    "item_name": row.get("item_name"),
                    "company_name": row.get("company_name"),
                    "permit_date": row.get("permit_date"),
                    "etc_otc_code": row.get("etc_otc_code"),
                    "main_item_ingredient": row.get("main_item_ingredient"),
                    "ingredient_name": row.get("ingredient_name"),
                    "material_name": row.get("material_name"),
                    "edi_code": row.get("edi_code"),
                    "atc_code": row.get("atc_code"),
                },
                "permission_text": {
                    "efficacy": row.get("efficacy_text"),
                    "dosage": row.get("dosage_text"),
                    "precautions": row.get("precautions_text"),
                    "storage_method": row.get("storage_method"),
                    "valid_term": row.get("valid_term"),
                    "pack_unit": row.get("pack_unit"),
                },
            }
        )
    return result


def build_payload(user_id: str) -> dict[str, Any]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            medications = fetch_active_medications(cursor, user_id)
            if not medications:
                return {
                    "schema": "personal-medication-analysis-v1",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "user_id": user_id,
                    "analysis_status": "NO_ACTIVE_MEDICATIONS",
                    "active_medications": [],
                    "drug_specific_warnings": [],
                    "active_concomitant_warnings": [],
                    "active_therapeutic_duplications": [],
                    "permission_information": [],
                }

            item_seqs = [str(row["item_seq"]) for row in medications]
            direct_rows = fetch_direct_rows(cursor, item_seqs)
            concomitant_rows = fetch_concomitant_rows(cursor, item_seqs)
            therapeutic_rows = fetch_therapeutic_rows(cursor, item_seqs)
            permission_rows = fetch_permission_rows(cursor, item_seqs)

        medication_by_item, ingredient_codes_by_item, ingredient_names_by_item = build_indices(
            medications, direct_rows
        )

        active_medications = []
        for item_seq in item_seqs:
            med = medication_by_item[item_seq]
            active_medications.append(
                {
                    "personal_medication_id": med.get("personal_medication_id"),
                    "item_seq": item_seq,
                    "item_name": med.get("item_name"),
                    "company_name": med.get("company_name"),
                    "product_form_raw": med.get("product_form_raw"),
                    "product_form_family": med.get("product_form_family"),
                    "ingredient_codes": sorted(ingredient_codes_by_item.get(item_seq, set())),
                    "ingredient_names": sorted(ingredient_names_by_item.get(item_seq, set())),
                    "dose_text": med.get("dose_text"),
                    "frequency_text": med.get("frequency_text"),
                    "administration_time_text": med.get("administration_time_text"),
                    "started_at": med.get("started_at"),
                    "ended_at": med.get("ended_at"),
                    "identification_source": med.get("identification_source"),
                    "confirmed_by_user": bool(med.get("confirmed_by_user")),
                }
            )

        active_concomitant = build_active_concomitant(
            concomitant_rows, medication_by_item, ingredient_codes_by_item
        )
        active_therapeutic = build_active_therapeutic(
            therapeutic_rows, medication_by_item, ingredient_codes_by_item
        )

        return {
            "schema": "personal-medication-analysis-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id,
            "analysis_status": "COMPLETED",
            "interpretation_policy": {
                "drug_specific_warnings": "각 품목에 직접 적용되는 DUR 경고이다.",
                "active_concomitant_warnings": "현재 활성 복용약 목록에서 실제 품목 또는 상대 성분이 확인된 병용금기이다.",
                "active_therapeutic_duplications": "현재 활성 복용약 목록에서 확인된 효능군 중복 후보이다. 최종 판단은 전문가 확인이 필요하다.",
                "permission_information": "품목별 허가정보 평문 원문이다.",
            },
            "analysis_summary": {
                "active_medication_count": len(active_medications),
                "active_concomitant_warning_count": len(active_concomitant),
                "active_therapeutic_duplication_count": len(active_therapeutic),
                "confirmed_medications_only": True,
                "active_date_filter_applied": True,
            },
            "active_medications": active_medications,
            "drug_specific_warnings": build_drug_specific_warnings(item_seqs, direct_rows),
            "active_concomitant_warnings": active_concomitant,
            "active_therapeutic_duplications": active_therapeutic,
            "permission_information": build_permission_payload(item_seqs, permission_rows),
        }
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="personal_medication 기준 다중 약물 DUR 분석 JSON 생성"
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    try:
        payload = build_payload(args.user_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        summary = payload.get("analysis_summary", {})
        print(f"[완료] {output_path}")
        print("활성 복용약:", summary.get("active_medication_count", 0))
        print("활성 병용금기:", summary.get("active_concomitant_warning_count", 0))
        print("활성 효능군 중복:", summary.get("active_therapeutic_duplication_count", 0))
        return 0
    except pymysql.MySQLError as exc:
        print(f"[DB 오류] {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[실행 오류] {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
