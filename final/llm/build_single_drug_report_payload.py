#!/usr/bin/env python3
"""Build a single-drug report payload with interaction checks against a user's active medications.

Place this file beside build_personal_medication_payload.py.

Example:
    python final/llm/build_single_drug_report_payload.py \
      --user-id test_user \
      --item-seq 198701676 \
      --output outputs/single_drug_198701676.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pymysql

from build_personal_medication_payload import (
    build_active_concomitant,
    build_active_therapeutic,
    build_drug_specific_warnings,
    build_indices,
    build_permission_payload,
    fetch_active_medications,
    fetch_concomitant_rows,
    fetch_direct_rows,
    fetch_permission_rows,
    fetch_therapeutic_rows,
    get_connection,
    json_default,
)


def fetch_target_basic(cursor, item_seq: str) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT item_seq, item_name, company_name
        FROM drug_permission_info
        WHERE item_seq COLLATE utf8mb4_0900_ai_ci = %s COLLATE utf8mb4_0900_ai_ci
        LIMIT 1
        """,
        (item_seq,),
    )
    row = cursor.fetchone()
    if row:
        return dict(row)

    cursor.execute(
        """
        SELECT item_seq, item_name, company_name
        FROM svc_dur_item_caution_v3_cache
        WHERE item_seq = %s
        LIMIT 1
        """,
        (item_seq,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def build_payload(user_id: str, target_item_seq: str) -> dict[str, Any]:
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            target_basic = fetch_target_basic(cursor, target_item_seq)
            if not target_basic:
                raise ValueError(f"대상 품목을 찾을 수 없습니다: {target_item_seq}")

            active_inventory = fetch_active_medications(cursor, user_id)
            comparison_inventory = [
                row for row in active_inventory
                if str(row["item_seq"]) != target_item_seq
            ]

            all_item_seqs = [target_item_seq] + [
                str(row["item_seq"]) for row in comparison_inventory
            ]
            # preserve order while removing duplicates
            all_item_seqs = list(dict.fromkeys(all_item_seqs))

            direct_rows_all = fetch_direct_rows(cursor, all_item_seqs)
            direct_rows_target = [
                row for row in direct_rows_all
                if str(row.get("item_seq")) == target_item_seq
            ]
            concomitant_rows = fetch_concomitant_rows(cursor, all_item_seqs)
            therapeutic_rows = fetch_therapeutic_rows(cursor, all_item_seqs)
            permission_rows = fetch_permission_rows(cursor, [target_item_seq])

        synthetic_medications: list[dict[str, Any]] = [
            {
                "personal_medication_id": None,
                "user_id": user_id,
                "item_seq": target_item_seq,
                "item_name": target_basic.get("item_name"),
                "company_name": target_basic.get("company_name"),
                "medication_status": "TARGET",
                "identification_source": "REPORT_TARGET",
                "confirmed_by_user": 1,
                "started_at": None,
                "ended_at": None,
                "dose_text": None,
                "frequency_text": None,
                "administration_time_text": None,
            }
        ] + comparison_inventory

        medication_by_item, ingredient_codes_by_item, ingredient_names_by_item = (
            build_indices(synthetic_medications, direct_rows_all)
        )

        all_concomitant = build_active_concomitant(
            concomitant_rows,
            medication_by_item,
            ingredient_codes_by_item,
        )
        target_concomitant = [
            warning for warning in all_concomitant
            if target_item_seq in {
                str(warning.get("drug_a", {}).get("item_seq")),
                str(warning.get("drug_b", {}).get("item_seq")),
            }
        ]

        interaction_warnings: list[dict[str, Any]] = []
        for warning in target_concomitant:
            drug_a = warning.get("drug_a", {})
            drug_b = warning.get("drug_b", {})
            other = drug_b if str(drug_a.get("item_seq")) == target_item_seq else drug_a
            interaction_warnings.append(
                {
                    "status": "ACTIVE",
                    "type": "CONCOMITANT_CONTRAINDICATION",
                    "other_drug": other,
                    "rule_ids": warning.get("rule_ids", []),
                    "risks": warning.get("risks", []),
                    "remarks": warning.get("remarks", []),
                    "evidence_row_count": warning.get("evidence_row_count", 0),
                }
            )

        all_therapeutic = build_active_therapeutic(
            therapeutic_rows,
            medication_by_item,
            ingredient_codes_by_item,
        )
        target_therapeutic = []
        for warning in all_therapeutic:
            drug_a = warning.get("drug_a", {})
            drug_b = warning.get("drug_b", {})
            if target_item_seq not in {
                str(drug_a.get("item_seq")),
                str(drug_b.get("item_seq")),
            }:
                continue
            other = drug_b if str(drug_a.get("item_seq")) == target_item_seq else drug_a
            target_therapeutic.append(
                {
                    "status": "ACTIVE",
                    "type": "THERAPEUTIC_DUPLICATION",
                    "other_drug": other,
                    "effect_code": warning.get("effect_code"),
                    "series_name": warning.get("series_name"),
                    "rule_ids": warning.get("rule_ids", []),
                    "risks": warning.get("risks", []),
                    "remarks": warning.get("remarks", []),
                }
            )

        target_med = medication_by_item[target_item_seq]
        target_drug = {
            "item_seq": target_item_seq,
            "item_name": target_med.get("item_name"),
            "company_name": target_med.get("company_name"),
            "product_form_raw": target_med.get("product_form_raw"),
            "product_form_family": target_med.get("product_form_family"),
            "ingredient_codes": sorted(ingredient_codes_by_item.get(target_item_seq, set())),
            "ingredient_names": sorted(ingredient_names_by_item.get(target_item_seq, set())),
        }

        inventory_summary = [
            {
                "item_seq": str(row["item_seq"]),
                "item_name": row.get("item_name"),
                "company_name": row.get("company_name"),
                "medication_status": row.get("medication_status"),
            }
            for row in comparison_inventory
        ]

        permission_list = build_permission_payload([target_item_seq], permission_rows)
        permission_info = permission_list[0] if permission_list else {
            "item_seq": target_item_seq,
            "source_status": "NOT_FOUND",
            "permission_text": {},
        }

        target_warning_list = build_drug_specific_warnings(
            [target_item_seq], direct_rows_target
        )
        target_warnings = (
            target_warning_list[0].get("direct_warnings", [])
            if target_warning_list else []
        )

        return {
            "schema": "single-drug-report-with-inventory-check-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id,
            "analysis_status": "COMPLETED",
            "target_drug": target_drug,
            "inventory_interaction_check": {
                "checked_active_medication_count": len(comparison_inventory),
                "checked_medications": inventory_summary,
                "active_concomitant_warnings": interaction_warnings,
                "active_therapeutic_duplications": target_therapeutic,
                "interpretation": (
                    "대상 의약품과 사용자의 현재 ACTIVE·확인완료 보유약 사이에서 "
                    "실제로 매칭된 관계만 포함한다."
                ),
            },
            "target_drug_specific_warnings": target_warnings,
            "target_permission_information": permission_info,
            "report_policy": {
                "primary_subject": "target_drug",
                "inventory_use": "interaction_check_only",
                "do_not_expand_other_drugs": True,
                "do_not_use_ingredient_pairs": True,
            },
        }
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="선택한 약 1개의 복약지도서용 JSON과 보유약 병용검사 결과를 생성합니다."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--item-seq", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    try:
        payload = build_payload(args.user_id, str(args.item_seq))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        interaction = payload["inventory_interaction_check"]
        print(f"[완료] {output_path}")
        print("대상 약:", payload["target_drug"].get("item_name"))
        print("비교한 보유약:", interaction["checked_active_medication_count"])
        print("활성 병용금기:", len(interaction["active_concomitant_warnings"]))
        print("활성 효능군 중복:", len(interaction["active_therapeutic_duplications"]))
        return 0
    except pymysql.MySQLError as exc:
        print(f"[DB 오류] {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[실행 오류] {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
