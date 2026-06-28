#!/usr/bin/env python3

import argparse
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL


DEFAULT_SOURCE = "llm_json_samples/json_test_v3.json"
DEFAULT_OUTPUT = "llm_json_samples/llm_delivery_final.json"
DEFAULT_PERMISSION_TABLE = "drug_permission_info"


def clean(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = value.strip()

        # 원문 바깥에 불필요하게 남은 따옴표만 제거
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1].strip()

        return value or None

    return value


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def unique_nonempty(values: list[Any]) -> list[Any]:
    result = []
    seen = set()

    for value in values:
        value = clean(value)

        if value is None:
            continue

        key = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def make_engine():
    required = [
        "DB_HOST",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME",
    ]

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "DB 환경변수가 없습니다: "
            + ", ".join(missing)
        )

    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.environ["DB_NAME"],
        query={"charset": "utf8mb4"},
    )

    return create_engine(
        url,
        pool_pre_ping=True,
    )


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    lower_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        matched = lower_map.get(candidate.lower())

        if matched:
            return matched

    return None


def normalize_permission_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    normalized = {}

    for key, value in record.items():
        value = json_safe(value)

        if value is None:
            continue

        if isinstance(value, str):
            value = clean(value)

        if value in (None, ""):
            continue

        normalized[str(key)] = value

    return normalized


def extract_text_sections(
    record: dict[str, Any],
) -> dict[str, str]:
    """
    허가정보 테이블의 컬럼명이 데이터 공급처마다 달라도
    긴 텍스트 필드는 원문 섹션으로 함께 전달한다.
    """
    excluded_tokens = {
        "item_seq",
        "item_name",
        "company",
        "manufacturer",
        "product_code",
        "bar_code",
        "barcode",
        "permit_date",
        "update_date",
        "created_at",
        "updated_at",
        "loaded_at",
    }

    sections = {}

    for key, value in record.items():
        if not isinstance(value, str):
            continue

        value = clean(value)

        if not value:
            continue

        key_lower = key.lower()

        if key_lower in excluded_tokens:
            continue

        # 짧은 분류값보다 실제 허가 원문 중심으로 추출
        if len(value) < 20:
            continue

        sections[key] = value

    return sections



def load_permission_information(
    engine,
    table_name: str,
    item_seq: str,
    existing: Any,
) -> dict[str, Any]:
    """
    LLM 전달용 허가정보.

    포함:
    - 효능·효과 평문
    - 용법·용량 평문
    - 사용상의 주의사항 평문
    - 저장방법
    - 유효기간
    - 포장단위

    제외:
    - XML 원문
    - raw_record
    - raw_payload
    """

    def build_permission_payload(
        record: dict[str, Any],
        match_method: str,
    ) -> dict[str, Any]:
        permission_text = {
            "efficacy": clean(
                record.get("efficacy_text")
            ),
            "dosage": clean(
                record.get("dosage_text")
            ),
            "precautions": clean(
                record.get("precautions_text")
            ),
            "storage_method": clean(
                record.get("storage_method")
            ),
            "valid_term": clean(
                record.get("valid_term")
            ),
            "pack_unit": clean(
                record.get("pack_unit")
            ),
        }

        permission_text = {
            key: value
            for key, value in permission_text.items()
            if value is not None
        }

        product_information = {
            "item_name": clean(
                record.get("item_name")
            ),
            "company_name": clean(
                record.get("company_name")
            ),
            "permit_date": clean(
                record.get("permit_date")
            ),
            "etc_otc_code": clean(
                record.get("etc_otc_code")
            ),
            "chart_text": clean(
                record.get("chart_text")
            ),
            "main_item_ingredient": clean(
                record.get("main_item_ingredient")
            ),
            "material_name": clean(
                record.get("material_name")
            ),
            "total_content": clean(
                record.get("total_content")
            ),
            "edi_code": clean(
                record.get("edi_code")
            ),
            "atc_code": clean(
                record.get("atc_code")
            ),
        }

        product_information = {
            key: value
            for key, value in product_information.items()
            if value is not None
        }

        return {
            "source_status": "AVAILABLE",
            "source_table": table_name,
            "match_method": match_method,
            "product_information": product_information,
            "permission_text": permission_text,
        }

    if isinstance(existing, dict) and existing:
        existing_record = normalize_permission_record(
            existing
        )

        return build_permission_payload(
            existing_record,
            "EXISTING_PAYLOAD",
        )

    inspector = inspect(engine)

    if not inspector.has_table(table_name):
        return {
            "source_status": "TABLE_NOT_FOUND",
            "source_table": table_name,
            "match_method": None,
            "permission_text": {},
        }

    columns = [
        column["name"]
        for column in inspector.get_columns(table_name)
    ]

    item_seq_column = find_column(
        columns,
        [
            "item_seq",
            "ITEM_SEQ",
            "품목일련번호",
            "품목기준코드",
            "item_code",
        ],
    )

    if not item_seq_column:
        return {
            "source_status": "KEY_COLUMN_NOT_FOUND",
            "source_table": table_name,
            "match_method": None,
            "permission_text": {},
        }

    permission_df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{table_name}`
            WHERE CAST(`{item_seq_column}` AS CHAR)
                = :item_seq
            LIMIT 1
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    if permission_df.empty:
        return {
            "source_status": "NOT_FOUND",
            "source_table": table_name,
            "match_method": "ITEM_SEQ_EXACT",
            "permission_text": {},
        }

    record = normalize_permission_record(
        permission_df.iloc[0].to_dict()
    )

    return build_permission_payload(
        record,
        "ITEM_SEQ_EXACT",
    )


def compact_direct_warning(
    warning: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "type": clean(warning.get("dur_type")),
        "content": clean(
            warning.get("prohibition_content")
        ),
        "remark": clean(warning.get("remark")),
        "age_standard": clean(
            warning.get("age_base_raw")
        ),
        "pregnancy_grade": clean(
            warning.get("pregnancy_grade")
        ),
        "maximum_quantity": clean(
            warning.get("max_quantity_raw")
        ),
        "maximum_duration": clean(
            warning.get("max_duration_raw")
        ),
        "applicability": clean(
            warning.get("applicability")
        ),
        "ingredient_match": clean(
            warning.get("ingredient_match")
        ),
        "warning_level": clean(
            warning.get("warning_level")
        ),
    }

    return {
        key: value
        for key, value in result.items()
        if value is not None
    }


def compact_dose_reference(
    reference: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "ingredient": clean(
            reference.get("ingredient")
        ),
        "maximum_quantity": clean(
            reference.get("max_quantity")
        ),
        "content": clean(
            reference.get("content")
        ),
        "applicability": clean(
            reference.get("applicability")
        ),
        "ingredient_match": clean(
            reference.get("ingredient_match")
        ),
    }

    return {
        key: value
        for key, value in result.items()
        if value is not None
    }


def compact_split_caution(
    caution: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "type": clean(caution.get("dur_type")),
        "content": clean(
            caution.get("prohibition_content")
        ),
        "remark": clean(caution.get("remark")),
        "applicability": clean(
            caution.get("applicability")
        ),
    }

    return {
        key: value
        for key, value in result.items()
        if value is not None
    }


def compact_concomitant_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "current_ingredient_code": clean(
            candidate.get("current_ingredient_code")
        ),
        "current_ingredient_name": clean(
            candidate.get("current_ingredient_name")
        ),
        "contraindicated_ingredient_code": clean(
            candidate.get(
                "contraindicated_ingredient_code"
            )
        ),
        "contraindicated_ingredient_name": clean(
            candidate.get(
                "contraindicated_ingredient_name"
            )
        ),
        "risks": unique_nonempty(
            candidate.get(
                "prohibition_contents",
                [],
            )
        ),
        "remarks": unique_nonempty(
            candidate.get("remarks", [])
        ),
        "matched_product_count": candidate.get(
            "matched_product_count"
        ),
        "example_products": unique_nonempty(
            candidate.get("example_products", [])
        )[:3],
        "example_item_seqs": unique_nonempty(
            candidate.get("example_item_seqs", [])
        )[:3],
        "status": "CANDIDATE",
    }

    return {
        key: value
        for key, value in result.items()
        if value not in (None, [], {})
    }


def compact_therapeutic_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result = {}

    for key, value in candidate.items():
        value = json_safe(value)

        if isinstance(value, str):
            value = clean(value)

        if value in (None, "", [], {}):
            continue

        result[key] = value

    result["status"] = "CANDIDATE"

    return result


def build_delivery_item(
    payload: dict[str, Any],
    engine,
    permission_table: str,
) -> dict[str, Any]:
    item = payload.get("item") or payload.get("drug") or {}
    dur = payload.get("dur") or {}
    multi = payload.get("multi_drug_rules") or {}

    ingredient_names = unique_nonempty(
        item.get("ingredient_names", [])
    )

    drug = {
        "item_seq": clean(item.get("item_seq")),
        "item_name": clean(item.get("item_name")),
        "company_name": clean(
            item.get("company_name")
        ),
        "product_form": clean(
            item.get("product_form")
        ),
        "ingredient_codes": unique_nonempty(
            item.get("ingredient_codes", [])
        ),
        "ingredient_names": ingredient_names,
    }

    direct_source = (
        payload.get("direct_warnings")
        or dur.get("direct_warnings")
        or []
    )

    dose_source = (
        payload.get("dose_references")
        or dur.get("ingredient_dose_references")
        or []
    )

    split_source = (
        payload.get("split_cautions")
        or dur.get(
            "split_extended_release_cautions"
        )
        or []
    )

    concomitant_source = (
        payload.get("concomitant_candidates")
        or multi.get("concomitant_candidates")
        or multi.get("concomitant_warnings")
        or []
    )

    therapeutic_source = (
        payload.get(
            "therapeutic_duplication_candidates"
        )
        or multi.get(
            "therapeutic_duplication_candidates"
        )
        or multi.get(
            "therapeutic_duplication_rules"
        )
        or []
    )

    permission = load_permission_information(
        engine=engine,
        table_name=permission_table,
        item_seq=drug["item_seq"],
        existing=payload.get(
            "permission_information"
        ),
    )

    return {
        "drug": {
            key: value
            for key, value in drug.items()
            if value not in (None, [], {})
        },
        "dur_information": {
            "direct_warnings": [
                compact_direct_warning(row)
                for row in direct_source
            ],
            "dose_references": [
                compact_dose_reference(row)
                for row in dose_source
            ],
            "split_cautions": [
                compact_split_caution(row)
                for row in split_source
            ],
        },
        "multi_drug_information": {
            "concomitant_candidates": [
                compact_concomitant_candidate(row)
                for row in concomitant_source
            ],
            "therapeutic_duplication_candidates": [
                compact_therapeutic_candidate(row)
                for row in therapeutic_source
            ],
            "interpretation": (
                "후보 정보는 현재 복용 중인 다른 약과 "
                "성분이 실제로 일치할 때만 활성 경고로 해석한다."
            ),
        },
        "permission_information": permission,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "검증용 DUR JSON을 최종 LLM 전달용 JSON으로 "
            "축약하고 허가정보 원문을 결합합니다."
        )
    )

    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--items",
        default="",
        help="쉼표로 구분한 품목번호. 비우면 전체 품목",
    )

    parser.add_argument(
        "--permission-table",
        default=DEFAULT_PERMISSION_TABLE,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    source_data = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    source_items = source_data.get("items", [])

    requested_items = {
        value.strip()
        for value in args.items.split(",")
        if value.strip()
    }

    if requested_items:
        source_items = [
            payload
            for payload in source_items
            if str(
                (
                    payload.get("item")
                    or payload.get("drug")
                    or {}
                ).get("item_seq")
            ) in requested_items
        ]

    engine = make_engine()

    delivery_items = [
        build_delivery_item(
            payload=payload,
            engine=engine,
            permission_table=args.permission_table,
        )
        for payload in source_items
    ]

    result = {
        "schema_version": "llm-delivery-v2",
        "payload_purpose": (
            "복약지도 생성을 위한 구조화된 DUR 정보와 "
            "품목 허가정보 원문의 결합"
        ),
        "interpretation_policy": {
            "direct_warnings": (
                "현재 품목에 직접 적용되는 DUR 경고"
            ),
            "dose_references": (
                "성분 단위 참고 기준이며 현재 품목의 "
                "직접 복용량으로 단정하지 않음"
            ),
            "concomitant_candidates": (
                "다른 복용약과 상대 성분이 일치할 때만 "
                "실제 병용금기 경고로 활성화"
            ),
            "therapeutic_duplication_candidates": (
                "다른 복용약과 효능군이 중복될 때만 "
                "실제 중복 경고로 활성화"
            ),
            "permission_information": (
                "허가정보 원문이며 요약 과정에서 의미를 "
                "임의로 추가하거나 변경하지 않음"
            ),
        },
        "item_count": len(delivery_items),
        "items": delivery_items,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("최종 LLM 전달용 JSON 생성 완료:")
    print(output_path.resolve())
    print("품목 수:", len(delivery_items))

    for payload in delivery_items:
        drug = payload["drug"]
        permission = payload[
            "permission_information"
        ]

        print(
            drug.get("item_seq"),
            drug.get("item_name"),
            "| 허가정보:",
            permission.get("source_status"),
            "| 허가 원문 필드:",
            len(permission.get("text_sections", {})),
        )


if __name__ == "__main__":
    main()
