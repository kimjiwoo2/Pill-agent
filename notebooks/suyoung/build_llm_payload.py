import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, inspect, text


# =========================================================
# 테이블 설정
# =========================================================

DUR_CACHE = "svc_dur_item_caution_v3_cache"
THERAPEUTIC_TABLE = "svc_dur_therapeutic_duplication_v3"
CONCOMITANT_TABLE = "svc_dur_concomitant_effective"
PERMISSION_TABLE = "drug_permission_info"

DEFAULT_OUTPUT = Path(
    "llm_json_samples/all_samples_v3_final.json"
)

DEFAULT_SAMPLE_ITEMS = [
    "198701676",  # 디낙스정
    "201505094",  # 트라콤세미정
    "197100097",  # 환인벤즈트로핀정
    "202301430",  # 3성분 당뇨병 복합제
    "198900901",  # 캡슐 예시
]


# =========================================================
# DB 연결
# =========================================================

DB_URL = (
    f"mysql+pymysql://{os.environ['DB_USER']}:"
    f"{quote_plus(os.environ['DB_PASSWORD'])}@"
    f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/"
    f"{os.environ['DB_NAME']}?charset=utf8mb4"
)

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
)


# =========================================================
# 공통 유틸
# =========================================================

def table_exists(table_name: str) -> bool:
    return inspect(engine).has_table(table_name)


def table_columns(table_name: str) -> list[str]:
    if not table_exists(table_name):
        return []

    return [
        column["name"]
        for column in inspect(engine).get_columns(table_name)
    ]


def first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def clean(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return value or None


def normalize_item_seq(value):
    value = clean(value)

    if not value:
        return None

    if value.endswith(".0"):
        value = value[:-2]

    return value


def records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    safe = df.astype(object).where(
        pd.notna(df),
        None,
    )

    return safe.to_dict(orient="records")


def remove_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    LLM에 불필요한 내부 검증용 컬럼 제거.
    """
    if df.empty:
        return df

    internal_columns = [
        "clinical_rule_key",
        "semantic_rule_key",
        "selected_v3",
        "accepted_v2",
        "is_active",
        "source_version",
    ]

    return df.drop(
        columns=[
            column
            for column in internal_columns
            if column in df.columns
        ],
        errors="ignore",
    )


# =========================================================
# DUR 조회
# =========================================================

def load_item_dur(item_seq: str) -> pd.DataFrame:
    if not table_exists(DUR_CACHE):
        raise RuntimeError(
            f"{DUR_CACHE} 테이블이 없습니다."
        )

    return pd.read_sql(
        text(f"""
            SELECT *
            FROM `{DUR_CACHE}`
            WHERE CAST(item_seq AS CHAR) = :item_seq
            ORDER BY
                CASE warning_level
                    WHEN 'DIRECT' THEN 1
                    WHEN 'REFERENCE' THEN 2
                    ELSE 3
                END,
                dur_type,
                rule_id
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )


def build_direct_warnings(cache_df: pd.DataFrame) -> list[dict]:
    if cache_df.empty:
        return []

    direct = cache_df[
        (cache_df["warning_level"] == "DIRECT")
        & (
            cache_df["dur_type"]
            != "SPLIT_EXTENDED_RELEASE_CAUTION"
        )
    ].copy()

    direct = remove_internal_columns(direct)

    return records(direct)



def build_dose_references(cache_df: pd.DataFrame) -> list[dict]:
    if cache_df.empty:
        return []

    reference = cache_df[
        cache_df["warning_level"] == "REFERENCE"
    ].copy()

    result = []

    for _, row in reference.iterrows():
        max_quantity = clean(
            row.get("max_quantity_raw")
        )

        ingredient = None

        if max_quantity:
            match = re.match(
                r"^\s*(.+?)\s+[\d,]+(?:\.\d+)?\s*"
                r"(?:mg|g|mcg|밀리그램|그램|마이크로그램)",
                max_quantity,
                flags=re.IGNORECASE,
            )

            if match:
                ingredient = match.group(1).strip()

        result.append({
            "ingredient": ingredient,
            "max_quantity": max_quantity,
            "content": (
                f"{ingredient} 성분의 최대용량 참고 기준"
                if ingredient
                else clean(row.get("prohibition_content"))
            ),
            "ingredient_match": clean(
                row.get("ingredient_match")
            ),
            "applicability": clean(
                row.get("applicability")
            ),
        })

    return result


def build_split_cautions(cache_df: pd.DataFrame) -> list[dict]:
    if cache_df.empty:
        return []

    split = cache_df[
        cache_df["dur_type"]
        == "SPLIT_EXTENDED_RELEASE_CAUTION"
    ].copy()

    split = remove_internal_columns(split)

    return records(split)


# =========================================================
# 효능군중복 조회
# =========================================================

def load_therapeutic_duplication(item_seq: str) -> list[dict]:
    if not table_exists(THERAPEUTIC_TABLE):
        return []

    columns = table_columns(THERAPEUTIC_TABLE)

    item_col = first_existing(
        columns,
        ["item_seq", "ITEM_SEQ"],
    )

    if not item_col:
        return []

    df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{THERAPEUTIC_TABLE}`
            WHERE CAST(`{item_col}` AS CHAR) = :item_seq
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    df = remove_internal_columns(df)

    return records(df)


# =========================================================
# 병용금기 조회
# =========================================================







def load_concomitant(item_seq: str) -> list[dict]:
    """
    cur_dur_product_map을 기준으로 병용금기 방향을 결정한다.

    ingredient_code:
        현재 조회 품목의 성분코드

    mixture_ingredient_code:
        병용 상대 성분코드
    """
    map_table = "cur_dur_product_map"

    if not table_exists(map_table):
        return []

    map_df = pd.read_sql(
        text(f"""
            SELECT
                ingredient_code,
                mixture_ingredient_code,
                item_name,
                mixture_item_seq,
                mixture_item_name,
                canonical_pair_key
            FROM `{map_table}`
            WHERE CAST(item_seq AS CHAR) = :item_seq
              AND dur_type = 'CONCOMITANT_CONTRAINDICATION'
              AND ingredient_code IS NOT NULL
              AND mixture_ingredient_code IS NOT NULL
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    if map_df.empty:
        return []

    effective_df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{CONCOMITANT_TABLE}`
            WHERE CAST(item_seq AS CHAR) = :item_seq
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    def extract_parenthetical_names(value):
        value = clean(value)

        if not value:
            return []

        matches = re.findall(r"\(([^()]*)\)", value)

        result = []

        for match in matches:
            for name in re.split(r"[,/+]", match):
                name = name.strip()

                if not name:
                    continue

                if any(token in name for token in [
                    "수출용",
                    "수출명",
                    "분류번호",
                ]):
                    continue

                result.append(name)

        return result

    def normalize_ingredient_name(name):
        name = clean(name)

        if not name:
            return None

        replacements = [
            "염산염수화물",
            "메실산염수화물",
            "나트륨수화물",
            "염산염",
            "메실산염",
            "타르타르산염",
            "나트륨염",
            "칼륨염",
            "수화물",
        ]

        normalized = name

        for suffix in replacements:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break

        return normalized.strip() or name

    def resolve_current_name(code):
        """
        같은 성분코드를 가진 품목명들의 괄호 성분명을 집계한다.
        복합제의 다른 성분은 품목마다 달라지지만,
        현재 코드의 실제 성분은 반복적으로 나타난다.
        """
        sample_df = pd.read_sql(
            text(f"""
                SELECT item_name
                FROM `{map_table}`
                WHERE ingredient_code = :code
                  AND item_name IS NOT NULL
                LIMIT 1000
            """),
            engine,
            params={"code": str(code)},
        )

        counts = {}

        for value in sample_df["item_name"].tolist():
            for name in extract_parenthetical_names(value):
                normalized = normalize_ingredient_name(name)

                if normalized:
                    counts[normalized] = (
                        counts.get(normalized, 0) + 1
                    )

        if not counts:
            return None

        return max(
            counts,
            key=lambda name: counts[name],
        )

    def resolve_mixture_name(group):
        counts = {}

        for value in group["mixture_item_name"].tolist():
            for name in extract_parenthetical_names(value):
                normalized = normalize_ingredient_name(name)

                if normalized:
                    counts[normalized] = (
                        counts.get(normalized, 0) + 1
                    )

        if not counts:
            return None

        return max(
            counts,
            key=lambda name: counts[name],
        )

    result = []

    for (
        current_code,
        other_code,
    ), group in map_df.groupby(
        [
            "ingredient_code",
            "mixture_ingredient_code",
        ],
        dropna=False,
        sort=False,
    ):
        pair_keys = {
            clean(value)
            for value in group["canonical_pair_key"].tolist()
            if clean(value)
        }

        matched_effective = effective_df

        if (
            pair_keys
            and "canonical_pair_key" in effective_df.columns
        ):
            matched_effective = effective_df[
                effective_df["canonical_pair_key"].isin(
                    pair_keys
                )
            ]

        contents = []

        if "prohibition_content" in matched_effective.columns:
            contents = list(dict.fromkeys(
                str(value).strip()
                for value in matched_effective[
                    "prohibition_content"
                ].dropna().tolist()
                if str(value).strip()
            ))

        remarks = []

        if "remark" in matched_effective.columns:
            remarks = list(dict.fromkeys(
                str(value).strip()
                for value in matched_effective[
                    "remark"
                ].dropna().tolist()
                if str(value).strip()
            ))

        product_names = list(dict.fromkeys(
            str(value).strip()
            for value in group[
                "mixture_item_name"
            ].dropna().tolist()
            if str(value).strip()
        ))

        product_seqs = list(dict.fromkeys(
            normalize_item_seq(value)
            for value in group[
                "mixture_item_seq"
            ].dropna().tolist()
            if normalize_item_seq(value)
        ))

        result.append({
            "current_ingredient_code": clean(current_code),
            "current_ingredient_name":
                resolve_current_name(current_code),
            "contraindicated_ingredient_code":
                clean(other_code),
            "contraindicated_ingredient_name":
                resolve_mixture_name(group),
            "prohibition_contents": contents,
            "remarks": remarks,
            "matched_product_count": len(product_seqs),
            "example_products": product_names[:3],
            "example_item_seqs": product_seqs[:3],
        })

    return result

# =========================================================
# 허가정보 조회
# =========================================================

def load_permission(item_seq: str):
    if not table_exists(PERMISSION_TABLE):
        return None

    columns = table_columns(PERMISSION_TABLE)

    item_col = first_existing(
        columns,
        ["item_seq", "ITEM_SEQ"],
    )

    if not item_col:
        return None

    df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{PERMISSION_TABLE}`
            WHERE CAST(`{item_col}` AS CHAR) = :item_seq
            LIMIT 1
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    if df.empty:
        return None

    return records(df)[0]


# =========================================================
# 품목 기본정보
# =========================================================


def extract_item_info(cache_df: pd.DataFrame, item_seq: str):
    def parse_json_list(value):
        value = clean(value)

        if not value:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except (json.JSONDecodeError, TypeError):
            pass

        return [value]

    def parse_names_from_item_name(item_name):
        item_name = clean(item_name)

        if not item_name:
            return []

        matches = re.findall(r"\(([^()]*)\)", item_name)

        if not matches:
            return []

        candidate = matches[-1]

        names = [
            value.strip()
            for value in re.split(r"[,/+]", candidate)
            if value.strip()
        ]

        result = []

        for name in names:
            if name not in result:
                result.append(name)

        return result

    if cache_df.empty:
        return {
            "item_seq": str(item_seq),
            "item_name": None,
            "company_name": None,
            "product_form": None,
            "ingredient_codes": [],
            "ingredient_names": [],
        }

    row = cache_df.iloc[0]

    item_name = clean(row.get("item_name"))

    ingredient_codes = parse_json_list(
        row.get("product_ingredient_codes")
    )

    ingredient_names = parse_json_list(
        row.get("product_ingredient_names")
    )

    if not ingredient_names:
        ingredient_names = parse_names_from_item_name(
            item_name
        )

    ingredients = []

    max_length = max(
        len(ingredient_codes),
        len(ingredient_names),
    )

    for index in range(max_length):
        ingredients.append({
            "code": (
                ingredient_codes[index]
                if index < len(ingredient_codes)
                else None
            ),
            "name": (
                ingredient_names[index]
                if index < len(ingredient_names)
                else None
            ),
        })

    return {
        "item_seq": str(item_seq),
        "item_name": item_name,
        "company_name": clean(
            row.get("company_name")
        ),
        "product_form": clean(
            row.get("product_form_raw")
        ),
        "ingredient_codes": ingredient_codes,
        "ingredient_names": ingredient_names,
    }

# =========================================================
# 최종 payload
# =========================================================


def build_payload(item_seq: str) -> dict:
    item_seq = normalize_item_seq(item_seq)

    cache_df = load_item_dur(item_seq)

    item_info = extract_item_info(
        cache_df,
        item_seq,
    )

    direct_warnings = build_direct_warnings(cache_df)
    dose_references = build_dose_references(cache_df)
    split_cautions = build_split_cautions(cache_df)

    therapeutic_rules = load_therapeutic_duplication(
        item_seq
    )
    concomitant_warnings = load_concomitant(
        item_seq
    )
    permission_information = load_permission(
        item_seq
    )

    return {
        "item": item_info,

        "dur": {
            "direct_warnings": direct_warnings,
            "ingredient_dose_references": dose_references,
            "split_extended_release_cautions": split_cautions,
        },

        "multi_drug_rules": {
            "therapeutic_duplication_candidates":
                therapeutic_rules,
            "concomitant_candidates":
                concomitant_warnings,
        },

        "permission_information":
            permission_information,

        "summary": {
            "direct_warning_count": len(direct_warnings),
            "dose_reference_count": len(dose_references),
            "split_caution_count": len(split_cautions),
            "therapeutic_duplication_count": len(
                therapeutic_rules
            ),
            "concomitant_warning_count": len(
                concomitant_warnings
            ),
            "has_permission_information": (
                permission_information is not None
            ),
        },
    }

# =========================================================
# 대상 품목 선택
# =========================================================

def load_all_item_seqs() -> list[str]:
    df = pd.read_sql(
        text(f"""
            SELECT DISTINCT CAST(item_seq AS CHAR) AS item_seq
            FROM `{DUR_CACHE}`
            WHERE item_seq IS NOT NULL
            ORDER BY item_seq
        """),
        engine,
    )

    return [
        normalize_item_seq(value)
        for value in df["item_seq"].tolist()
        if normalize_item_seq(value)
    ]


def resolve_item_seqs(args) -> list[str]:
    if args.all:
        return load_all_item_seqs()

    if args.items:
        return [
            normalize_item_seq(item)
            for item in args.items.split(",")
            if normalize_item_seq(item)
        ]

    env_items = os.environ.get("SAMPLE_ITEM_SEQS")

    if env_items:
        return [
            normalize_item_seq(item)
            for item in env_items.split(",")
            if normalize_item_seq(item)
        ]

    return DEFAULT_SAMPLE_ITEMS


# =========================================================
# 실행
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pilliot DUR v3 LLM JSON 생성기"
    )

    parser.add_argument(
        "--items",
        type=str,
        help="쉼표로 구분한 ITEM_SEQ 목록",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="최종 캐시의 모든 품목 JSON 생성",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="출력 JSON 파일 경로",
    )

    args = parser.parse_args()

    item_seqs = resolve_item_seqs(args)

    if not item_seqs:
        raise RuntimeError(
            "JSON을 생성할 ITEM_SEQ가 없습니다."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("DUR 캐시:", DUR_CACHE)
    print("대상 품목 수:", len(item_seqs))

    payloads = []

    for index, item_seq in enumerate(item_seqs, start=1):
        payloads.append(
            build_payload(item_seq)
        )

        if index % 100 == 0:
            print(
                f"처리 중: {index}/{len(item_seqs)}"
            )

    result = {
        "schema_version": "dur-v3",
        "source_tables": {
            "dur_cache": DUR_CACHE,
            "therapeutic_duplication":
                THERAPEUTIC_TABLE,
            "concomitant":
                CONCOMITANT_TABLE,
            "permission":
                PERMISSION_TABLE,
        },
        "item_count": len(payloads),
        "items": payloads,
    }

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("JSON 생성 완료:")
    print(output_path.resolve())

    print("품목 수:", len(payloads))

    direct_count = sum(
        item["summary"]["direct_warning_count"]
        for item in payloads
    )

    reference_count = sum(
        item["summary"]["dose_reference_count"]
        for item in payloads
    )

    print("DIRECT 경고 수:", direct_count)
    print("REFERENCE 수:", reference_count)


if __name__ == "__main__":
    main()
