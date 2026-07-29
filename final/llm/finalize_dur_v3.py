import json
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import (
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)


# =========================================================
# 설정
# =========================================================

DIRECT_VIEW = "svc_dur_item_caution_v3_direct"
REFERENCE_VIEW = "svc_dur_item_caution_v3_reference"
REVIEW_VIEW = "svc_dur_item_caution_v3_review"
REJECTED_VIEW = "svc_dur_item_caution_v3_rejected"

LEGACY_CACHE = "svc_dur_item_caution_v2_cache"

FINAL_CACHE = "svc_dur_item_caution_v3_cache"
THERAPEUTIC_CACHE = "svc_dur_therapeutic_duplication_v3"

OUTPUT_DIR = Path("llm_json_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path("dur_matching_v3_final_report")
REPORT_DIR.mkdir(exist_ok=True)

DB_URL = (
    f"mysql+pymysql://{os.environ['DB_USER']}:"
    f"{quote_plus(os.environ['DB_PASSWORD'])}@"
    f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/"
    f"{os.environ['DB_NAME']}?charset=utf8mb4"
)

engine = create_engine(DB_URL, pool_pre_ping=True)


# =========================================================
# 공통 함수
# =========================================================

def table_exists(table_name: str) -> bool:
    return inspect(engine).has_table(table_name)


def table_columns(table_name: str) -> list[str]:
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

    # pandas가 숫자로 읽어 198701676.0이 되는 경우 보정
    if value.endswith(".0"):
        value = value[:-2]

    return value


def read_table(table_name: str) -> pd.DataFrame:
    if not table_exists(table_name):
        return pd.DataFrame()

    return pd.read_sql(
        f"SELECT * FROM `{table_name}`",
        engine,
    )


def ensure_column(df, column_name, default=None):
    if column_name not in df.columns:
        df[column_name] = default
    return df


def standardize_v3_rows(df: pd.DataFrame, warning_level: str):
    if df.empty:
        return pd.DataFrame()

    result = df.copy()

    required_columns = [
        "item_seq",
        "item_name",
        "company_name",
        "product_form_raw",
        "product_form_family",
        "dur_type",
        "rule_id",
        "primary_ingredient_code",
        "product_ingredient_codes",
        "product_ingredient_names",
        "rule_ingredient_codes",
        "rule_form_raw",
        "rule_form_families",
        "form_match",
        "ingredient_match",
        "applicability",
        "match_reason",
        "prohibition_content",
        "remark",
        "max_quantity_raw",
        "max_duration_raw",
        "age_base_raw",
        "pregnancy_grade",
        "clinical_rule_key",
    ]

    for column in required_columns:
        ensure_column(result, column)

    result["item_seq"] = result["item_seq"].map(
        normalize_item_seq
    )
    result["warning_level"] = warning_level
    result["source_version"] = "V3"
    result["is_active"] = 1

    return result[
        required_columns
        + [
            "warning_level",
            "source_version",
            "is_active",
        ]
    ]


def standardize_legacy_split(df: pd.DataFrame):
    """
    기존 캐시에서 서방정 분할주의만 가져온다.
    기존 캐시 컬럼명이 달라도 최대한 자동 탐색한다.
    """
    if df.empty:
        return pd.DataFrame()

    columns = list(df.columns)

    item_seq_col = first_existing(
        columns,
        ["item_seq", "ITEM_SEQ"],
    )
    item_name_col = first_existing(
        columns,
        ["item_name", "ITEM_NAME", "item_name_kor"],
    )
    company_col = first_existing(
        columns,
        ["company_name", "ENTP_NAME", "entp_name"],
    )
    dur_type_col = first_existing(
        columns,
        ["dur_type", "DUR_TYPE", "type"],
    )
    rule_id_col = first_existing(
        columns,
        ["rule_id", "canonical_rule_id"],
    )
    ingredient_code_col = first_existing(
        columns,
        [
            "primary_ingredient_code",
            "ingredient_code",
            "ingr_code",
            "INGR_CODE",
        ],
    )
    content_col = first_existing(
        columns,
        [
            "prohibition_content",
            "content",
            "warning_content",
            "PROHBT_CONTENT",
        ],
    )
    remark_col = first_existing(
        columns,
        ["remark", "REMARK"],
    )

    if not item_seq_col or not dur_type_col:
        raise RuntimeError(
            "기존 캐시에서 item_seq 또는 dur_type 컬럼을 "
            "찾지 못했습니다."
        )

    legacy = df[
        df[dur_type_col].astype(str)
        == "SPLIT_EXTENDED_RELEASE_CAUTION"
    ].copy()

    if legacy.empty:
        return pd.DataFrame()

    result = pd.DataFrame()

    result["item_seq"] = legacy[item_seq_col].map(
        normalize_item_seq
    )
    result["item_name"] = (
        legacy[item_name_col]
        if item_name_col
        else None
    )
    result["company_name"] = (
        legacy[company_col]
        if company_col
        else None
    )
    result["product_form_raw"] = None
    result["product_form_family"] = None
    result["dur_type"] = "SPLIT_EXTENDED_RELEASE_CAUTION"
    result["rule_id"] = (
        legacy[rule_id_col]
        if rule_id_col
        else None
    )
    result["primary_ingredient_code"] = (
        legacy[ingredient_code_col]
        if ingredient_code_col
        else None
    )
    result["product_ingredient_codes"] = None
    result["product_ingredient_names"] = None
    result["rule_ingredient_codes"] = None
    result["rule_form_raw"] = None
    result["rule_form_families"] = None
    result["form_match"] = "LEGACY"
    result["ingredient_match"] = "LEGACY"
    result["applicability"] = "DIRECT_APPLICABLE"
    result["match_reason"] = "기존 서방정 분할주의 캐시 유지"
    result["prohibition_content"] = (
        legacy[content_col]
        if content_col
        else None
    )
    result["remark"] = (
        legacy[remark_col]
        if remark_col
        else None
    )
    result["max_quantity_raw"] = None
    result["max_duration_raw"] = None
    result["age_base_raw"] = None
    result["pregnancy_grade"] = None
    result["clinical_rule_key"] = None
    result["warning_level"] = "DIRECT"
    result["source_version"] = "V2_LEGACY"
    result["is_active"] = 1

    return result


def build_therapeutic_duplication_cache(
    legacy_df: pd.DataFrame,
):
    if legacy_df.empty:
        return pd.DataFrame()

    columns = list(legacy_df.columns)

    dur_type_col = first_existing(
        columns,
        ["dur_type", "DUR_TYPE", "type"],
    )

    if not dur_type_col:
        return pd.DataFrame()

    therapeutic = legacy_df[
        legacy_df[dur_type_col].astype(str)
        == "THERAPEUTIC_DUPLICATION"
    ].copy()

    if therapeutic.empty:
        return therapeutic

    therapeutic["source_version"] = "V2_LEGACY"
    therapeutic["is_active"] = 1

    return therapeutic


def save_dataframe(
    df: pd.DataFrame,
    table_name: str,
):
    if df.empty:
        print(f"[주의] {table_name}: 저장할 데이터가 없습니다.")
        return

    dtype_map = {}

    varchar_columns = {
        "item_seq": 32,
        "item_name": 500,
        "company_name": 300,
        "product_form_family": 100,
        "dur_type": 100,
        "rule_id": 150,
        "primary_ingredient_code": 100,
        "form_match": 100,
        "ingredient_match": 100,
        "applicability": 100,
        "pregnancy_grade": 100,
        "warning_level": 50,
        "source_version": 50,
    }

    for column, length in varchar_columns.items():
        if column in df.columns:
            dtype_map[column] = String(length)

    text_columns = [
        "product_form_raw",
        "product_ingredient_codes",
        "product_ingredient_names",
        "rule_ingredient_codes",
        "rule_form_raw",
        "rule_form_families",
        "match_reason",
        "prohibition_content",
        "remark",
        "max_quantity_raw",
        "max_duration_raw",
        "age_base_raw",
        "clinical_rule_key",
    ]

    for column in text_columns:
        if column in df.columns:
            dtype_map[column] = Text()

    if "is_active" in df.columns:
        dtype_map["is_active"] = Integer()

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        dtype=dtype_map,
        chunksize=2000,
        method="multi",
    )


def create_final_indexes():
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE INDEX idx_v3_cache_item
            ON `{FINAL_CACHE}` (`item_seq`)
        """))

        conn.execute(text(f"""
            CREATE INDEX idx_v3_cache_item_type
            ON `{FINAL_CACHE}` (`item_seq`, `dur_type`)
        """))

        conn.execute(text(f"""
            CREATE INDEX idx_v3_cache_level
            ON `{FINAL_CACHE}` (`warning_level`, `dur_type`)
        """))


# =========================================================
# JSON 생성
# =========================================================

def dataframe_records(df: pd.DataFrame):
    if df.empty:
        return []

    safe_df = df.where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


def query_by_item(table_name, item_seq):
    if not table_exists(table_name):
        return pd.DataFrame()

    columns = table_columns(table_name)
    item_seq_col = first_existing(
        columns,
        ["item_seq", "ITEM_SEQ"],
    )

    if not item_seq_col:
        return pd.DataFrame()

    return pd.read_sql(
        text(f"""
            SELECT *
            FROM `{table_name}`
            WHERE CAST(`{item_seq_col}` AS CHAR) = :item_seq
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )


def load_permission(item_seq):
    table_name = "drug_permission_info"

    if not table_exists(table_name):
        return None

    df = query_by_item(table_name, item_seq)

    if df.empty:
        return None

    row = dataframe_records(df.head(1))[0]
    return row


def load_concomitant(item_seq):
    table_name = "svc_dur_concomitant_effective"

    if not table_exists(table_name):
        return []

    columns = table_columns(table_name)

    candidate_columns = [
        column
        for column in [
            "item_seq",
            "item_seq_a",
            "item_seq_b",
            "left_item_seq",
            "right_item_seq",
        ]
        if column in columns
    ]

    if not candidate_columns:
        return []

    conditions = " OR ".join(
        f"CAST(`{column}` AS CHAR) = :item_seq"
        for column in candidate_columns
    )

    df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{table_name}`
            WHERE {conditions}
        """),
        engine,
        params={"item_seq": str(item_seq)},
    )

    return dataframe_records(df)


def load_therapeutic_duplication(item_seq):
    if not table_exists(THERAPEUTIC_CACHE):
        return []

    df = query_by_item(
        THERAPEUTIC_CACHE,
        item_seq,
    )

    return dataframe_records(df)


def build_item_payload(item_seq):
    cache_df = pd.read_sql(
        text(f"""
            SELECT *
            FROM `{FINAL_CACHE}`
            WHERE item_seq = :item_seq
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

    direct = cache_df[
        cache_df["warning_level"] == "DIRECT"
    ].copy()

    reference = cache_df[
        cache_df["warning_level"] == "REFERENCE"
    ].copy()

    item_name = None
    company_name = None

    if not cache_df.empty:
        item_name = clean(cache_df.iloc[0].get("item_name"))
        company_name = clean(
            cache_df.iloc[0].get("company_name")
        )

    return {
        "item_seq": str(item_seq),
        "item_name": item_name,
        "company_name": company_name,
        "direct_dur_warnings": dataframe_records(direct),
        "ingredient_dose_references": dataframe_records(reference),
        "split_extended_release_cautions": dataframe_records(
            direct[
                direct["dur_type"]
                == "SPLIT_EXTENDED_RELEASE_CAUTION"
            ]
        ),
        "therapeutic_duplication_rules": (
            load_therapeutic_duplication(item_seq)
        ),
        "concomitant_warnings": load_concomitant(item_seq),
        "permission_information": load_permission(item_seq),
    }


def choose_sample_items(final_df):
    """
    환경변수 SAMPLE_ITEM_SEQS가 있으면 해당 품목을 사용.
    없으면 대표 품목과 캐시 내 일부 품목을 사용.
    """
    env_value = os.environ.get("SAMPLE_ITEM_SEQS")

    if env_value:
        return [
            item.strip()
            for item in env_value.split(",")
            if item.strip()
        ]

    preferred = [
        "198701676",  # 디낙스정
        "201505094",  # 트라콤세미정
        "197100097",  # 벤즈트로핀
        "202301430",  # 3성분 복합제
        "198900901",  # 캡슐 예시
    ]

    available = set(
        final_df["item_seq"]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected = [
        item
        for item in preferred
        if item in available
    ]

    if len(selected) < 5:
        extra = (
            final_df["item_seq"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        for item in extra:
            if item not in selected:
                selected.append(item)

            if len(selected) >= 5:
                break

    return selected


# =========================================================
# 실행
# =========================================================

def main():
    print("[1/7] v3 서비스 뷰 로딩")

    direct_df = read_table(DIRECT_VIEW)
    reference_df = read_table(REFERENCE_VIEW)
    review_df = read_table(REVIEW_VIEW)
    rejected_df = read_table(REJECTED_VIEW)

    print("DIRECT:", len(direct_df))
    print("REFERENCE:", len(reference_df))
    print("REVIEW:", len(review_df))
    print("REJECTED:", len(rejected_df))

    if direct_df.empty:
        raise RuntimeError(
            "DIRECT 뷰가 비어 있습니다. "
            "svc_dur_item_caution_v3_direct를 확인하세요."
        )

    print("[2/7] v3 행 표준화")

    final_direct = standardize_v3_rows(
        direct_df,
        "DIRECT",
    )

    final_reference = standardize_v3_rows(
        reference_df,
        "REFERENCE",
    )

    print("[3/7] 기존 분할주의·효능군중복 로딩")

    legacy_df = read_table(LEGACY_CACHE)

    legacy_split = standardize_legacy_split(
        legacy_df
    )

    therapeutic_df = (
        build_therapeutic_duplication_cache(
            legacy_df
        )
    )

    print("분할주의 유지:", len(legacy_split))
    print("효능군중복 유지:", len(therapeutic_df))

    print("[4/7] 최종 품목주의 캐시 생성")

    frames = [
        df
        for df in [
            final_direct,
            final_reference,
            legacy_split,
        ]
        if not df.empty
    ]

    final_df = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    final_df = final_df.drop_duplicates(
        subset=[
            "item_seq",
            "dur_type",
            "rule_id",
            "warning_level",
            "prohibition_content",
            "max_quantity_raw",
            "max_duration_raw",
        ],
        keep="first",
    ).reset_index(drop=True)

    save_dataframe(final_df, FINAL_CACHE)
    create_final_indexes()

    if not therapeutic_df.empty:
        save_dataframe(
            therapeutic_df,
            THERAPEUTIC_CACHE,
        )

    print("최종 캐시:", len(final_df))

    print("[5/7] 보고서 저장")

    final_df.to_csv(
        REPORT_DIR / "final_v3_cache.csv",
        index=False,
        encoding="utf-8-sig",
    )

    review_df.to_csv(
        REPORT_DIR / "review_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rejected_df.to_csv(
        REPORT_DIR / "rejected_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        final_df
        .groupby(
            ["warning_level", "dur_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    summary.to_csv(
        REPORT_DIR / "final_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 최종 캐시 유형별 건수 =====")
    print(summary.to_string(index=False))

    print("[6/7] 대표 샘플 JSON 생성")

    sample_items = choose_sample_items(final_df)

    payloads = [
        build_item_payload(item_seq)
        for item_seq in sample_items
    ]

    output_path = (
        OUTPUT_DIR / "all_samples_v3_final.json"
    )

    output_path.write_text(
        json.dumps(
            payloads,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("샘플 품목:", ", ".join(sample_items))
    print("JSON:", output_path.resolve())

    print("[7/7] 최종 검증")

    with engine.connect() as conn:
        total = conn.execute(
            text(
                f"SELECT COUNT(*) FROM `{FINAL_CACHE}`"
            )
        ).scalar_one()

        duplicate_count = conn.execute(
            text(f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        item_seq,
                        dur_type,
                        COALESCE(rule_id, ''),
                        warning_level,
                        COUNT(*) AS cnt
                    FROM `{FINAL_CACHE}`
                    GROUP BY
                        item_seq,
                        dur_type,
                        COALESCE(rule_id, ''),
                        warning_level
                    HAVING COUNT(*) > 1
                ) x
            """)
        ).scalar_one()

    print("DB 최종 캐시 행:", total)
    print("중복 키 그룹:", duplicate_count)

    print("\n생성 테이블:")
    print("-", FINAL_CACHE)
    print("-", THERAPEUTIC_CACHE)

    print("\n기존 테이블은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
