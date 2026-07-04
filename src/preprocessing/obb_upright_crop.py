"""
OBB 예측을 이용해 알약을 '수평으로 세운' 왜곡 없는 crop 생성 — 분류/OCR 팀 배포용.

입력:
  - manifest_with_obb.csv  : OBB 예측이 굳혀진 manifest (33,340행)
      · GT      : bbox_x/y/w/h, width, height (manifest 픽셀 좌표)
      · 예측    : pred_cx/cy/w/h, pred_angle_deg(0~180 장축), pred_conf,
                  matched(bool), px1..py4(폴리곤) — 모두 [0,1] 정규화(실제 파일 크기 기준)
      · angle_reliable(bool): 갸름한(캡슐) 알약만 True(~7%). True일 때만 회전 신뢰.
  - 원본 full 이미지 폴더 or zip (image_file = 파일명)

핵심 결정(변경 금지):
  matched & angle_reliable → 폴리곤 장축각 θ로 IMAGE를 회전(BORDER_REPLICATE)해 장축을 수평화.
  그 외(미매칭/각도 비신뢰) → 회전 없이 GT bbox 중심에서 crop.
  회전은 점사상 p'=R(θ)(p−c)+c, R(θ)=[[cos,sin],[−sin,cos]] 이 각도 θ 벡터를 θ−a로 보내므로
  a=θ 를 쓰면 장축이 수평이 됨 → ±부호 논쟁 불필요.

  ※ 색/명암/그레이스케일 전처리는 범위 밖(다운스트림 담당).
  ※ 180° 상하 모호성(각인이 뒤집힘)은 여기서 해소하지 않음 — 다운스트림(OCR/DB) 몫.
     프로젝트 규칙상 flip 절대 금지이므로 뒤집기로 세우지 않는다.

사용(CLI):
  python obb_upright_crop.py --manifest manifest_with_obb.csv --images imgs/ --out crops/
  python obb_upright_crop.py --manifest manifest_with_obb.csv --images imgs.zip --out crops/ \
         --split val --mode rect --margin 1.3 --only-reliable --viz-samples 12

사용(노트북):
  from obb_upright_crop import load_obb_manifest, upright_crop
  df = load_obb_manifest("manifest_with_obb.csv", split="val")
  crop, meta = upright_crop(cv2.imread(path), df.iloc[0])
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# passthrough 라벨 컬럼(있으면 upright_manifest에 그대로 실어 보냄)
LABEL_COLS = ["print_front", "print_back", "drug_shape", "color_class1", "color_class2",
              "item_seq", "rotation_label_deg", "rotation_label_quality"]


# --------------------------------------------------------------------------- data / io
def _as_bool(v) -> bool:
    """bool/문자열("True"/"False")/NaN 을 견고하게 bool 로 정규화."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return bool(v)


def _safe_dim(value, fallback: float) -> float:
    """width/height 스칼라를 견고하게 float 로. None/NaN/0 이하면 fallback."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    value = float(value)
    return value if value > 0 else fallback


def load_obb_manifest(csv_path, split: str | None = None) -> pd.DataFrame:
    """manifest_with_obb.csv 로드. matched/angle_reliable 을 실제 bool 로 정규화.
    split 지정 시 해당 split 만 반환."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["object_id"] = df["object_id"].astype(str)
    for col in ("matched", "angle_reliable"):
        if col in df.columns:
            df[col] = df[col].map(_as_bool)
        else:
            df[col] = False
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def ensure_images(src: Path, out_dir: Path) -> dict[str, Path]:
    """zip이면 out_dir/_raw 에 한 번만 풀고, 폴더면 그대로 사용. {파일명: 경로} 반환."""
    if src.is_dir():
        root = src
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / ".extracted"
        if not marker.exists():
            print(f"  압축 해제: {src.name} → {out_dir}")
            with zipfile.ZipFile(src) as zf:
                zf.extractall(out_dir)
            marker.touch()
        root = out_dir
    lut: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(ext):
            lut[p.name] = p
    print(f"  이미지 {len(lut):,}장 인덱싱: {root}")
    return lut


# ---------------------------------------------------------------------------- geometry
def _long_axis_angle_deg(pts: np.ndarray) -> tuple[float, float, float]:
    """OBB 4점(픽셀) → (장축각 θ°, 장축 길이 L, 단축 길이 S).
    e12=pts[1]-pts[0], e23=pts[2]-pts[1] 중 긴 쪽이 장축. θ=atan2(dy,dx)."""
    e12 = pts[1] - pts[0]
    e23 = pts[2] - pts[1]
    l12 = float(np.hypot(e12[0], e12[1]))
    l23 = float(np.hypot(e23[0], e23[1]))
    if l12 >= l23:
        long_e, L, S = e12, l12, l23
    else:
        long_e, L, S = e23, l23, l12
    theta = math.degrees(math.atan2(float(long_e[1]), float(long_e[0])))
    return theta, L, S


def _crop_exact(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    """(cx,cy) 중심에서 정확히 w×h crop. 경계 넘으면 REPLICATE 패딩해 크기 보장(축소 없음)."""
    import cv2
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1, y1 = x0 + w, y0 + h
    H, W = img.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - W)
    pad_b = max(0, y1 - H)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l
        y0 += pad_t
    return img[y0:y0 + h, x0:x0 + w]


def upright_crop(img: np.ndarray, row, mode: str = "square", margin: float = 1.3):
    """알약 1개를 수평으로 세운 crop + meta 반환.

    matched & angle_reliable → 폴리곤 장축각 θ로 이미지를 회전(BORDER_REPLICATE)해 수평화 후 crop.
    그 외 → 회전 없이 GT bbox 중심에서 crop (manifest↔실제 파일 크기 다르면 스케일 보정).

    반환: (crop, {rotated, angle_used, mode, side|w|h, center})
    """
    import cv2
    H, W = img.shape[:2]                       # 실제 파일 크기(정규화 좌표는 이 값에 곱함)

    matched = _as_bool(row.get("matched"))
    reliable = _as_bool(row.get("angle_reliable"))
    pred_cx = row.get("pred_cx")
    use_pred = matched and reliable and pred_cx is not None and not (
        isinstance(pred_cx, float) and math.isnan(pred_cx))

    if use_pred:
        # ---- 회전 경로: 폴리곤 장축을 수평으로 ----
        pts = np.array([[float(row[f"px{i}"]) * W, float(row[f"py{i}"]) * H]
                        for i in range(1, 5)], dtype=np.float64)
        theta, L, S = _long_axis_angle_deg(pts)
        cx, cy = float(row["pred_cx"]) * W, float(row["pred_cy"]) * H
        # a=θ 로 회전하면 장축(각 θ)이 θ−θ=0°(수평)로 감. 부호 논쟁 없음.
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        work = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        rotated, angle_used = True, theta
        ccx, ccy = cx, cy                       # 회전은 center 기준이라 중심 불변
        dim_x, dim_y = L, S                     # 회전 후 장축은 항상 수평
    else:
        # ---- 무회전 경로: GT bbox 중심(실제 픽셀로 스케일) ----
        work = img
        Wm = _safe_dim(row.get("width"), W)
        Hm = _safe_dim(row.get("height"), H)
        sx, sy = W / Wm, H / Hm                  # manifest→실제 파일 스케일
        bw = float(row["bbox_w"]) * sx
        bh = float(row["bbox_h"]) * sy
        ccx = (float(row["bbox_x"]) + float(row["bbox_w"]) / 2.0) * sx
        ccy = (float(row["bbox_y"]) + float(row["bbox_h"]) / 2.0) * sy
        L, S = max(bw, bh), min(bw, bh)
        dim_x, dim_y = bw, bh                    # 축정렬 그대로: 가로=bw, 세로=bh
        rotated, angle_used = False, None

    # ---- crop 크기 결정 후 정확 크기로 잘라내기 ----
    meta = {"rotated": rotated, "angle_used": angle_used, "mode": mode,
            "center": (float(ccx), float(ccy))}
    if mode == "rect":
        w = int(round(dim_x * margin))
        h = int(round(dim_y * margin))
        meta["w"], meta["h"] = w, h
    else:                                        # square: 긴 축 기준 정사각
        side = int(round(max(dim_x, dim_y) * margin))
        w = h = side
        meta["side"] = side
    crop = _crop_exact(work, ccx, ccy, max(w, 1), max(h, 1))
    return crop, meta


# ------------------------------------------------------------------------- sanity viz
def _viz_grid(crops, out_path, n=12):
    """rotated=True 우선으로 crop 격자 저장 → upright_check_{split}.png"""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crops = [c for c in crops if c is not None and c[0] is not None and c[0].size]
    crops.sort(key=lambda c: 0 if c[1] else 1)   # rotated 먼저
    crops = crops[:n]
    if not crops:
        print("  [viz] 샘플 없음, 스킵")
        return
    cols = min(4, len(crops))
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, (im, rot, name) in zip(axes, crops):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\n{'rot' if rot else 'no-rot'}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(crops):]:
        ax.axis("off")
    plt.suptitle("upright check — 각인이 수평인지 확인", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"  [viz] 저장: {out_path}")


# ---------------------------------------------------------------------------- pipeline
def _iter_progress(seq, total):
    try:
        from tqdm import tqdm
        return tqdm(seq, total=total)
    except Exception:
        def gen():
            for i, x in enumerate(seq):
                if i % 500 == 0:
                    print(f"    {i:,}/{total:,}")
                yield x
        return gen()


def run_split(df: pd.DataFrame, lut: dict, out_root: Path, split: str,
              mode: str, margin: float, only_reliable: bool, viz_n: int) -> None:
    import cv2
    sub = df[df["split"] == split] if "split" in df.columns else df
    if only_reliable:
        sub = sub[sub["angle_reliable"]]
    sub = sub.reset_index(drop=True)
    crop_dir = out_root / split
    crop_dir.mkdir(parents=True, exist_ok=True)
    present = [c for c in LABEL_COLS if c in sub.columns]

    rows, viz = [], []
    n_rot = n_norot = n_miss = 0
    for _, r in _iter_progress((r for _, r in sub.iterrows()), len(sub)):
        src = lut.get(r["image_file"])
        img = cv2.imread(str(src)) if src is not None else None
        if img is None:
            n_miss += 1
            continue
        crop, meta = upright_crop(img, r, mode=mode, margin=margin)
        if crop is None or crop.size == 0:
            n_miss += 1
            continue
        fname = f"{r['object_id']}.png"
        cv2.imwrite(str(crop_dir / fname), crop)
        if meta["rotated"]:
            n_rot += 1
        else:
            n_norot += 1
        rec = {"object_id": r["object_id"], "crop_file": f"{split}/{fname}", "split": split,
               "rotated": meta["rotated"], "angle_used": meta["angle_used"], "mode": mode,
               "margin": margin, "angle_reliable": _as_bool(r.get("angle_reliable")),
               "matched": _as_bool(r.get("matched"))}
        for c in present:
            rec[c] = r.get(c)
        rows.append(rec)
        if len(viz) < viz_n * 3:
            viz.append((crop, meta["rotated"], r["object_id"]))

    mani = out_root / f"upright_manifest_{split}.csv"
    pd.DataFrame(rows).to_csv(mani, index=False)
    print(f"  [{split}] total {len(rows):,} / rotated {n_rot:,} / unrotated {n_norot:,} / "
          f"missing-image {n_miss:,}  → {mani.name}")
    try:
        _viz_grid(viz, out_root / f"upright_check_{split}.png", viz_n)
    except Exception as e:
        print(f"  [viz] {split} 실패(무시): {e}")


# --------------------------------------------------------------------------- cli / main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBB 예측 → 수평 세운 crop 배포")
    p.add_argument("--manifest", type=Path, required=True, help="manifest_with_obb.csv")
    p.add_argument("--images", type=Path, required=True, help="full 이미지 폴더 또는 zip")
    p.add_argument("--out", type=Path, required=True, help="crop 저장 위치")
    p.add_argument("--split", nargs="+", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--mode", choices=["square", "rect"], default="square",
                   help="square=정사각(분류용) / rect=갸름한 직사각(OCR strip용)")
    p.add_argument("--margin", type=float, default=1.3, help="crop 여백 배수")
    p.add_argument("--only-reliable", action="store_true",
                   help="angle_reliable 행만 처리(OCR 팀이 회전된 것만 원할 때)")
    p.add_argument("--viz-samples", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_obb_manifest(args.manifest)
    lut = ensure_images(args.images, args.out / "_raw")
    print(f"[upright] mode={args.mode} margin={args.margin} "
          f"only_reliable={args.only_reliable}")
    for split in args.split:
        run_split(df, lut, args.out, split, args.mode, args.margin,
                  args.only_reliable, args.viz_samples)
    print(f"[upright] 완료 → {args.out}")


if __name__ == "__main__":
    main()
