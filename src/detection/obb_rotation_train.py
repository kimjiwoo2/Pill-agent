"""
Pill OBB (oriented bbox) 학습 파이프라인 — 연구실 GPU / VSCode-SSH 용.

입력:
  - 회전각 라벨  : ys_rotation_labels_{train,val}_v1.csv   (object_id, rotation_label_deg, quality, ...)
  - 좌표 manifest: manifest_clean_20k_33340.csv            (object_id, image_file, bbox_x/y/w/h, width, height)
  - 원본 이미지  : images_train.zip / images_val.zip        (full image = image_file)

흐름:
  1) rotation ⨝ manifest (object_id)  → 축정렬 bbox + 각도 결합
  2) 이미지 단위 필터: 완전라벨(모든 알약 존재) + quality(high[+medium]) + bbox 유효
  3) 축정렬 bbox + 각도  → bbox를 '포함'하는 회전박스(안 잘림) → YOLO OBB 4점 폴리곤 라벨
     (다운스트림에서 한 번 더 tight crop 전제라 '자르지 않고 감싸기'만 하면 됨)
  4) 각도 회전방향(convention)이 데이터마다 다를 수 있어 4가지를 겹쳐 그려 눈으로 확정
  5) yolo11n-obb 학습

사용:
  python obb_rotation_train.py --mode verify   --data-root /data/pill_obb      # 먼저: convention_grid.png 확인
  python obb_rotation_train.py --mode build    --data-root /data/pill_obb --convention 0
  python obb_rotation_train.py --mode train    --data-root /data/pill_obb
  python obb_rotation_train.py --mode predict  --data-root /data/pill_obb   # 예측 라벨 굳혀 배포
  python obb_rotation_train.py --mode angle-diag --data-root /data/pill_obb --convention <build값>  # 각도 붕괴 진단
  # (verify에서 어느 열이 알약에 딱 붙는지 보고 --convention 0~3 지정)
  # (박스가 알약을 자르면 --box-margin 1.15 처럼 키우면 됨)

배포(predict):
  best.pt 예측을 object_id별 top-1 OBB로 굳혀 CSV(work/predictions/obb_predictions_{split}.csv)로 저장.
  팀원은 YOLO 없이  manifest.merge(pred, on="object_id", how="left")  한 줄로 붙임.
  컬럼: object_id, image_file, pred_cx/cy/w/h([0,1] 정규화; ×이미지 W/H로 픽셀화),
        pred_angle_deg(0~180 기울기), pred_conf, match_iou, n_cand(중복검출 collapse 수),
        matched, px1..py4(폴리곤 [0,1]). ※좌표는 파일 크기·manifest W/H 불일치에 안전한 정규화.
  provenance(.meta.json): 모델 sha256 · git commit · conf/iou 임계 → 원본 갱신 시 재생성 근거.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# (sign, offset_deg) — 박스 각도 phi = (sign*rotation_label_deg + offset) % 180
CONVENTIONS = [(+1, 0), (+1, 90), (-1, 0), (-1, 90)]


# ----------------------------------------------------------------------------- config
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["verify", "build", "train", "all", "straighten", "predict", "angle-diag"],
                   default="build")
    p.add_argument("--data-root", type=Path, required=True,
                   help="아래 파일들이 있는 로컬 디렉토리")
    p.add_argument("--rot-train", default="ys_rotation_labels_train_v1.csv")
    p.add_argument("--rot-val",   default="ys_rotation_labels_val_v1.csv")
    p.add_argument("--manifest",  default="manifest_clean_20k_33340.csv")
    p.add_argument("--img-train", default="images_train.zip", help="zip 또는 이미 푼 폴더")
    p.add_argument("--img-val",   default="images_val.zip")
    p.add_argument("--work-dir",  type=Path, default=None, help="산출물 위치 (기본: data-root/obb_work)")

    # 필터
    p.add_argument("--quality", nargs="+", default=["high", "medium"],
                   help="각도 신뢰도 등급 채택 (기본 high+medium; 엄격히 하려면 high)")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="한 이미지의 일부 알약만 라벨돼도 사용 (기본: 완전라벨만)")

    # 기하 convention (verify로 정함)
    p.add_argument("--convention", type=int, default=0, choices=range(len(CONVENTIONS)))
    p.add_argument("--box-margin", type=float, default=1.0,
                   help="회전박스 크기 배수 (1.0=bbox 감쌈, >1 여유 더). 절대 안 잘리게 감싸는 방식")
    # straighten (OCR/분류 학습용: GT 각도로 펴서 저장 + zip)
    p.add_argument("--straight-out", type=Path, default=None,
                   help="펴진 crop 저장 위치 (기본 data-root/straightened)")
    p.add_argument("--straight-pad", type=float, default=0.10, help="crop 여백 비율")
    p.add_argument("--straight-sign", type=int, default=-1, choices=[-1, 1],
                   help="회전 부호. straighten_check.png에서 글자가 뒤집혀 나오면 반대로")

    # predict (예측 라벨 굳혀 배포: object_id별 top-1 OBB → manifest 조인용 CSV)
    p.add_argument("--weights", type=Path, default=None,
                   help="best.pt 경로 (기본: work/runs/**/weights/best.pt 최신본)")
    p.add_argument("--pred-out", type=Path, default=None,
                   help="예측 CSV 저장 위치 (기본 work/predictions)")
    p.add_argument("--pred-conf", type=float, default=0.25,
                   help="검출 conf 임계 (top-1 dedup 이전 raw 검출)")
    p.add_argument("--pred-iou", type=float, default=0.0,
                   help="선택: 외접박스 IoU 최소 게이트 (기본 0=off; OBB가 느슨해 매칭은 중심 기반)")
    p.add_argument("--pred-center-tol", type=float, default=0.03,
                   help="예측 중심이 GT bbox 밖이어도 이 거리(정규화) 이내면 매칭 (기본 0.03)")
    p.add_argument("--pred-splits", nargs="+", default=["train", "val"],
                   choices=["train", "val"], help="예측을 굳힐 split")

    # angle-diag (OBB 각도 붕괴 진단: GT 라벨 각도·박스종횡비 vs best.pt 예측 각도)
    p.add_argument("--diag-n", type=int, default=300,
                   help="예측 각도 분포 측정용 val 이미지 수")

    # 학습
    p.add_argument("--model", default="yolo11n-obb.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="yolo11n_obb_v1")
    p.add_argument("--viz-samples", type=int, default=12, help="sanity 시각화 샘플 수")
    return p.parse_args()


# --------------------------------------------------------------------------- data join
def load_merged(args: argparse.Namespace, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rot_path = args.data_root / (args.rot_train if split == "train" else args.rot_val)
    man = pd.read_csv(args.data_root / args.manifest, low_memory=False)
    rot = pd.read_csv(rot_path, low_memory=False)

    rot["object_id"] = rot["object_id"].astype(str)
    man["object_id"] = man["object_id"].astype(str)
    man = man.drop_duplicates("object_id", keep="first")   # object_id 1:1 보장 (중복 라벨 방지)
    m = rot.merge(man, on="object_id", how="left", suffixes=("", "_man"))
    return m, man


def filter_split(m: pd.DataFrame, man: pd.DataFrame, quality: list[str],
                 require_complete: bool) -> pd.DataFrame:
    need = ["image_file", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "width", "height",
            "rotation_label_deg"]                          # 각도 NaN 행 제거 (라벨에 nan 새는 것 방지)
    m = m.dropna(subset=need).copy()

    valid_box = (
        (m["bbox_w"] > 0) & (m["bbox_h"] > 0)
        & (m["bbox_x"] >= 0) & (m["bbox_y"] >= 0)
        & (m["bbox_x"] + m["bbox_w"] <= m["width"])
        & (m["bbox_y"] + m["bbox_h"] <= m["height"])
    )
    m = m[valid_box]

    good = m[m["rotation_label_quality"].isin(quality)].copy()

    if require_complete:
        man_cnt = man.groupby("image_file")["object_id"].nunique()
        good_cnt = good.groupby("image_file")["object_id"].nunique()
        keep = good_cnt.index[good_cnt.values >= man_cnt.reindex(good_cnt.index).values]
        good = good[good["image_file"].isin(keep)]

    return good.reset_index(drop=True)


# ---------------------------------------------------------------------------- geometry
def obb_corners_px(cx, cy, bw, bh, theta_deg, sign, offset, margin=1.0):
    """이미지 픽셀 좌표(x→우, y→하) 기준 회전사각형 4점.
    축정렬 bbox를 '포함'하는 회전박스 → 알약이 절대 안 잘림(각도만 정확, 크기는 여유).
    margin>1 이면 사방 여유 더. 다운스트림에서 한 번 더 tight crop 전제."""
    phi = (sign * theta_deg + offset) % 180
    r = math.radians(phi)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    L = (bw * c + bh * s) * margin      # 긴 축: 축정렬 bbox를 감싸는 크기
    S = (bw * s + bh * c) * margin      # 짧은 축
    ux, uy = math.cos(r), math.sin(r)   # length 축
    vx, vy = -math.sin(r), math.cos(r)  # width 축
    hl, hs = L / 2, S / 2
    return [
        (cx + hl * ux + hs * vx, cy + hl * uy + hs * vy),
        (cx + hl * ux - hs * vx, cy + hl * uy - hs * vy),
        (cx - hl * ux - hs * vx, cy - hl * uy - hs * vy),
        (cx - hl * ux + hs * vx, cy - hl * uy + hs * vy),
    ]


def obb_label_line(row, sign, offset, margin=1.0) -> str:
    cx = row["bbox_x"] + row["bbox_w"] / 2
    cy = row["bbox_y"] + row["bbox_h"] / 2
    pts = obb_corners_px(cx, cy, row["bbox_w"], row["bbox_h"],
                         row["rotation_label_deg"], sign, offset, margin)
    W, H = row["width"], row["height"]
    coords = []
    for x, y in pts:
        coords.append(min(max(x / W, 0.0), 1.0))
        coords.append(min(max(y / H, 0.0), 1.0))
    return "0 " + " ".join(f"{v:.6f}" for v in coords)


# ------------------------------------------------------------------------- images / io
def ensure_images(src: Path, out_dir: Path) -> dict[str, Path]:
    """zip이면 풀고, 폴더면 그대로 사용. {파일명: 경로} 반환."""
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
    lut = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(ext):
            lut[p.name] = p
    print(f"  이미지 {len(lut):,}장 인덱싱: {root}")
    return lut


def img_source(args, split):
    return args.data_root / (args.img_train if split == "train" else args.img_val)


# ------------------------------------------------------------------------- sanity viz
def _draw_obb_grid(items, out_path, title, n_cols=4):
    """items: [(image_path, [poly_px,...])] → 격자 저장. poly_px = [(x,y) 4개]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    items = [it for it in items if it is not None]
    if not items:
        print(f"  [viz] {out_path.name}: 샘플 없음, 스킵")
        return
    n = len(items)
    cols = min(n_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, (img_path, polys) in zip(axes, items):
        img = cv2.imread(str(img_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for poly in polys:
                ax.add_patch(plt.Polygon(poly, closed=True, fill=False,
                                         edgecolor="lime", linewidth=2))
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"  [viz] 저장: {out_path}")


def viz_build_check(work, split, n=12):
    """디스크에 쓴 OBB 라벨(.txt)을 이미지에 겹쳐 확인 → build_check_{split}.png"""
    import cv2
    img_dir = work / "dataset" / "images" / split
    lbl_dir = work / "dataset" / "labels" / split
    lbls = sorted(lbl_dir.glob("*.txt"))
    if not lbls:
        return
    pick = np.unique(np.linspace(0, len(lbls) - 1, min(n, len(lbls))).astype(int))
    items = []
    for i in pick:
        lp = lbls[i]
        cand = list(img_dir.glob(lp.stem + ".*"))
        if not cand:
            continue
        img = cv2.imread(str(cand[0]))
        if img is None:
            continue
        H, W = img.shape[:2]
        polys = []
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) < 9:
                continue
            xy = list(map(float, v[1:9]))
            polys.append([(xy[j] * W, xy[j + 1] * H) for j in range(0, 8, 2)])
        items.append((cand[0], polys))
    _draw_obb_grid(items, work / f"build_check_{split}.png",
                   f"build check — {split} (디스크 라벨 → 이미지 오버레이)")


def viz_predict_check(model, work, imgsz, n=12):
    """학습된 모델로 val 이미지 예측 OBB 그려 확인 → predict_check.png"""
    img_dir = work / "dataset" / "images" / "val"
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not imgs:
        return
    idx = np.unique(np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int))
    pick = [imgs[i] for i in idx]
    results = model.predict(source=[str(p) for p in pick], imgsz=imgsz,
                            conf=0.25, verbose=False)
    items = []
    for p, r in zip(pick, results):
        polys = []
        obb = getattr(r, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
            for poly in obb.xyxyxyxy.cpu().numpy():
                polys.append([(float(x), float(y)) for x, y in poly])
        items.append((p, polys))
    _draw_obb_grid(items, work / "predict_check.png",
                   "predict check — val (모델 예측 OBB)")


# ---------------------------------------------------------------------------- commands
def cmd_build(args, work):
    sign, offset = CONVENTIONS[args.convention]
    print(f"[build] convention={args.convention} (sign={sign}, offset={offset}), "
          f"box_margin={args.box_margin}, quality={args.quality}, "
          f"complete={'no' if args.allow_incomplete else 'yes'}")
    _, man = load_merged(args, "train")

    yaml_lines = [f"path: {work / 'dataset'}", "train: images/train", "val: images/val",
                  "nc: 1", "names: [pill]"]
    (work / "dataset").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        m, _ = load_merged(args, split)
        df = filter_split(m, man, args.quality, require_complete=not args.allow_incomplete)
        lut = ensure_images(img_source(args, split), work / "raw" / split)

        img_dir = work / "dataset" / "images" / split
        lbl_dir = work / "dataset" / "labels" / split
        for d in (img_dir, lbl_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        n_img, n_obj, miss = 0, 0, 0
        for fname, grp in df.groupby("image_file"):
            src = lut.get(fname)
            if src is None:
                miss += 1
                continue
            lines = [obb_label_line(r, sign, offset, args.box_margin) for _, r in grp.iterrows()]
            (lbl_dir / f"{Path(fname).stem}.txt").write_text("\n".join(lines))
            link = img_dir / fname
            if not link.exists():
                os.symlink(src.resolve(), link)
            n_img += 1
            n_obj += len(lines)
        print(f"  [{split}] 이미지 {n_img:,} / 알약 {n_obj:,}  (원본 누락 {miss})")

    (work / "dataset" / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"[build] dataset.yaml → {work / 'dataset' / 'dataset.yaml'}")
    for sp in ["train", "val"]:                      # sanity: 만든 라벨 눈으로 확인
        try:
            viz_build_check(work, sp, args.viz_samples)
        except Exception as e:
            print(f"  [viz] build check {sp} 실패(무시): {e}")


def cmd_verify(args, work, n_samples=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    _, man = load_merged(args, "val")
    m, _ = load_merged(args, "val")
    df = filter_split(m, man, args.quality, require_complete=not args.allow_incomplete)
    lut = ensure_images(img_source(args, "val"), work / "raw" / "val")

    # 각도가 다양하게 걸리도록 샘플
    df = df.sort_values("rotation_label_deg")
    idx = np.linspace(0, len(df) - 1, n_samples).astype(int)
    samples = df.iloc[idx]

    fig, axes = plt.subplots(n_samples, len(CONVENTIONS),
                             figsize=(len(CONVENTIONS) * 3, n_samples * 3))
    for c, (sign, offset) in enumerate(CONVENTIONS):
        axes[0, c].set_title(f"conv {c}\n(sign={sign}, off={offset})", fontsize=9)
    for r, (_, row) in enumerate(samples.iterrows()):
        src = lut.get(row["image_file"])
        img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB) if src else None
        cx = row["bbox_x"] + row["bbox_w"] / 2
        cy = row["bbox_y"] + row["bbox_h"] / 2
        for c, (sign, offset) in enumerate(CONVENTIONS):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img)
                pts = obb_corners_px(cx, cy, row["bbox_w"], row["bbox_h"],
                                     row["rotation_label_deg"], sign, offset, args.box_margin)
                poly = plt.Polygon(pts, fill=False, edgecolor="lime", linewidth=2)
                ax.add_patch(poly)
            if c == 0:
                ax.set_ylabel(f"deg={int(row['rotation_label_deg'])}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    out = work / "convention_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"[verify] 저장: {out}")
    print("  → 각 열 중 박스가 알약에 '딱' 붙는 열의 번호를 --convention 으로 지정하세요.")


# --------------------------------------------------------------------- straighten set
def straighten_crop(img, cx, cy, bw, bh, deg, sign, pad):
    """GT 회전각(deg, 360°)으로 알약을 정방향 수평으로 펴서 crop 반환.
    회전 후에도 안 잘리게: 원 bbox 네 모서리를 같은 회전으로 옮겨 그 bounding box를 crop
    (bbox 폭으로 자르면 기울어진 길쭉한 알약의 길이가 잘리는 문제 방지)."""
    import cv2
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), sign * float(deg), 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    corners = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                        [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]], dtype=np.float64)
    pts = (M[:, :2] @ corners.T + M[:, 2:3]).T          # 회전 프레임 좌표
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    px, py = (x1 - x0) * pad / 2, (y1 - y0) * pad / 2
    X0, Y0 = max(0, int(round(x0 - px))), max(0, int(round(y0 - py)))
    X1, Y1 = int(round(x1 + px)), int(round(y1 + py))
    crop = rot[Y0:Y1, X0:X1]
    if crop.size and crop.shape[0] > crop.shape[1]:     # 세로로 길면 가로로 눕히기
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _viz_straighten(crop_dir, out_path, n=12):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(crop_dir.glob("*.png"))
    if not files:
        return
    pick = np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).astype(int))
    imgs = [(cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB), files[i].stem)
            for i in pick]
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, (im, name) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    plt.suptitle("straighten check — 각인이 수평·정방향인지 확인", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"  [viz] 저장: {out_path}")


def cmd_straighten(args, work):
    """GT 회전각으로 알약을 정방향으로 펴서 저장 + split별 zip (+ 라벨 manifest).
    OCR/분류 학습용 데이터. 추론(파이프라인)은 저장 없이 편 crop만 쓰면 됨."""
    import cv2
    out_root = args.straight_out or (args.data_root / "straightened")
    out_root.mkdir(parents=True, exist_ok=True)
    _, man = load_merged(args, "train")
    label_cols = ["object_id", "image_file", "split", "print_front", "print_back",
                  "drug_shape", "color_class1", "color_class2", "dataset_type",
                  "item_seq", "rotation_label_deg", "rotation_label_quality"]
    print(f"[straighten] sign={args.straight_sign}, pad={args.straight_pad} → {out_root}")
    for split in ["train", "val"]:
        m, _ = load_merged(args, split)
        df = filter_split(m, man, args.quality, require_complete=not args.allow_incomplete)
        lut = ensure_images(img_source(args, split), work / "raw" / split)
        crop_dir = out_root / split
        shutil.rmtree(crop_dir, ignore_errors=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

        rows, n_ok, n_miss = [], 0, 0
        for _, r in df.iterrows():
            src = lut.get(r["image_file"])
            img = cv2.imread(str(src)) if src is not None else None
            if img is None:
                n_miss += 1
                continue
            cx = r["bbox_x"] + r["bbox_w"] / 2
            cy = r["bbox_y"] + r["bbox_h"] / 2
            crop = straighten_crop(img, cx, cy, r["bbox_w"], r["bbox_h"],
                                   r["rotation_label_deg"], args.straight_sign, args.straight_pad)
            if crop is None or crop.size == 0:
                n_miss += 1
                continue
            cv2.imwrite(str(crop_dir / f"{r['object_id']}.png"), crop)
            rows.append({c: r.get(c) for c in label_cols if c in df.columns})
            n_ok += 1

        mani_path = out_root / f"straightened_manifest_{split}.csv"
        pd.DataFrame(rows).to_csv(mani_path, index=False)
        zip_path = out_root / f"straightened_{split}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in crop_dir.glob("*.png"):
                zf.write(p, arcname=p.name)
            zf.write(mani_path, arcname=mani_path.name)
        print(f"  [{split}] 저장 {n_ok:,} / 누락 {n_miss}  → {zip_path.name} "
              f"({zip_path.stat().st_size / 1e9:.2f} GB)")
        try:
            _viz_straighten(crop_dir, out_root / f"straighten_check_{split}.png", args.viz_samples)
        except Exception as e:
            print(f"  [viz] straighten check {split} 실패(무시): {e}")
    print(f"[straighten] 완료 → {out_root}")


def _poly_to_xyxy(poly):
    """OBB 4점 → 축정렬 외접박스 (x0,y0,x1,y1). GT bbox와 IoU 매칭용."""
    xs, ys = poly[:, 0], poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _resolve_weights(args, work):
    if args.weights is not None:
        return Path(args.weights)
    cands = sorted((work / "runs").rglob("weights/best.pt"),
                   key=lambda p: p.stat().st_mtime)
    assert cands, "best.pt 자동탐색 실패 — --weights 로 경로 지정"
    return cands[-1]


def _provenance(weights: Path, args) -> dict:
    import hashlib
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    try:
        import ultralytics
        uv = ultralytics.__version__
    except Exception:
        uv = None
    return {"weights": str(weights), "weights_sha256_16": sha, "git_commit": commit,
            "ultralytics": uv, "imgsz": args.imgsz, "pred_conf": args.pred_conf,
            "match": "center-in-gtbox", "pred_center_tol": args.pred_center_tol,
            "pred_iou_gate": args.pred_iou, "coords": "normalized_[0,1]"}


def viz_pred_labels(out_root, split, df, lut, n=12):
    """굳힌 top-1 OBB(CSV의 px1..py4 정규화폴리곤)를 이미지에 그려 확인.
    raw 예측이 아니라 '알약당 1박스'로 dedup된 배포 라벨을 그림 → predict_labels_check_{split}.png"""
    import cv2
    if "matched" not in df.columns or not all(f"px{i}" in df.columns for i in range(1, 5)):
        return
    d = df[df["matched"] == True].copy()
    if d.empty:
        return
    counts = d.groupby("image_file").size().sort_values(ascending=False)
    multi = [f for f in counts.index if counts[f] >= 2][:4]      # 조합(다중 알약) 이미지 우선
    singles = [f for f in counts.index if counts[f] == 1]
    pick = list(multi)
    if singles and n > len(pick):
        idx = np.unique(np.linspace(0, len(singles) - 1, n - len(pick)).astype(int))
        pick += [singles[i] for i in idx]
    items = []
    for f in pick[:n]:
        p = lut.get(f)
        img = cv2.imread(str(p)) if p is not None else None
        if img is None:
            continue
        H, W = img.shape[:2]
        polys = [[(r[f"px{i}"] * W, r[f"py{i}"] * H) for i in range(1, 5)]
                 for _, r in d[d["image_file"] == f].iterrows()]
        items.append((p, polys))
    _draw_obb_grid(items, out_root / f"predict_labels_check_{split}.png",
                   f"predict labels — {split} (굳힌 top-1 OBB, 알약당 1박스)")


def cmd_predict(args, work):
    """best.pt 예측을 object_id별 top-1 OBB로 굳혀 CSV 배포 (팀원 manifest 조인용).

    좌표는 [0,1] 정규화(예측은 실제 파일 orig_shape, GT는 manifest width/height 기준)로 맞춰
    IoU 매칭 → 파일 크기와 manifest W/H가 달라도 안전. 한 알약에 예측이 여러 개면(박스 2개
    중복검출) conf 최고 1개만 남김(top-1). GT는 있는데 예측 없으면 matched=False(NaN)로 남겨
    조인이 안전하게 degrade. angle은 OBB 기울기(0~180°); 세우려면 -pred_angle_deg 회전 후
    위/아래는 별도 해소(OCR/DB). 조인:  manifest.merge(pred, on='object_id', how='left')
    """
    from ultralytics import YOLO
    weights = _resolve_weights(args, work)
    model = YOLO(str(weights))
    prov = _provenance(weights, args)
    out_root = args.pred_out or (work / "predictions")
    out_root.mkdir(parents=True, exist_ok=True)
    _, man = load_merged(args, "train")          # 좌표 GT(모든 split 공용); 회전 quality 필터 안 함
    print(f"[predict] weights={weights.name} (sha {prov['weights_sha256_16']}), "
          f"conf={args.pred_conf}, iou_assign={args.pred_iou}")

    for split in args.pred_splits:
        lut = ensure_images(img_source(args, split), work / "raw" / split)
        gt = man[man["image_file"].isin(lut.keys())].copy()
        img_files = sorted(gt["image_file"].unique())
        rows, st = [], dict(imgs=0, gt=0, matched=0, dup=0, extra=0, no_pred=0)
        dbg = True                                   # 첫 GT 1건 좌표계 진단 출력
        BATCH = 48
        for i in range(0, len(img_files), BATCH):
            chunk = img_files[i:i + BATCH]
            results = model.predict(source=[str(lut[f]) for f in chunk],
                                    imgsz=args.imgsz, conf=args.pred_conf, verbose=False)
            for fname, r in zip(chunk, results):
                st["imgs"] += 1
                oh, ow = getattr(r, "orig_shape", (None, None))   # 실제 파일 (H, W)
                preds = []
                obb = getattr(r, "obb", None)
                if ow and obb is not None and getattr(obb, "xyxyxyxy", None) is not None and len(obb) > 0:
                    polys = obb.xyxyxyxy.cpu().numpy() / np.array([ow, oh], dtype=np.float64)  # → [0,1]
                    xywhr = obb.xywhr.cpu().numpy()
                    confs = obb.conf.cpu().numpy()
                    for k in range(len(polys)):
                        env = _poly_to_xyxy(polys[k])
                        preds.append({"poly": polys[k], "xywhr": xywhr[k], "conf": float(confs[k]),
                                      "env": env, "c": ((env[0] + env[2]) / 2, (env[1] + env[3]) / 2)})
                used = [False] * len(preds)
                for _, g in gt[gt["image_file"] == fname].iterrows():
                    st["gt"] += 1
                    Wm, Hm = float(g["width"]), float(g["height"])
                    row = {"object_id": g["object_id"], "image_file": fname}
                    cand = []
                    if Wm > 0 and Hm > 0 and preds:
                        x0, y0 = g["bbox_x"] / Wm, g["bbox_y"] / Hm
                        x1, y1 = (g["bbox_x"] + g["bbox_w"]) / Wm, (g["bbox_y"] + g["bbox_h"]) / Hm
                        gcx, gcy = (x0 + x1) / 2, (y0 + y1) / 2
                        # 매칭 = 예측 중심이 GT bbox 안(또는 tol 이내). OBB가 느슨/회전이라 외접
                        # IoU는 중심이 맞아도 낮게 나오므로, 중심 기반이 견고. IoU는 참고용 게이트.
                        for k in range(len(preds)):
                            pcx, pcy = preds[k]["c"]
                            inside = x0 <= pcx <= x1 and y0 <= pcy <= y1
                            dist = math.hypot(pcx - gcx, pcy - gcy)
                            if inside or dist <= args.pred_center_tol:
                                iou = _iou_xyxy(preds[k]["env"], (x0, y0, x1, y1))
                                if iou >= args.pred_iou:
                                    cand.append((k, 1 if inside else 0, dist, iou))
                        if dbg:
                            bi = max((c[3] for c in cand), default=0.0)
                            print(f"  [dbg] {fname}: file={ow}x{oh} manifest={int(Wm)}x{int(Hm)} "
                                  f"cand={len(cand)} best_iou={bi:.2f}")
                            dbg = False
                    if cand:
                        kb = max(cand, key=lambda t: (t[1], preds[t[0]]["conf"]))[0]
                        sel = next(c for c in cand if c[0] == kb)
                        p = preds[kb]
                        cx, cy, w, h, rad = (float(v) for v in p["xywhr"])
                        row.update(pred_cx=cx / ow, pred_cy=cy / oh, pred_w=w / ow, pred_h=h / oh,
                                   pred_angle_deg=math.degrees(rad) % 180.0,
                                   pred_conf=p["conf"], match_iou=float(sel[3]),
                                   match_dist=float(sel[2]), n_cand=len(cand), matched=True)
                        for ci, (px, py) in enumerate(p["poly"]):
                            row[f"px{ci + 1}"], row[f"py{ci + 1}"] = float(px), float(py)
                        used[kb] = True
                        st["matched"] += 1
                        if len(cand) > 1:
                            st["dup"] += 1
                    else:
                        row["matched"] = False
                        st["no_pred"] += 1
                    rows.append(row)
                st["extra"] += sum(1 for u in used if not u)

        df = pd.DataFrame(rows)
        csv_path = out_root / f"obb_predictions_{split}.csv"
        df.to_csv(csv_path, index=False)
        (out_root / f"obb_predictions_{split}.meta.json").write_text(
            json.dumps({**prov, "split": split, "n_rows": len(df), **st},
                       ensure_ascii=False, indent=2))
        cov = st["matched"] / st["gt"] if st["gt"] else 0.0
        print(f"  [{split}] 이미지 {st['imgs']:,} / GT {st['gt']:,} / 매칭 {st['matched']:,} "
              f"({cov:.1%}) / 중복collapse {st['dup']:,} / 미검출 {st['no_pred']:,} / "
              f"GT밖 예측 {st['extra']:,}")
        print(f"          → {csv_path.name} (+ .meta.json)")
        try:                                         # sanity: 굳힌 top-1 라벨 눈으로 확인
            viz_pred_labels(out_root, split, df, lut, args.viz_samples)
        except Exception as e:
            print(f"  [viz] predict labels {split} 실패(무시): {e}")
    print(f"[predict] 완료 → {out_root}\n"
          f"          조인:  manifest.merge(pd.read_csv('obb_predictions_<split>.csv'), "
          f"on='object_id', how='left')")


def cmd_train(args, work):
    from ultralytics import YOLO
    data_yaml = work / "dataset" / "dataset.yaml"
    assert data_yaml.exists(), "먼저 --mode build 실행"
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers,
        optimizer="AdamW", lr0=0.002, patience=15,
        fliplr=0.0, flipud=0.0,          # 방향이 라벨이라 좌우/상하 반전 금지
        project=str(work / "runs"), name=args.name,
    )
    metrics = model.val()
    print(f"[train] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    try:                                             # sanity: 예측 결과 눈으로 확인
        viz_predict_check(model, work, args.imgsz, args.viz_samples)
    except Exception as e:
        print(f"  [viz] predict check 실패(무시): {e}")


def cmd_angle_diag(args, work):
    """OBB 각도 붕괴 진단. 배포된 pred_angle_deg가 ≈상수(49.5°±4°)로 죽어 있는데,
    원인이 (a)컨테인먼트 라벨 기하(near-square라 각도 소실) / (b)회전라벨 자체가 안 변함 /
    (c)predict export가 w/h 스왑을 안 반영 중 무엇인지 가른다. 모델 없이도 GT 쪽 신호로 절반 판정.

    GT 박스 각도 phi=(sign*rotation_label+offset)%180, 컨테인먼트 종횡비 L/S=(bw·|c|+bh·|s|)/(bw·|s|+bh·|c|)
    — 이 L/S가 phi≈45°에서 1(정사각)로 붕괴하면 각도가 학습 불가.
    """
    sign, offset = CONVENTIONS[args.convention]
    _, man = load_merged(args, "train")
    print("=" * 66)
    print(f"[angle-diag] convention={args.convention} (sign={sign}, offset={offset}), "
          f"box_margin={args.box_margin}")
    print("-- GT 라벨: 회전각 다양성 & 컨테인먼트 박스 종횡비 --")
    for split in ("train", "val"):
        m, _ = load_merged(args, split)
        df = filter_split(m, man, args.quality, require_complete=not args.allow_incomplete)
        if df.empty:
            print(f"  [{split}] (없음)")
            continue
        theta = df["rotation_label_deg"].to_numpy(float)
        phi = (sign * theta + offset) % 180.0
        r = np.radians(phi)
        c, s = np.abs(np.cos(r)), np.abs(np.sin(r))
        bw, bh = df["bbox_w"].to_numpy(float), df["bbox_h"].to_numpy(float)
        L, S = bw * c + bh * s, bw * s + bh * c
        asp = np.maximum(L, S) / np.maximum(np.minimum(L, S), 1e-9)
        bbox_asp = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1e-9)
        print(f"  [{split}] n={len(df):,}  rotation_label std={theta.std():.1f}°  "
              f"box_phi std={phi.std():.1f}°")
        print(f"        bbox 종횡비 median={np.median(bbox_asp):.2f}  →  "
              f"컨테인먼트 L/S median={np.median(asp):.2f} "
              f"(p10={np.percentile(asp,10):.2f} p90={np.percentile(asp,90):.2f})  "
              f"near-square(<1.2)={np.mean(asp < 1.2)*100:.0f}%")

    # ---- 모델 예측 각도 분포 ----
    from ultralytics import YOLO
    weights = _resolve_weights(args, work)
    model = YOLO(str(weights))
    lut = ensure_images(img_source(args, "val"), work / "raw" / "val")
    files = sorted(lut.keys())[: args.diag_n]
    raw_r, long_ax, wgeh = [], [], []
    BATCH = 48
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        res = model.predict(source=[str(lut[f]) for f in chunk],
                            imgsz=args.imgsz, conf=args.pred_conf, verbose=False)
        for rr in res:
            obb = getattr(rr, "obb", None)
            if obb is None or getattr(obb, "xywhr", None) is None or len(obb) == 0:
                continue
            for x in obb.xywhr.cpu().numpy():
                _, _, w, h, rad = (float(v) for v in x)
                raw_r.append(math.degrees(rad) % 180.0)
                long_ax.append((math.degrees(rad) + (0.0 if w >= h else 90.0)) % 180.0)
                wgeh.append(1 if w >= h else 0)
    raw_r, long_ax = np.array(raw_r), np.array(long_ax)
    print("-" * 66)
    print(f"-- 예측 (best.pt={weights.name}, val {len(files)} 이미지, 검출 {len(raw_r):,} 알약) --")
    if len(raw_r):
        print(f"  raw r     : mean={raw_r.mean():.1f}° std={raw_r.std():.1f}° "
              f"min={raw_r.min():.1f} max={raw_r.max():.1f}")
        print(f"  long-axis : mean={long_ax.mean():.1f}° std={long_ax.std():.1f}° "
              f"(w>h 비율={np.mean(wgeh):.2f})")
    else:
        print("  (검출 0 — --weights/이미지 경로 확인)")
    print("=" * 66)
    print("판정:")
    print("  · pred std < 10°  → 각도 붕괴 확정.")
    print("  · GT rotation_label std 넓음(>25°) + near-square 높음  → 원인(a) 컨테인먼트 라벨이 각도 소실")
    print("      → 처방: tight/seg 라벨 재구축 or OBB 각도 폐기하고 회전분류기.")
    print("  · long-axis std >> raw r std        → 방향이 w/h 스왑에 있음 → 원인(c) predict export만 수정.")
    print("  · GT rotation_label std 좁음(<10°)  → 원인(b) 회전라벨이 안 변함 → build 조인/라벨 점검.")


def main():
    args = parse_args()
    work = args.work_dir or (args.data_root / "obb_work")
    work.mkdir(parents=True, exist_ok=True)
    if args.mode in ("verify",):
        cmd_verify(args, work)
    if args.mode in ("build", "all"):
        cmd_build(args, work)
    if args.mode in ("straighten",):
        cmd_straighten(args, work)
    if args.mode in ("predict",):
        cmd_predict(args, work)
    if args.mode in ("angle-diag",):
        cmd_angle_diag(args, work)
    if args.mode in ("train", "all"):
        cmd_train(args, work)


if __name__ == "__main__":
    main()
