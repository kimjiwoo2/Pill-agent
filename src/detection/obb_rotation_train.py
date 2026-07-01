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
  # (verify에서 어느 열이 알약에 딱 붙는지 보고 --convention 0~3 지정)
  # (박스가 알약을 자르면 --box-margin 1.15 처럼 키우면 됨)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# (sign, offset_deg) — 박스 각도 phi = (sign*rotation_label_deg + offset) % 180
CONVENTIONS = [(+1, 0), (+1, 90), (-1, 0), (-1, 90)]


# ----------------------------------------------------------------------------- config
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["verify", "build", "train", "all"], default="build")
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


def main():
    args = parse_args()
    work = args.work_dir or (args.data_root / "obb_work")
    work.mkdir(parents=True, exist_ok=True)
    if args.mode in ("verify",):
        cmd_verify(args, work)
    if args.mode in ("build", "all"):
        cmd_build(args, work)
    if args.mode in ("train", "all"):
        cmd_train(args, work)


if __name__ == "__main__":
    main()
