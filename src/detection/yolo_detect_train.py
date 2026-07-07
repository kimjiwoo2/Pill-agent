"""
Pill 1-class YOLO 검출 (축정렬 bbox) — OBB 없이 '원래' 방식.

manifest의 축정렬 bbox(bbox_x/y/w/h)로 표준 YOLO detect 라벨을 만들어 yolo11n 학습.
회전각·OBB 안 씀 → 위치(검출)만. rotation-quality 필터도 없음(전체 알약 사용).

입력:
  - manifest : manifest_clean_20k_33340.csv (object_id, image_file, bbox_x/y/w/h, width, height, split)
  - 이미지   : images_train.zip / images_val.zip (또는 이미 푼 폴더). image_file = 파일명.

사용:
  # zip에서 (없으면 work/raw 로 추출)
  python yolo_detect_train.py --mode all --data-root /mnt/data/jh/data --name yolo11n_detect_v1

  # ★ 이미 obb_work/raw 에 푼 이미지 재사용(31G 중복 추출 방지) — 삭제 전 학습 권장
  python yolo_detect_train.py --mode all --data-root /mnt/data/jh/data \
      --img-train /mnt/data/jh/data/obb_work/raw/train \
      --img-val   /mnt/data/jh/data/obb_work/raw/val \
      --name yolo11n_detect_v1

라벨 포맷: YOLO detect  "0 cx cy w h"  (모두 [0,1] 정규화, class 0 = pill).
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- config
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pill 1-class YOLO detect (axis-aligned) 학습")
    p.add_argument("--mode", choices=["build", "train", "all"], default="all")
    p.add_argument("--data-root", type=Path, required=True, help="manifest/이미지 있는 디렉토리")
    p.add_argument("--manifest", default="manifest_clean_20k_33340.csv")
    p.add_argument("--img-train", default="images_train.zip", help="zip 또는 이미 푼 폴더")
    p.add_argument("--img-val", default="images_val.zip")
    p.add_argument("--work-dir", type=Path, default=None, help="산출물 위치(기본 data-root/detect_work)")
    p.add_argument("--model", default="yolo11n.pt", help="detect 모델(OBB 아님)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="yolo11n_detect_v1")
    p.add_argument("--viz-samples", type=int, default=12)
    return p.parse_args()


# --------------------------------------------------------------------------- data / io
def load_manifest(args: argparse.Namespace, split: str) -> pd.DataFrame:
    df = pd.read_csv(args.data_root / args.manifest, low_memory=False)
    df["object_id"] = df["object_id"].astype(str)
    df = df[df["split"] == split]
    need = ["image_file", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "width", "height"]
    df = df.dropna(subset=need).copy()
    valid = (
        (df["bbox_w"] > 0) & (df["bbox_h"] > 0)
        & (df["bbox_x"] >= 0) & (df["bbox_y"] >= 0)
        & (df["bbox_x"] + df["bbox_w"] <= df["width"])
        & (df["bbox_y"] + df["bbox_h"] <= df["height"])
    )
    return df[valid].reset_index(drop=True)


def img_source(args: argparse.Namespace, split: str) -> Path:
    return args.data_root / (args.img_train if split == "train" else args.img_val)


def ensure_images(src: Path, out_dir: Path) -> dict:
    """zip이면 out_dir에 (없을 때만) 풀고, 폴더면 그대로 사용. {파일명: 경로} 반환."""
    if src.is_dir():
        root = src
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not any(out_dir.iterdir()):
            print(f"  압축 해제: {src.name} → {out_dir}")
            with zipfile.ZipFile(src) as zf:
                zf.extractall(out_dir)
        root = out_dir
    lut = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(ext):
            lut[p.name] = p
    print(f"  이미지 {len(lut):,}장 인덱싱: {root}")
    return lut


def yolo_label_line(row) -> str:
    """축정렬 bbox → YOLO detect 라벨. bbox_x/y=좌상단 → 중심으로 변환 후 정규화."""
    W, H = float(row["width"]), float(row["height"])
    cx = (float(row["bbox_x"]) + float(row["bbox_w"]) / 2.0) / W
    cy = (float(row["bbox_y"]) + float(row["bbox_h"]) / 2.0) / H
    w = float(row["bbox_w"]) / W
    h = float(row["bbox_h"]) / H
    # 클램프(수치 오차로 1.0 초과 방지)
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(w, 1.0), min(h, 1.0)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ------------------------------------------------------------------------- sanity viz
def viz_build_check(work: Path, split: str, n: int = 12) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img_dir = work / "dataset" / "images" / split
    lbl_dir = work / "dataset" / "labels" / split
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not imgs:
        return
    idx = np.unique(np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int))
    pick = [imgs[i] for i in idx]
    cols = 4
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax, p in zip(axes, pick):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        lbl = lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                _, cx, cy, w, h = (float(v) for v in line.split())
                x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(pick):]:
        ax.axis("off")
    plt.suptitle(f"detect build check — {split} (축정렬 bbox)", fontsize=11)
    plt.tight_layout()
    out = work / f"detect_build_check_{split}.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"  [viz] 저장: {out}")


# ---------------------------------------------------------------------------- commands
def cmd_build(args: argparse.Namespace, work: Path) -> None:
    print(f"[build] detect 라벨 생성 (manifest={args.manifest})")
    yaml_lines = [f"path: {work / 'dataset'}", "train: images/train", "val: images/val",
                  "nc: 1", "names: [pill]"]
    (work / "dataset").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        df = load_manifest(args, split)
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
            lines = [yolo_label_line(r) for _, r in grp.iterrows()]
            (lbl_dir / f"{Path(fname).stem}.txt").write_text("\n".join(lines))
            link = img_dir / fname
            if not link.exists():
                os.symlink(src.resolve(), link)
            n_img += 1
            n_obj += len(lines)
        print(f"  [{split}] 이미지 {n_img:,} / 알약 {n_obj:,}  (원본 누락 {miss})")

    (work / "dataset" / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"[build] dataset.yaml → {work / 'dataset' / 'dataset.yaml'}")
    for sp in ["train", "val"]:
        try:
            viz_build_check(work, sp, args.viz_samples)
        except Exception as e:
            print(f"  [viz] build check {sp} 실패(무시): {e}")


def cmd_train(args: argparse.Namespace, work: Path) -> None:
    from ultralytics import YOLO
    data_yaml = work / "dataset" / "dataset.yaml"
    assert data_yaml.exists(), "먼저 --mode build 실행"
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers,
        optimizer="AdamW", lr0=0.002, patience=15,
        project=str(work / "runs"), name=args.name,
        # 1-class 검출은 방향 라벨이 없으므로 좌우/상하 반전 증강 허용(ultralytics 기본값 사용)
    )
    metrics = model.val()
    print(f"[train] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


def main() -> None:
    args = parse_args()
    work = args.work_dir or (args.data_root / "detect_work")
    work.mkdir(parents=True, exist_ok=True)
    if args.mode in ("build", "all"):
        cmd_build(args, work)
    if args.mode in ("train", "all"):
        cmd_train(args, work)


if __name__ == "__main__":
    main()
