"""
train_cls.py — 색·모양 분류기 학습 (최종 채택 설정 v3-5)

이 파일은 **학습 재현 전용**이다.
추론은 `final/matching/pill_e2e.py` 의 `Classifier` 가 담당하므로
여기서 만든 산출물을 그쪽이 로드해서 쓴다.

  train_cls.py  ──(학습)──>  best_*.pth + temperature_*.pkl
                                    │
                                    └──> pill_e2e.Classifier (추론)

산출물›
------
  best_20k_v3_5_nosampler_ep16_v3.pth          모델 가중치
  temperature_20k_v3_5_nosampler_ep16_v3.pkl   {'color_T', 'shape_T', ...}
  history_20k_v3_5_nosampler_ep16_v3.csv       에폭별 지표

최종 설정 (v3-5)
---------------
  색 11cls / 모양 4cls · sampler 없음 · 에폭 16 (patience 7)
  AdamW wd 0.1 · backbone lr 1e-5 / head lr 5e-5
  선택 기준: val exp_recall@2

  분류기는 **후보 압축기**이지 식별기가 아니다. DB 4,461종을
  (색,모양) 조합으로 평균 604종까지 좁히는 게 역할이고,
  최종 식별은 OCR 각인 + fusion 이 한다. 그래서 accuracy 가 아니라
  "정답을 후보 안에 남기는" exp_recall@K 로 best 를 고른다.

⚠️ 모델 구조·전처리는 pill_e2e.Classifier 와 반드시 일치해야 한다.
   (letterbox 224/fill=128, ImageNet 정규화, ConvNeXt-Tiny +
    Dropout0.3+Linear 헤드, 출력 키 'color'/'shape')
   한쪽만 바꾸면 체크포인트 로드는 되지만 성능이 조용히 망가진다.

   단 bilateralFilter(d=3,15,15) 는 예외다. crop 생성 시점에 이미
   적용되어 있으므로 학습 transform 에는 없고, 추론은 crop 을 실시간
   생성하므로 pill_e2e 안에서 직접 건다. 결과적으로 모델이 보는
   픽셀은 양쪽 모두 "bilateral 1회 적용" 상태로 동일하다.

   입력 crop 규격: bbox 중심 margin 1.1 정사각 (crop_v7 / CROP_MARGIN)

사용법
------
  python train_cls.py \
      --manifest        manifest_train.csv \
      --crop-dir        /content/20k_cropped_images_v2 \
      --encoders        label_encoders_20k_11cls.pkl \
      --drug-master-csv drug_master.csv \
      --out-dir         ./checkpoints

  # 이미 학습된 체크포인트에 Temperature 만 다시 fit
  python train_cls.py ... --calibrate-only --ckpt ./checkpoints/best_....pth

DB 를 직접 쓰려면 (--drug-master-csv 미지정 시) 환경변수 필요:
  PILLIOT_DB_HOST, PILLIOT_DB_USER, PILLIOT_DB_PW,
  PILLIOT_DB_NAME(기본 pilliot_db), PILLIOT_DB_PORT(기본 3306)
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── 라벨 체계 (v3-5 최종) ────────────────────────────────────────
# 11색 = 아래 10색 + '기타'.
# 청록·회색은 v3-2 에서 분리했다가 하양·갈색 성능이 떨어져 '기타'로 되돌렸다.
COLOR_11 = {'하양', '노랑', '분홍', '갈색', '파랑',
            '주황', '초록', '연두', '빨강', '보라'}
SHAPE_4 = {'원형', '타원형', '장방형'}   # + '기타'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_color_11(v):
    """DB 색 문자열 → 11색. 복합값('청록,투명')은 앞 색만 사용."""
    if pd.isna(v) or str(v).strip() == '':
        return '기타'
    first = str(v).split(',')[0].strip()
    return first if first in COLOR_11 else '기타'


def normalize_shape_4(s):
    """DB 모양 문자열 → 4모양 (원형/타원형/장방형/기타)."""
    s = str(s).strip()
    return s if s in SHAPE_4 else '기타'


# ── 전처리 ──────────────────────────────────────────────────────
class LetterboxResize:
    """종횡비 보존 리사이즈 + 회색(128) 패딩.

    알약은 가로세로 비율 자체가 모양 판별 단서라, 단순 resize 로 찌그러뜨리면
    타원형/장방형 구분이 무너진다.
    ★ pill_e2e.Classifier._Letterbox 와 동일 규격 — 함께 바꿔야 한다.
    """

    def __init__(self, size=224, fill=128):
        self.size, self.fill = size, fill

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = int(w * s), int(h * s)
        img = TF.resize(img, (nh, nw))
        pw, ph = self.size - nw, self.size - nh
        return TF.pad(img, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2),
                      fill=self.fill)


def build_transforms(size=224):
    """(train, val) transform.

    증강 근거:
      회전·상하좌우 flip — 알약은 방향 정보가 없어 강하게 줘도 안전
      ColorJitter      — 색이 예측 대상이므로 약하게. **hue=0 필수**
                         (hue 를 건드리면 색 라벨 자체가 파괴된다)

    ※ bilateralFilter(d=3, sigmaColor=15, sigmaSpace=15) 는 여기 없다.
      crop 생성 단계(20k_cropped_images_v2)에서 이미 적용됐기 때문.
      추론(pill_e2e.Classifier.predict)은 YOLO bbox 로 crop 을 그 자리에서
      만들므로 같은 필터를 직접 건다. 즉 양쪽이 다른 게 정상이고,
      **학습 transform 에 bilateral 을 추가하면 이중 적용이 되어 어긋난다.**
    """
    train_tf = transforms.Compose([
        LetterboxResize(size, fill=128),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.05, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        LetterboxResize(size, fill=128),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


class PillDataset(Dataset):
    """crop 이미지 + (color, shape) 멀티라벨. 파일명 규칙 `{object_id}.png`."""

    def __init__(self, df, crop_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            os.path.join(self.crop_dir, f"{row['object_id']}.png")).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, {
            'shape': torch.tensor(row['shape_id'], dtype=torch.long),
            'color': torch.tensor(row['color_id'], dtype=torch.long),
        }


# ── 모델 ────────────────────────────────────────────────────────
class PillClassifier(nn.Module):
    """ConvNeXt-Tiny 백본 공유 + 태스크별 선형 헤드.

    색과 모양은 같은 시각 특징에서 갈라져 나오므로 백본 공유가 자연스럽다.
    ★ pill_e2e.Classifier 내부 정의와 동일 구조 — 함께 바꿔야 한다.
    """

    def __init__(self, backbone, num_classes, feature_dim=768):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(p=0.3), nn.Linear(feature_dim, n))
            for name, n in num_classes.items()})

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return {name: head(feat) for name, head in self.heads.items()}


def build_model(n_color, n_shape, pretrained=True, device='cpu'):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.convnext_tiny(weights=weights)
    backbone.classifier = nn.Identity()
    return PillClassifier(backbone, {'shape': n_shape, 'color': n_color}).to(device)


# ── VALID_COMBOS (DB 실존 조합) ─────────────────────────────────
def build_valid_combos(color_names, shape_names, drug_master_csv=None):
    """DB(또는 CSV)에 실제로 존재하는 (색idx, 모양idx) 조합.

    exp_recall 은 DB 에 있는 조합 안에서만 순위를 매긴다.
    존재하지 않는 조합(예: DB 에 0종인 보라 장방형)을 후보로 세면
    지표가 부당하게 낮아지기 때문.
    """
    if drug_master_csv:
        db = pd.read_csv(drug_master_csv)[['color_class1', 'drug_shape']].dropna()
        db.columns = ['color', 'shape']
    else:
        import pymysql
        need = ['PILLIOT_DB_HOST', 'PILLIOT_DB_USER', 'PILLIOT_DB_PW']
        missing = [k for k in need if not os.environ.get(k)]
        if missing:
            raise RuntimeError(
                f"환경변수 미설정: {', '.join(missing)}\n"
                "--drug-master-csv 를 쓰면 DB 접속 없이 실행할 수 있습니다.")
        conn = pymysql.connect(
            host=os.environ['PILLIOT_DB_HOST'],
            user=os.environ['PILLIOT_DB_USER'],
            password=os.environ['PILLIOT_DB_PW'],
            database=os.environ.get('PILLIOT_DB_NAME', 'pilliot_db'),
            port=int(os.environ.get('PILLIOT_DB_PORT', 3306)))
        db = pd.read_sql(
            'SELECT DISTINCT color_class1 AS color, drug_shape AS shape '
            'FROM drug_master '
            'WHERE color_class1 IS NOT NULL AND drug_shape IS NOT NULL', conn)
        conn.close()

    db['cn'] = db['color'].apply(normalize_color_11)
    db['sn'] = db['shape'].apply(normalize_shape_4)
    c2i = {c: i for i, c in enumerate(color_names)}
    s2i = {s: i for i, s in enumerate(shape_names)}
    db['ci'] = db['cn'].map(c2i)
    db['si'] = db['sn'].map(s2i)
    db = db.dropna(subset=['ci', 'si'])

    combos = sorted({(int(r.ci), int(r.si)) for r in db.itertuples()})
    return combos, np.array([c for c, _ in combos]), np.array([s for _, s in combos])


def exp_recall_at_k(cp, sp, yc, ys, combos, vc_c, vc_s, ks=(1, 2, 3)):
    """P(색)×P(모양) 조합 랭킹에서 정답 조합이 top-K 에 드는 비율.

    DB 에 없는 정답 조합(라벨 오류 등)은 평가 대상에서 제외한다.
    """
    valid = set(combos)
    hits = {k: 0 for k in ks}
    total = 0
    for n in range(len(yc)):
        tc, ts = yc[n], ys[n]
        if (tc, ts) not in valid:
            continue
        total += 1
        true_score = cp[n][tc] * sp[n][ts]
        rank = (cp[n][vc_c] * sp[n][vc_s] > true_score).sum()
        for k in ks:
            if rank < k:
                hits[k] += 1
    return {k: (hits[k] / total if total else 0.0) for k in ks}, total


# ── 학습 / 평가 ─────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, crit_c, crit_s, log_every=100):
    model.train()
    total_loss = 0.0
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        optimizer.zero_grad()
        out = model(imgs)
        loss = (crit_s(out['shape'], labels['shape'])
                + crit_c(out['color'], labels['color']))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)} | loss {loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, crit_c, crit_s, color_names, shape_names,
             combos, vc_c, vc_s, main_labels,
             watch=('주황', '갈색', '하양', '보라', '빨강')):
    model.eval()
    cp_all, sp_all, yc, ys = [], [], [], []
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        lab = {k: v.to(device) for k, v in labels.items()}
        out = model(imgs)
        total_loss += (crit_s(out['shape'], lab['shape'])
                       + crit_c(out['color'], lab['color'])).item()
        cp_all.append(F.softmax(out['color'], 1).cpu().numpy())
        sp_all.append(F.softmax(out['shape'], 1).cpu().numpy())
        yc.extend(lab['color'].cpu().numpy())
        ys.extend(lab['shape'].cpu().numpy())

    cp, sp = np.concatenate(cp_all), np.concatenate(sp_all)
    yc, ys = np.array(yc), np.array(ys)
    yc_p, ys_p = cp.argmax(1), sp.argmax(1)
    n = len(yc)

    recall, n_eval = exp_recall_at_k(cp, sp, yc, ys, combos, vc_c, vc_s)
    full = list(range(len(color_names)))
    rep = classification_report(yc, yc_p, labels=full, target_names=color_names,
                               zero_division=0, output_dict=True)

    m = {
        'val_loss': total_loss / len(loader),
        'exp_recall_1': recall[1], 'exp_recall_2': recall[2],
        'exp_recall_3': recall[3],
        'hard_joint': ((yc_p == yc) & (ys_p == ys)).mean(),
        'soft_joint': (cp[np.arange(n), yc] * sp[np.arange(n), ys]).mean(),
        'f1_color': f1_score(yc, yc_p, labels=main_labels, average='macro',
                             zero_division=0),
        'f1_shape': f1_score(ys, ys_p, average='macro', zero_division=0),
        'color_acc': (yc_p == yc).mean(),
        'shape_acc': (ys_p == ys).mean(),
    }
    for c in color_names:
        m[f'{c}_f1'] = rep[c]['f1-score']
        m[f'{c}_recall'] = rep[c]['recall']

    print(f"  ★ exp_recall@2 {recall[2]:.4f} "
          f"(@1 {recall[1]:.4f} / @3 {recall[3]:.4f})")
    print(f"     color_acc {m['color_acc']:.4f} | shape_acc {m['shape_acc']:.4f} | "
          f"f1_main {m['f1_color']:.4f} | 평가대상 {n_eval}/{n}")
    mon = ' | '.join(f"{c} {rep[c]['f1-score']:.2f}(r{rep[c]['recall']:.2f})"
                     for c in watch if c in rep)
    print(f"     [모니터링] {mon}")
    return m


# ── Temperature Scaling ─────────────────────────────────────────
@torch.no_grad()
def collect_logits(model, loader, device):
    """softmax 이전 raw logit 수집 (TS 는 logit 단계에서 보정)."""
    model.eval()
    cl, sl, cy, sy = [], [], [], []
    for imgs, labels in loader:
        out = model(imgs.to(device))
        cl.append(out['color'].cpu())
        sl.append(out['shape'].cpu())
        cy.append(labels['color'])
        sy.append(labels['shape'])
    return torch.cat(cl), torch.cat(sl), torch.cat(cy), torch.cat(sy)


def fit_temperature(logits, labels, name, init=1.5, lr=0.01, max_iter=50):
    """val NLL 을 최소화하는 스칼라 T 하나를 LBFGS 로 학습.

    모델 가중치는 건드리지 않는 post-hoc 보정.
    T>1 이면 과확신이었다는 뜻(분포를 부드럽게 만든다).
    """
    logits, labels = logits.detach().clone(), labels.detach().clone()
    T = nn.Parameter(torch.ones(1) * init)
    opt = optim.LBFGS([T], lr=lr, max_iter=max_iter)
    before = F.cross_entropy(logits, labels).item()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    Tv = T.item()
    after = F.cross_entropy(logits / Tv, labels).item()
    print(f"  [{name}] T={Tv:.4f} | NLL {before:.4f} → {after:.4f} "
          f"({'개선' if after < before else '악화'})")
    return Tv


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error. confidence 구간별 |평균확신 - 실제정확도| 가중합."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return ece


def run_calibration(model, val_loader, device, out_dir, exp_name,
                    combos, vc_c, vc_s, best_epoch=None):
    """헤드별 T fit → ECE 검증 → exp_recall 불변 확인 → pkl 저장."""
    print('\n=== Temperature Scaling ===')
    cl, sl, cy, sy = collect_logits(model, val_loader, device)
    print(f"val logit: color {tuple(cl.shape)} | shape {tuple(sl.shape)}")

    color_T = fit_temperature(cl, cy, 'color')
    shape_T = fit_temperature(sl, sy, 'shape')

    cp_b, cp_a = F.softmax(cl, 1).numpy(), F.softmax(cl / color_T, 1).numpy()
    sp_b, sp_a = F.softmax(sl, 1).numpy(), F.softmax(sl / shape_T, 1).numpy()
    yc, ys = cy.numpy(), sy.numpy()

    print('\nECE (낮을수록 보정 잘 됨):')
    print(f"  color {compute_ece(cp_b, yc):.4f} → {compute_ece(cp_a, yc):.4f}")
    print(f"  shape {compute_ece(sp_b, ys):.4f} → {compute_ece(sp_a, ys):.4f}")

    # 불변식: T 는 argmax 순위를 바꾸지 않으므로 recall 이 같아야 정상
    r_b, _ = exp_recall_at_k(cp_b, sp_b, yc, ys, combos, vc_c, vc_s)
    r_a, _ = exp_recall_at_k(cp_a, sp_a, yc, ys, combos, vc_c, vc_s)
    print(f"\nexp_recall@2 before {r_b[2]:.4f} / after {r_a[2]:.4f}")
    print('  ✓ 불변 확인 (정상)' if abs(r_b[2] - r_a[2]) < 1e-6
          else '  ⚠️ TS 후 recall 이 변함 — 구현 버그 가능성')

    ts_path = os.path.join(out_dir, f'temperature_{exp_name}.pkl')
    payload = {
        'exp_name': exp_name, 'best_epoch': best_epoch,
        'color_T': float(color_T), 'shape_T': float(shape_T),
        'ece_color_before': float(compute_ece(cp_b, yc)),
        'ece_color_after': float(compute_ece(cp_a, yc)),
        'ece_shape_before': float(compute_ece(sp_b, ys)),
        'ece_shape_after': float(compute_ece(sp_a, ys)),
        'exp_recall_1': float(r_a[1]), 'exp_recall_2': float(r_a[2]),
        'exp_recall_3': float(r_a[3]),
    }
    with open(ts_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n저장: {ts_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ── main ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='PILLAR 색·모양 분류기 학습 (v3-5)')
    p.add_argument('--manifest', required=True, help='학습용 manifest CSV')
    p.add_argument('--crop-dir', required=True, help='crop PNG 디렉터리')
    p.add_argument('--encoders', required=True, help='label_encoders_20k_11cls.pkl')
    p.add_argument('--out-dir', required=True, help='체크포인트 저장 경로')
    p.add_argument('--drug-master-csv', default=None,
                   help='지정 시 DB 접속 없이 CSV 로 VALID_COMBOS 구축 (권장)')
    p.add_argument('--exp-name', default='20k_v3_5_nosampler_ep16_v3')
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--color-col', default='color_class1_original')
    p.add_argument('--shape-col', default='shape_group_unified')
    p.add_argument('--calibrate-only', action='store_true',
                   help='학습 건너뛰고 Temperature 만 fit')
    p.add_argument('--ckpt', default=None, help='--calibrate-only 시 로드할 .pth')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # 인코더 = source of truth. 절대 재fit 금지.
    # 재fit 하면 클래스 인덱스 순서가 바뀌어 기존 체크포인트가 무의미해진다.
    with open(args.encoders, 'rb') as f:
        enc = pickle.load(f)
    le_color, le_shape = enc['color'], enc['shape']
    color_names = [str(c) for c in le_color.classes_]
    shape_names = [str(s) for s in le_shape.classes_]
    print(f"색 {len(color_names)}cls: {color_names}")
    print(f"모양 {len(shape_names)}cls: {shape_names}")

    idx_etc = color_names.index('기타')
    main_labels = [i for i in range(len(color_names)) if i != idx_etc]

    # manifest → color_id / shape_id
    df = pd.read_csv(args.manifest, low_memory=False)
    df['color_group_v11'] = df[args.color_col].apply(normalize_color_11)
    df['color_id'] = le_color.transform(df['color_group_v11'])
    df['shape_id'] = le_shape.transform(df[args.shape_col])

    # crop 파일이 실제로 있는 행만 (방어)
    saved = set(os.listdir(args.crop_dir))
    def _filter(sub):
        return sub[sub['object_id'].astype(str).add('.png').isin(saved)
                   ].reset_index(drop=True)
    df_train, df_val = _filter(df[df['split'] == 'train']), _filter(df[df['split'] == 'val'])
    print(f"train {len(df_train)} | val {len(df_val)}")

    train_tf, val_tf = build_transforms()
    train_loader = DataLoader(
        PillDataset(df_train, args.crop_dir, train_tf),
        batch_size=args.batch_size, shuffle=True,      # v3-5: sampler 없음
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        PillDataset(df_val, args.crop_dir, val_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    combos, vc_c, vc_s = build_valid_combos(
        color_names, shape_names, args.drug_master_csv)
    print(f"VALID_COMBOS: {len(combos)}개")

    model = build_model(len(color_names), len(shape_names),
                        pretrained=not args.calibrate_only, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f'best_{args.exp_name}.pth')
    best_epoch = None

    if args.calibrate_only:
        ckpt = args.ckpt or ckpt_path
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"체크포인트 없음: {ckpt}\n--ckpt 로 경로를 지정하세요.")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"체크포인트 로드: {ckpt}")
    else:
        # class weight balanced — sampler 를 안 쓰는 대신 loss 로만 불균형 보정.
        # train 에 한 장도 없는 색이 있으면 compute_class_weight 가 에러를 내므로
        # 존재하는 클래스만 계산하고 나머지는 1.0.
        y_color = df_train['color_id'].values
        present = np.unique(y_color)
        cw = np.ones(len(color_names))
        cw[present] = compute_class_weight('balanced', classes=present, y=y_color)
        absent = [color_names[i] for i in range(len(color_names)) if i not in present]
        if absent:
            print(f"  ⚠️ train 에 없는 색 (weight 1.0): {absent}")

        crit_c = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float).to(device))
        crit_s = nn.CrossEntropyLoss()

        # backbone 은 낮은 lr(사전학습 특징 보존), head 는 높은 lr(빠른 적응)
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': args.backbone_lr},
            {'params': model.heads.parameters(), 'lr': args.head_lr},
        ], weight_decay=args.weight_decay)

        # -1.0 시작 — best 가 0.0 인 퇴화 케이스에서도 첫 에폭이 저장되도록
        # (0.0 이면 `0.0 > 0.0` 이 False 라 체크포인트가 하나도 안 생긴다)
        best_recall, best_epoch, no_improve = -1.0, 0, 0
        history = []

        for epoch in range(args.epochs):
            print(f"\n[Epoch {epoch+1:02d}/{args.epochs}]")
            tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                      crit_c, crit_s)
            m = evaluate(model, val_loader, device, crit_c, crit_s,
                         color_names, shape_names, combos, vc_c, vc_s, main_labels)
            print(f"  train loss {tr_loss:.4f} | val loss {m['val_loss']:.4f}")
            m['epoch'] = epoch + 1
            m['train_loss'] = tr_loss
            history.append(m)

            if m['exp_recall_2'] > best_recall:
                best_recall, best_epoch, no_improve = m['exp_recall_2'], epoch + 1, 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ★ best saved (Ep{best_epoch}, exp_recall@2 {best_recall:.4f})")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  early stop ({args.patience} epochs no improve)")
                    break

        hist_path = os.path.join(args.out_dir, f'history_{args.exp_name}.csv')
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"\nbest exp_recall@2 {best_recall:.4f} @ Ep{best_epoch}")
        print(f"체크포인트: {ckpt_path}")
        print(f"히스토리:   {hist_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    run_calibration(model, val_loader, device, args.out_dir, args.exp_name,
                    combos, vc_c, vc_s, best_epoch)


if __name__ == '__main__':
    main()