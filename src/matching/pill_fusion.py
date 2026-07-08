"""
Pill 매칭 fusion — retrieval(색·형태 압축) + rerank(각인 CER) — val/test 공용 .py.

05_jw_cv_pipeline 노트북의 fusion 을 .py 로 이식 + 업그레이드.
  구식 재현:  --legacy       → 노트북 그대로 (α·g·(−CER) + β·(logP_c+logP_s))
  업그레이드: 기본 활성 옵션
    · --dual180   : 각인 매칭에 180° 뒤집힘 dual-match (min CER) — 회전 미보정 흡수
    · --cs-gate   : 색·형태 항에 (1−g(conf)) 곱 — OCR 강할 때 색·형태 죽여 각인 지배
    · --fit logreg: α·β·(w_c·w_s) 를 val 라벨로 로지스틱회귀 학습(grid 대체, 초~분)

입력 (경로는 전부 인자):
  --drug-master-csv : object 아님, drug_master 덤프 CSV (item_seq, color_class1, drug_shape, print_front/back)
                      (없으면 --db 로 MySQL 직접)
  --encoders        : label_encoders_20k_11cls.pkl
  --ckpt / --ts     : 분류기 .pth / temperature .pkl
  --labels-csv      : object_id, item_seq (, split) — 평가 정답 (val=manifest_clean, test=팀원 매칭본)
  --crops           : crop 폴더 또는 zip (분류기 추론용)
  --ocr-csv         : object_id, ocr_text, ocr_conf — OCR 결과 (pretrained 로 통일 권장)

사용:
  python pill_fusion.py --split val \
      --drug-master-csv drug_master.csv --encoders label_encoders_20k_11cls.pkl \
      --ckpt best_20k_v3_5_nosampler_ep16_v3.pth --ts temperature_...v3.pkl \
      --labels-csv manifest_clean_20k_33340.csv --crops crop_v7_val.zip \
      --ocr-csv pretrained_v9_val_result.csv --report
  # test: --split test --labels-csv test_matched.csv --crops test_filtered.zip --ocr-csv test_ocr.csv
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================ 각인 정규화 · CER (윤수 이식)
IGNORE_IMPRINT_TOKENS = {'', 'NAN', 'NONE', 'NULL', '마크', '분할선', '없음', '무', '-', '십자'}
_RE_STRIP_TOKENS = re.compile(r'\s+|분할선|마크|\|')
_RE_ALLOWED = re.compile(r'[^0-9A-Z가-힣+\-/]')


def normalize_imprint(text):
    if pd.isna(text):
        return ''
    text = str(text).strip().upper()
    if text in IGNORE_IMPRINT_TOKENS:
        return ''
    text = _RE_STRIP_TOKENS.sub('', text)
    text = _RE_ALLOWED.sub('', text)
    return '' if text in IGNORE_IMPRINT_TOKENS else text


def levenshtein(pred, target):
    p, t = list(pred), list(target)
    dp = list(range(len(t) + 1))
    for pc in p:
        ndp = [dp[0] + 1]
        for j, tc in enumerate(t):
            ndp.append(min(dp[j] + (pc != tc), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[len(t)]


# 180° 회전 시 같은 모양으로 보이는 글자 매핑(대문자+숫자). 없는 글자가 있으면 회전 불가.
_ROT180 = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6',
           'H': 'H', 'I': 'I', 'N': 'N', 'O': 'O', 'S': 'S', 'X': 'X', 'Z': 'Z',
           'M': 'W', 'W': 'M', '+': '+', '-': '-'}


def rot180(s):
    """문자열을 180° 회전한 형태(글자 flip + 순서 역순). 매핑 없는 글자 포함이면 None."""
    out = []
    for ch in reversed(s):
        if ch not in _ROT180:
            return None
        out.append(_ROT180[ch])
    return ''.join(out)


def best_cer(pred, cand, dual180=False):
    """pred(정규화됨) vs 후보 각인 cand → 최소 CER (0~1). dual180이면 180° 뒤집힌 pred도 비교."""
    denom = max(len(cand), 1)
    c = levenshtein(pred, cand) / denom
    if dual180:
        pr = rot180(pred)
        if pr is not None:
            c = min(c, levenshtein(pr, cand) / denom)
    return c


def score_one(pred, candidates, dual180=False):
    """쿼리 각인 vs 후보 각인들 → 최고 매칭점수(1−CER). 후보 유효각인 없으면 None."""
    pred = normalize_imprint(pred)
    valid = [c for c in candidates if c]
    if not valid:
        return None
    return max(0.0, 1.0 - min(best_cer(pred, c, dual180) for c in valid))


# ============================================================ 게이트 g(conf)
GATE_DEFAULTS = {
    'hard': {'thr': 0.8},
    'sigmoid': {'c0': 0.6, 'k': 12.0},
    'linear': {'lo': 0.3, 'hi': 0.7},
}


def gate(conf, mode='hard', params=None):
    p = {**GATE_DEFAULTS[mode], **(params or {})}
    if mode == 'hard':
        return 1.0 if conf >= p['thr'] else 0.0
    if mode == 'sigmoid':
        return float(1.0 / (1.0 + np.exp(-p['k'] * (conf - p['c0']))))
    if mode == 'linear':
        return float(np.clip((conf - p['lo']) / max(p['hi'] - p['lo'], 1e-9), 0.0, 1.0))
    raise ValueError(mode)


# ============================================================ DB / 후보 캐시
def load_drug_master(args):
    """drug_master → (dm_valid, combo_to_items, COLOR_CLASSES, SHAPE_CLASSES). CSV 우선, 없으면 MySQL."""
    with open(args.encoders, 'rb') as f:
        enc = pickle.load(f)
    COLOR_CLASSES = list(enc['color'].classes_)
    SHAPE_CLASSES = list(enc['shape'].classes_)
    COLOR_SET = {c for c in COLOR_CLASSES if c != '기타'}
    SHAPE_SET = {s for s in SHAPE_CLASSES if s != '기타'}

    if args.drug_master_csv and os.path.exists(args.drug_master_csv):
        dm = pd.read_csv(args.drug_master_csv, low_memory=False)
    else:
        from sqlalchemy import create_engine
        eng = create_engine(f"mysql+pymysql://{args.db_user}:{args.db_pw}@{args.db_host}/{args.db_name}?charset=utf8mb4")
        dm = pd.read_sql("SELECT item_seq, color_class1, drug_shape, print_front, print_back FROM drug_master", eng)

    def norm_color(v):
        if pd.isna(v) or str(v).strip() == '':
            return None
        first = str(v).split(',')[0].strip()
        return first if first in COLOR_SET else '기타'

    def norm_shape(v):
        if pd.isna(v) or str(v).strip() == '':
            return None
        s = str(v).strip()
        return s if s in SHAPE_SET else '기타'

    dm['color_norm'] = dm['color_class1'].apply(norm_color)
    dm['shape_norm'] = dm['drug_shape'].apply(norm_shape)
    dm_valid = dm[dm['color_norm'].notna() & dm['shape_norm'].notna()].copy()

    combo_to_items = (dm_valid.groupby(['color_norm', 'shape_norm'])['item_seq']
                      .apply(set).to_dict())

    color_idx = {str(c): i for i, c in enumerate(COLOR_CLASSES)}
    shape_idx = {str(s): i for i, s in enumerate(SHAPE_CLASSES)}
    cand_info = {}
    for _, r in dm_valid.iterrows():
        eng_list = [t for t in (normalize_imprint(r['print_front']),
                                normalize_imprint(r['print_back'])) if t]
        cand_info[r['item_seq']] = {'ci': color_idx.get(str(r['color_norm'])),
                                    'si': shape_idx.get(str(r['shape_norm'])),
                                    'eng': eng_list}
    n_noeng = sum(1 for v in cand_info.values() if not v['eng'])
    print(f"[db] drug_master {len(dm)} → 유효 {len(dm_valid)}종 · 조합 {len(combo_to_items)} · 비각인 {n_noeng}종")
    return dict(dm_valid=dm_valid, combo_to_items=combo_to_items, cand_info=cand_info,
                COLOR_CLASSES=COLOR_CLASSES, SHAPE_CLASSES=SHAPE_CLASSES)


# ============================================================ 분류기 추론
def infer_probs(args, object_ids):
    """crop 폴더/zip → 각 object_id 의 TS 보정 색·형태 확률 (순서=object_ids)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    import torchvision.transforms.functional as TF
    from PIL import Image

    with open(args.encoders, 'rb') as f:
        enc = pickle.load(f)
    n_color, n_shape = len(enc['color'].classes_), len(enc['shape'].classes_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    backbone = models.convnext_tiny(weights=None)
    backbone.classifier = nn.Identity()

    class PillClassifier(nn.Module):
        def __init__(self, backbone, num_classes, feature_dim=768):
            super().__init__()
            self.backbone = backbone
            self.heads = nn.ModuleDict({
                name: nn.Sequential(nn.Dropout(p=0.3), nn.Linear(feature_dim, n))
                for name, n in num_classes.items()})

        def forward(self, x):
            feat = self.backbone(x).flatten(1)
            return {name: head(feat) for name, head in self.heads.items()}

    model = PillClassifier(backbone, {'shape': n_shape, 'color': n_color}).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    T_color = T_shape = 1.0
    if args.ts and os.path.exists(args.ts):
        with open(args.ts, 'rb') as f:
            ts = pickle.load(f)
        T_color, T_shape = ts.get('color_T', 1.0), ts.get('shape_T', 1.0)
    print(f"[cls] {Path(args.ckpt).name} · TS color={T_color:.3f}/shape={T_shape:.3f} · {device}")

    class Letterbox:
        def __init__(self, size=224, fill=128): self.size, self.fill = size, fill
        def __call__(self, img):
            w, h = img.size
            s = self.size / max(w, h)
            nw, nh = int(w * s), int(h * s)
            img = TF.resize(img, (nh, nw))
            pw, ph = self.size - nw, self.size - nh
            return TF.pad(img, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2), fill=self.fill)

    tf = transforms.Compose([Letterbox(224, 128), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    crop_lut = _ensure_crops(args.crops)
    pcs, pss, miss = [], [], 0
    batch, ids_ok = [], []

    @torch.no_grad()
    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        out = model(x)
        pc = F.softmax(out['color'] / T_color, 1).cpu().numpy()
        ps = F.softmax(out['shape'] / T_shape, 1).cpu().numpy()
        pcs.append(pc); pss.append(ps)
        batch.clear()

    p_by_id = {}
    for oid in object_ids:
        fp = crop_lut.get(oid)
        if fp is None:
            miss += 1
            continue
        batch.append(tf(Image.open(fp).convert('RGB')))
        ids_ok.append(oid)
        if len(batch) >= 256:
            flush()
    flush()
    all_pc = np.concatenate(pcs) if pcs else np.zeros((0, n_color))
    all_ps = np.concatenate(pss) if pss else np.zeros((0, n_shape))
    for i, oid in enumerate(ids_ok):
        p_by_id[oid] = (all_pc[i], all_ps[i])
    print(f"[cls] 추론 {len(ids_ok)} · crop 없음 {miss}")
    return p_by_id


def _ensure_crops(crops):
    """crop 폴더/zip → {object_id: png경로} LUT.
    zip은 zip 이름별 폴더에 풂 (val/test 같은 폴더 재사용으로 인한 crop 오염 방지).
    중첩 폴더(zip 안 crops_test/ 등)도 rglob 으로 흡수."""
    p = Path(crops)
    if p.is_dir():
        root = p
    else:
        base = Path('/content/pill_crops') if os.path.exists('/content') else Path('/tmp/pill_crops')
        root = base / p.stem                      # zip별 격리: pill_crops/crop_v7_val, .../test_filtered
        root.mkdir(parents=True, exist_ok=True)
        if not any(root.rglob('*.png')):
            print(f"[crops] 압축 해제: {p.name} → {root}")
            with zipfile.ZipFile(p) as zf:
                zf.extractall(root)
    lut = {q.stem: q for q in root.rglob('*.png')}
    print(f"[crops] {len(lut):,}장 인덱싱: {root}")
    return lut


# ============================================================ 압축 (retrieval)
def compress_candidates(pc, ps, combo_to_items, COLOR_CLASSES, SHAPE_CLASSES, topk_combo=3):
    joint = [(pc[ci] * ps[si], COLOR_CLASSES[ci], SHAPE_CLASSES[si])
             for ci in range(len(pc)) for si in range(len(ps))]
    joint.sort(reverse=True)
    cand, used = set(), 0
    for _, cname, sname in joint:
        if (cname, sname) in combo_to_items:
            cand |= combo_to_items[(cname, sname)]
            used += 1
            if used >= topk_combo:
                break
    return cand


# ============================================================ fusion (match_fused 업그레이드)
def match_fused(query_faces, conf, cand_seqs, pc, ps, cand_info,
                alpha=2.5, beta=0.1, eps=1e-3, k=3,
                use_engrave=True, use_gating=True, use_neutralize=True,
                gate_mode='hard', gate_params=None,
                dual180=True, cs_gate=False, return_scores=False):
    """후보별 naive-Bayes 로그우도 합 → top-k item_seq.
      score(d) = use_engrave·α·g·(−CER)  +  cs_w·β·(logP_color + logP_shape)
      cs_gate=True → cs_w=(1−g)  (OCR 강하면 색·형태 죽임). False → cs_w=1 (구식).
    """
    g = gate(conf, gate_mode, gate_params) if use_gating else 1.0
    cs_w = (1.0 - g) if cs_gate else 1.0
    query_has_engrave = any(query_faces)

    scored = []
    for seq in cand_seqs:
        info = cand_info.get(seq)
        if info is None:
            continue
        ci, si, eng = info['ci'], info['si'], info['eng']

        eng_term = 0.0
        if use_engrave:
            if eng:
                s = max((score_one(q, eng, dual180) for q in query_faces), default=0.0)
                eng_term = alpha * g * (s - 1.0)
            else:  # 후보 비각인
                if use_neutralize:
                    eng_term = alpha * g * (-1.0) if query_has_engrave else 0.0
                else:
                    eng_term = alpha * g * (-1.0)

        lpc = np.log(pc[ci] + eps) if ci is not None else np.log(eps)
        lps = np.log(ps[si] + eps) if si is not None else np.log(eps)
        cs_term = cs_w * beta * (lpc + lps)

        tb = (pc[ci] if ci is not None else 0.0) + (ps[si] if si is not None else 0.0)
        scored.append((eng_term + cs_term, tb, seq))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if return_scores:
        return [(seq, sc) for sc, tb, seq in scored]
    return [seq for sc, tb, seq in scored[:k]]


def rank_with_ties(scored, k=3):
    """(seq,score) 정렬 → 공동순위 top-k. 진짜 동점을 임의정렬로 안 가름."""
    out, rank, prev, distinct = [], 0, None, 0
    for idx, (seq, sc) in enumerate(scored):
        if prev is None or sc != prev:
            distinct += 1
            if distinct > k:
                break
            rank, prev = idx + 1, sc
        out.append((rank, seq, sc))
    return out


def in_topk_ties(scored, gt_seq, k=3):
    """공동순위 채점: 정답이 top-k 공동순위 안에 있으면 True (동점을 운으로 안 가름)."""
    return any(seq == gt_seq for _, seq, _ in rank_with_ties(scored, k))


# ============================================================ 평가
def evaluate(labels, faces, confs, p_by_id, db, params, ks=(1, 2, 3)):
    """coverage@k, recall@k(공동순위), reach@3 — 전체 + seen/unseen + 각인유무 슬라이스."""
    from collections import defaultdict
    ci_combo = db['combo_to_items']; cand_info = db['cand_info']
    CC, SC = db['COLOR_CLASSES'], db['SHAPE_CLASSES']
    topk = params['topk_combo']
    covkey = 'coverage@%d' % topk

    def new_acc():
        return {'n': 0, 'cov': 0, 'read': 0, 'rec': {k: 0 for k in ks}}
    sl = defaultdict(new_acc)
    per_color_tot, per_color_cov = {}, {}
    in_pool = miss_crop = 0

    def bump(acc, hit_cov, has_read, rec_hits):
        acc['n'] += 1; acc['cov'] += int(hit_cov); acc['read'] += int(has_read)
        for k in ks:
            acc['rec'][k] += rec_hits[k]

    for r in labels:
        oid, gt = r['oid'], r['seq']
        if oid not in p_by_id:
            miss_crop += 1
            continue
        pc, ps = p_by_id[oid]
        in_pool += int(gt in cand_info)
        cand = compress_candidates(pc, ps, ci_combo, CC, SC, topk)
        hit_cov = gt in cand
        qf = faces.get(oid, []); cf = confs.get(oid, 0.0)
        scored = match_fused(qf, cf, cand, pc, ps, cand_info,
                             alpha=params['alpha'], beta=params['beta'], eps=params['eps'],
                             gate_mode=params['gate_mode'], use_gating=params['use_gating'],
                             use_neutralize=params['use_neutralize'], use_engrave=params['use_engrave'],
                             dual180=params['dual180'], cs_gate=params['cs_gate'], return_scores=True)
        rec_hits = {k: int(in_topk_ties(scored, gt, k)) for k in ks}

        bump(sl['all'], hit_cov, bool(qf), rec_hits)
        if r.get('seen') is True:
            bump(sl['seen'], hit_cov, bool(qf), rec_hits)
        elif r.get('seen') is False:
            bump(sl['unseen'], hit_cov, bool(qf), rec_hits)
        if r.get('printed') is True:
            bump(sl['printed'], hit_cov, bool(qf), rec_hits)
        elif r.get('printed') is False:
            bump(sl['no_print'], hit_cov, bool(qf), rec_hits)

        per_color_tot[r['color']] = per_color_tot.get(r['color'], 0) + 1
        per_color_cov[r['color']] = per_color_cov.get(r['color'], 0) + int(hit_cov)

    def summ(acc):
        n = max(acc['n'], 1)
        o = {'n': acc['n'], covkey: acc['cov'] / n, 'read_rate': acc['read'] / n}
        for k in ks:
            o['recall@%d' % k] = acc['rec'][k] / n
        o['reach@3'] = o['recall@3'] / max(o[covkey], 1e-9)
        return o

    out = {s: summ(a) for s, a in sl.items()}
    out['_per_color'] = {c: per_color_cov[c] / per_color_tot[c] for c in per_color_tot}
    out['_pool'] = {'in_pool': in_pool, 'evaluated': sl['all']['n'], 'miss_crop': miss_crop}
    return out


# ============================================================ 라벨/OCR 로드
def _as_bool(v):
    if pd.isna(v):
        return None
    return str(v).strip().lower() in ('1', '1.0', 'true', 't', 'yes', 'y')


def load_labels(args):
    df = pd.read_csv(args.labels_csv, low_memory=False)
    df['object_id'] = df['object_id'].astype(str)
    if 'split' in df.columns and args.split in set(df['split'].astype(str)):
        df = df[df['split'] == args.split]
    assert 'item_seq' in df.columns, f"labels-csv 에 item_seq 없음 ({args.labels_csv})"
    color_col = next((c for c in ['color_group_v11', 'color_group_normalized',
                                  'color_class1_original', 'color_class1'] if c in df.columns), None)
    seen_col = next((c for c in ['is_seen_item_seq_bool', 'is_seen_item_seq'] if c in df.columns), None)
    print_col = 'has_print_any' if 'has_print_any' in df.columns else None
    labels = []
    for _, r in df.iterrows():
        labels.append(dict(
            oid=str(r['object_id']), seq=r['item_seq'],
            color=str(r[color_col]) if color_col else '?',
            seen=_as_bool(r[seen_col]) if seen_col else None,
            printed=_as_bool(r[print_col]) if print_col else None,
        ))
    print(f"[labels] {args.split}: {len(labels)}건 · 색={color_col} · seen={seen_col} · print={print_col}")
    return labels


def load_ocr(args, object_ids):
    df = pd.read_csv(args.ocr_csv, low_memory=False)
    df['object_id'] = df['object_id'].astype(str)
    df = df.drop_duplicates('object_id', keep='first').set_index('object_id')
    faces, confs = {}, {}
    for oid in object_ids:
        if oid in df.index:
            r = df.loc[oid]
            txt = str(r['ocr_text']) if pd.notna(r['ocr_text']) else ''
            txt = '' if txt.strip().upper() in {'', 'NAN', 'NONE'} else txt
            cf = float(r['ocr_conf']) if pd.notna(r['ocr_conf']) else 0.0
        else:
            txt, cf = '', 0.0
        faces[oid] = [txt] if txt else []
        confs[oid] = cf
    # OCR 일관성 진단: val/test 가 같은 OCR 모델인지 통계로 드러냄
    #   (ft 모델 혼입 시 conf≈1.0 몰림 + gate 통과율 급등 → 즉시 티 남)
    cv = np.array([confs[o] for o in object_ids])
    n_read = sum(1 for o in object_ids if faces[o])
    print(f"[ocr] {Path(args.ocr_csv).name} · 판독 {n_read}/{len(object_ids)} ({n_read/max(len(object_ids),1):.3f})"
          f" · conf 평균 {cv.mean():.3f} · conf>0.99 {(cv > 0.99).mean():.3f}"
          f" · hard-gate(0.8) 통과 {(cv >= 0.8).mean():.3f}")
    return faces, confs


# ============================================================ main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--drug-master-csv', default=None)
    p.add_argument('--encoders', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ts', default=None)
    p.add_argument('--labels-csv', required=True)
    p.add_argument('--crops', required=True)
    p.add_argument('--ocr-csv', required=True)
    # DB (csv 없을 때)
    p.add_argument('--db-host', default='103.218.161.72')
    p.add_argument('--db-name', default='pilliot_db')
    p.add_argument('--db-user', default='zuhyeong_admin')
    p.add_argument('--db-pw', default=os.environ.get('PILLIOT_PW', ''))
    # fusion 파라미터
    p.add_argument('--topk-combo', type=int, default=3)
    p.add_argument('--alpha', type=float, default=2.5)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--eps', type=float, default=1e-3)
    p.add_argument('--gate-mode', default='hard', choices=['hard', 'sigmoid', 'linear'])
    # 업그레이드 토글
    p.add_argument('--dual180', action='store_true', help='180° 뒤집힘 dual-match (권장)')
    p.add_argument('--cs-gate', action='store_true', help='색·형태에 (1−g) 곱')
    p.add_argument('--legacy', action='store_true', help='노트북 그대로 재현(dual180·cs_gate 끔)')
    p.add_argument('--limit', type=int, default=0, help='스모크 테스트: 앞 N건만 평가 (0=전체)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.legacy:
        args.dual180 = False
        args.cs_gate = False
    params = dict(topk_combo=args.topk_combo, alpha=args.alpha, beta=args.beta, eps=args.eps,
                  gate_mode=args.gate_mode, use_gating=True, use_neutralize=True, use_engrave=True,
                  dual180=args.dual180, cs_gate=args.cs_gate)

    db = load_drug_master(args)
    labels = load_labels(args)
    if args.limit > 0:
        labels = labels[:args.limit]
        print(f"[smoke] --limit {args.limit} → 앞 {len(labels)}건만 평가")
    object_ids = [r['oid'] for r in labels]
    p_by_id = infer_probs(args, object_ids)
    faces, confs = load_ocr(args, object_ids)

    print(f"\n[config] {'LEGACY' if args.legacy else 'UPGRADE'} · α={args.alpha} β={args.beta} "
          f"gate={args.gate_mode} dual180={args.dual180} cs_gate={args.cs_gate} combo{args.topk_combo}")
    res = evaluate(labels, faces, confs, p_by_id, db, params)
    covk = 'coverage@%d' % args.topk_combo
    pool = res['_pool']

    print("\n" + "=" * 60)
    print(f"[{args.split}]  평가 {pool['evaluated']}건 · crop없음 {pool['miss_crop']} · "
          f"후보풀 내 정답 {pool['in_pool']}/{pool['evaluated']} (coverage 천장)")
    hdr = f"{'slice':<10}{'n':>6}{'판독':>7}{'cov':>8}{'R@1':>8}{'R@2':>8}{'R@3':>8}{'reach@3':>9}"
    print(hdr); print("-" * 60)
    for name in ['all', 'seen', 'unseen', 'printed', 'no_print']:
        if name not in res:
            continue
        r = res[name]
        print(f"{name:<10}{r['n']:>6}{r['read_rate']:>7.2f}{r[covk]:>8.3f}"
              f"{r['recall@1']:>8.3f}{r['recall@2']:>8.3f}{r['recall@3']:>8.3f}{r['reach@3']:>9.3f}")
    print("-" * 60)
    print("색별 coverage:", {c: round(v, 3) for c, v in sorted(res['_per_color'].items(), key=lambda x: -x[1])})
    print("=" * 60)


if __name__ == '__main__':
    main()
