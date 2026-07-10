"""
분류기 head 정확도(color/shape) 측정 — pill_fusion.py 함수 재사용, 비침습.
GT = drug_master에 등록된 그 item_seq의 색/모양, Pred = 분류기 argmax.
※ Temperature Scaling은 argmax를 바꾸지 않으므로 정확도엔 무관(있어도 무방).

pill_fusion.py 와 같은 폴더(src/matching/)에 두고 동일 인자로 실행.

사용 (test):
  python cls_head_accuracy.py --split test \
    --drug-master-csv drug_master.csv --encoders label_encoders_20k_11cls.pkl \
    --ckpt best_20k_v3_5_nosampler_ep16_v3.pth --ts temperature_20k_v3_5_nosampler_ep16_v3.pkl \
    --labels-csv final_test_manifest.csv --crops test_filtered.zip \
    --ocr-csv ocr_result_final_v1_test.csv
  # val 이면: --split val --labels-csv manifest_clean_20k_33340.csv --crops crop_v7_val.zip ...
"""
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pill_fusion as pf


def main():
    args = pf.parse_args()                       # pill_fusion 과 동일 인자
    db = pf.load_drug_master(args)
    labels = pf.load_labels(args)
    oids = [r['oid'] for r in labels]
    p_by_id = pf.infer_probs(args, oids)         # {oid: (pc, ps)} TS 보정 확률

    cand_info = db['cand_info']
    CC, SC = db['COLOR_CLASSES'], db['SHAPE_CLASSES']
    combo = db['combo_to_items']

    n = c_ok = s_ok = both = 0
    c_top2 = c_top3 = s_top2 = s_top3 = 0
    miss_crop = miss_gt = 0
    # per-class 집계: true(gt 총) · pred(예측 총) · correct(gt==pred)
    ct = defaultdict(int); cp = defaultdict(int); cc = defaultdict(int)   # color
    st = defaultdict(int); sp = defaultdict(int); sc = defaultdict(int)   # shape
    cm_color = defaultdict(int)                   # (gt, pred) 오분류
    # class-head ranking (정답 클래스의 순위 기반)
    col_rr = shp_rr = col_pgt = shp_pgt = 0.0
    col_ranksum = shp_ranksum = 0
    # 약품 레벨 retrieval (색·모양만으로 4,461종 후보 축소·랭킹)
    cov = {1: 0, 2: 0, 3: 0}                       # coverage@combo-k (정답 약이 후보풀에)
    dr = {1: 0, 3: 0, 5: 0, 10: 0}                 # drug recall@k (풀 내 정답 랭크)
    drug_rr = ndcg = 0.0
    pool_sz = in_pool = 0

    for r in labels:
        oid, gt = r['oid'], r['seq']
        if oid not in p_by_id:
            miss_crop += 1; continue
        info = cand_info.get(gt)
        if info is None or info['ci'] is None or info['si'] is None:
            miss_gt += 1; continue
        pc, ps = p_by_id[oid]
        gci, gsi = info['ci'], info['si']
        order = np.argsort(pc)[::-1]              # color 확률 내림차순
        sorder = np.argsort(ps)[::-1]             # shape 확률 내림차순
        pcol, psh = int(order[0]), int(sorder[0])
        n += 1
        col_ok, sh_ok = (pcol == gci), (psh == gsi)
        c_ok += col_ok; s_ok += sh_ok; both += (col_ok and sh_ok)
        c_top2 += (gci in order[:2]); c_top3 += (gci in order[:3])
        s_top2 += (gsi in sorder[:2]); s_top3 += (gsi in sorder[:3])
        gcn, pcn = CC[gci], CC[pcol]; gsn, psn = SC[gsi], SC[psh]
        ct[gcn] += 1; cp[pcn] += 1; cc[gcn] += col_ok
        st[gsn] += 1; sp[psn] += 1; sc[gsn] += sh_ok
        if not col_ok:
            cm_color[(gcn, pcn)] += 1

        # ── class-head ranking (정답 클래스 순위) ──
        col_rank = int(np.where(order == gci)[0][0]) + 1
        shp_rank = int(np.where(sorder == gsi)[0][0]) + 1
        col_rr += 1.0 / col_rank; shp_rr += 1.0 / shp_rank
        col_ranksum += col_rank; shp_ranksum += shp_rank
        col_pgt += float(pc[gci]); shp_pgt += float(ps[gsi])

        # ── 약품 레벨 retrieval (색·모양만, pill_fusion 로직 재사용 → 그래프와 동일) ──
        for k in (1, 2, 3):
            if gt in pf.compress_candidates(pc, ps, combo, CC, SC, topk_combo=k):
                cov[k] += 1
        pool = pf.compress_candidates(pc, ps, combo, CC, SC, topk_combo=3)
        pool_sz += len(pool)
        if gt in pool:
            in_pool += 1
        # 각인 없이 색·모양만으로 후보 랭킹 (competition-rank 채점 = 노트북/fusion 동일)
        scored = pf.match_fused([], 0.0, pool, pc, ps, cand_info,
                                use_engrave=False, use_gating=False,
                                dual180=False, cs_gate=False, return_scores=True)
        for k in (1, 3, 5, 10):
            if pf.in_topk_ties(scored, gt, k):
                dr[k] += 1
        rk = pf.rank_of_gt(scored, gt)
        if rk is not None:
            drug_rr += 1.0 / rk
            if rk <= 10:
                ndcg += 1.0 / np.log2(rk + 1)

    def prf(true_tot, pred_tot, corr, keys):
        """per-class precision/recall/f1 + macro·weighted 평균 반환."""
        rows = []
        for k in keys:
            tp = corr[k]; sup = true_tot[k]; pp = pred_tot.get(k, 0)
            p = tp / pp if pp else 0.0
            rec = tp / sup if sup else 0.0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            rows.append((k, p, rec, f1, sup))
        N = sum(true_tot.values())
        cls = [r for r in rows if r[4] > 0]
        macro = (np.mean([r[1] for r in cls]), np.mean([r[2] for r in cls]), np.mean([r[3] for r in cls]))
        wf1 = sum(r[3] * r[4] for r in rows) / N if N else 0.0
        wp = sum(r[1] * r[4] for r in rows) / N if N else 0.0
        wr = sum(r[2] * r[4] for r in rows) / N if N else 0.0
        return rows, macro, (wp, wr, wf1)

    def report(title, true_tot, pred_tot, corr):
        keys = sorted(set(true_tot) | set(pred_tot), key=lambda k: -true_tot.get(k, 0))
        rows, macro, weigh = prf(true_tot, pred_tot, corr, keys)
        print(f"\n[{title}]  {'class':<8}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
        print("  " + "-" * 46)
        for k, p, rec, f1, sup in rows:
            print(f"  {k:<8}{p:>7.3f}{rec:>8.3f}{f1:>7.3f}{sup:>9}")
        print("  " + "-" * 46)
        print(f"  {'macro':<8}{macro[0]:>7.3f}{macro[1]:>8.3f}{macro[2]:>7.3f}")
        print(f"  {'weighted':<8}{weigh[0]:>7.3f}{weigh[1]:>8.3f}{weigh[2]:>7.3f}{sum(true_tot.values()):>9}")

    print("\n" + "=" * 56)
    print(f"[{args.split}] 분류기 head 성능  (평가 {n} · crop없음 {miss_crop} · GT색형태없음 {miss_gt})")
    print("-" * 56)
    print(f"  color  accuracy(top-1) = {c_ok/max(n,1):.4f}  ({c_ok}/{n})")
    print(f"  color  recall@2 / @3   = {c_top2/max(n,1):.4f} / {c_top3/max(n,1):.4f}")
    print(f"  shape  accuracy(top-1) = {s_ok/max(n,1):.4f}  ({s_ok}/{n})")
    print(f"  shape  recall@2 / @3   = {s_top2/max(n,1):.4f} / {s_top3/max(n,1):.4f}")
    print(f"  color+shape 동시        = {both/max(n,1):.4f}  ({both}/{n})")

    report('color', ct, cp, cc)
    report('shape', st, sp, sc)

    nn = max(n, 1)
    print("\n" + "-" * 56)
    print("[class-head ranking]  (정답 클래스 순위 기반)")
    print(f"  color  MRR={col_rr/nn:.4f}  mean-rank={col_ranksum/nn:.3f}  "
          f"exp_recall(E[P_gt])={col_pgt/nn:.4f}")
    print(f"  shape  MRR={shp_rr/nn:.4f}  mean-rank={shp_ranksum/nn:.3f}  "
          f"exp_recall(E[P_gt])={shp_pgt/nn:.4f}")

    print("\n[drug-level retrieval]  색·모양만 (4,461종 대상 · drug recall@3 = 전략비교 그래프의 '색·모양만')")
    print(f"  평균 후보풀 크기        = {pool_sz/nn:.1f} 종")
    print(f"  coverage@combo 1/2/3   = {cov[1]/nn:.4f} / {cov[2]/nn:.4f} / {cov[3]/nn:.4f}")
    print(f"  drug recall@1/3/5/10   = {dr[1]/nn:.4f} / {dr[3]/nn:.4f} / {dr[5]/nn:.4f} / {dr[10]/nn:.4f}")
    print(f"  drug MRR               = {drug_rr/nn:.4f}")
    print(f"  reach@3 (recall/cov)   = {(dr[3]/nn)/max(cov[3]/nn,1e-9):.4f}")
    print(f"  NDCG@10                = {ndcg/nn:.4f}")
    print(f"  후보풀 내 정답 비율      = {in_pool/nn:.4f}  (= coverage@3 상한)")

    print("\n주요 색 오분류 (gt→pred, top 8):")
    for (g, p), cnt in sorted(cm_color.items(), key=lambda x: -x[1])[:8]:
        print(f"  {g:<6} → {p:<6} {cnt}")
    print("=" * 56)


if __name__ == '__main__':
    main()
