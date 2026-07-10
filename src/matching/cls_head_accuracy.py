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

    n = c_ok = s_ok = both = 0
    miss_crop = miss_gt = 0
    per_color = defaultdict(lambda: [0, 0])      # gt색: [맞음, 전체]
    per_shape = defaultdict(lambda: [0, 0])
    cm_color = defaultdict(int)                   # (gt, pred) 오분류

    for r in labels:
        oid, gt = r['oid'], r['seq']
        if oid not in p_by_id:
            miss_crop += 1; continue
        info = cand_info.get(gt)
        if info is None or info['ci'] is None or info['si'] is None:
            miss_gt += 1; continue
        pc, ps = p_by_id[oid]
        pcol, psh = int(np.argmax(pc)), int(np.argmax(ps))
        gci, gsi = info['ci'], info['si']
        n += 1
        cc, ss = (pcol == gci), (psh == gsi)
        c_ok += cc; s_ok += ss; both += (cc and ss)
        per_color[CC[gci]][1] += 1; per_color[CC[gci]][0] += cc
        per_shape[SC[gsi]][1] += 1; per_shape[SC[gsi]][0] += ss
        if not cc:
            cm_color[(CC[gci], CC[pcol])] += 1

    print("\n" + "=" * 56)
    print(f"[{args.split}] 분류기 head 정확도  (평가 {n} · crop없음 {miss_crop} · GT색형태없음 {miss_gt})")
    print("-" * 56)
    print(f"  color accuracy      = {c_ok/max(n,1):.4f}  ({c_ok}/{n})")
    print(f"  shape accuracy      = {s_ok/max(n,1):.4f}  ({s_ok}/{n})")
    print(f"  color+shape 동시     = {both/max(n,1):.4f}  ({both}/{n})")
    print("-" * 56)
    print("색별 accuracy:")
    for name, (ok, tot) in sorted(per_color.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<6} {ok/max(tot,1):.3f}  ({ok}/{tot})")
    print("모양별 accuracy:")
    for name, (ok, tot) in sorted(per_shape.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<6} {ok/max(tot,1):.3f}  ({ok}/{tot})")
    print("-" * 56)
    print("주요 색 오분류 (gt→pred, top 8):")
    for (g, p), cnt in sorted(cm_color.items(), key=lambda x: -x[1])[:8]:
        print(f"  {g:<6} → {p:<6} {cnt}")
    print("=" * 56)


if __name__ == '__main__':
    main()
