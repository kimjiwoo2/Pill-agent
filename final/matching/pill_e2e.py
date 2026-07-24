"""
pill_e2e.py — PILLiOT end-to-end 모듈 (앞단 YOLO + 뒷단 분류/OCR/fusion)

두 진입점:
  PillPipeline  : crop 1장 → top-k        (뒷단 ③④⑤. 이미 crop된 알약 입력)
  ScenePipeline : 원본 씬 → 알약별 top-k  (앞단 ①② + 뒷단. YOLO 검출 포함) ★ 데모용

파이프라인:
  원본 씬
    → ① YOLO 검출          알약 bbox (축정렬)
    → ② bbox_crop          margin 1.1 정사각 crop (crop_v7 규격)
    → ③a Classifier        색·형태 TS 보정 확률
    → ③b OCRReader         각인 raw text + conf (★ 정규화 안 함)
    → ④  FusionInferencer  학습된 w 로 재순위 (★ 정규화 1회)
    → ⑤  top-k item_seq

의존:
  - 주형 pill_fusion.py import (fusion 로직 단일 소스)
  - 윤수 final_v1 OCR 전처리를 OCRReader 에 이식 (val 재현 경로)
  - YOLO 추론은 ultralytics (앞단, 별도 학습된 best.pt)
"""
import json
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torchvision.transforms.functional as TF

# ── 주형 pill_fusion.py 에서 재사용 (로직 단일 소스) ──
from pill_fusion import (
    compress_candidates,
    cand_features,
    gate,
    FEAT_NAMES,
    load_drug_master,
)

CROP_MARGIN = 1.1     # crop_v7 규격 (10% 패딩, 정사각). 학습 crop 정합 — 바꾸지 말 것


# ============================================================ ③a 분류기 (단건)
class Classifier:
    """crop 1장(BGR np.array) → TS 보정 색·형태 확률."""

    class _Letterbox:
        def __init__(self, size=224, fill=128):
            self.size, self.fill = size, fill

        def __call__(self, img):
            w, h = img.size
            s = self.size / max(w, h)
            nw, nh = int(w * s), int(h * s)
            img = TF.resize(img, (nh, nw))
            pw, ph = self.size - nw, self.size - nh
            return TF.pad(img, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2), fill=self.fill)

    def __init__(self, ckpt, ts_path, encoders, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        with open(encoders, 'rb') as f:
            enc = pickle.load(f)
        n_color = len(enc['color'].classes_)
        n_shape = len(enc['shape'].classes_)

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

        self.model = PillClassifier(backbone, {'shape': n_shape, 'color': n_color}).to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        self.T_color = self.T_shape = 1.0
        if ts_path:
            with open(ts_path, 'rb') as f:
                ts = pickle.load(f)
            self.T_color = ts.get('color_T', 1.0)
            self.T_shape = ts.get('shape_T', 1.0)

        self.tf = transforms.Compose([
            self._Letterbox(224, 128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, crop_bgr):
        """crop (BGR) → (p_color, p_shape). bilateral → RGB → letterbox → normalize."""
        arr = cv2.bilateralFilter(crop_bgr, d=3, sigmaColor=15, sigmaSpace=15)
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        x = self.tf(img).unsqueeze(0).to(self.device)
        out = self.model(x)
        pc = F.softmax(out['color'] / self.T_color, 1).cpu().numpy()[0]
        ps = F.softmax(out['shape'] / self.T_shape, 1).cpu().numpy()[0]
        return pc, ps


# ============================================================ ③b OCR (단건, raw 반환)
class OCRReader:
    """crop 1장(BGR) → (raw_text, conf).  ★ 정규화 안 함 — fusion 내부에서 1회.
    윤수 final_v1 전처리 이식: load_and_prepare → 12각도 confidence 탐색 → _extract.

    mode: 'accurate'(12각도, val 재현, 느림) | 'fast'(단일 predict, 데모 속도용)
    """

    def __init__(self, use_gpu=False, mode='accurate', angle_step=30):
        from paddleocr import PaddleOCR, TextDetection
        self.mode = mode
        self.angle_step = angle_step
        dev = 'gpu' if use_gpu else 'cpu'

        self.ocr = PaddleOCR(
            device=dev,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_detection_model_name='PP-OCRv5_server_det',
            text_recognition_model_name='PP-OCRv5_server_rec',
        )
        self.det_model = TextDetection(model_name='PP-OCRv5_server_det')

        self._ori = None
        try:
            from paddleocr import TextLineOrientationClassification
            self._ori = TextLineOrientationClassification(model_name='PP-LCNet_x1_0_textline_ori')
        except Exception as e:
            print(f"[OCRReader] textline_ori 로드 실패({e}) — 180 보정 없이 진행")

    @staticmethod
    def _preprocess_crop(image_bgr, clahe_clip=2.5, clahe_tile=8,
                         unsharp_strength=1.0, unsharp_sigma=1.5):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        enhanced = clahe.apply(gray)
        if unsharp_strength > 0:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), unsharp_sigma)
            enhanced = cv2.addWeighted(enhanced, 1 + unsharp_strength, blurred, -unsharp_strength, 0)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _rotate_image(img, angle):
        if angle == 0:
            return img
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @classmethod
    def _align_to_long_axis(cls, img):
        h, w = img.shape[:2]
        return cls._rotate_image(img, 90) if h > w * 1.2 else img

    @staticmethod
    def _upscale_if_small(img, area_thresh=71818, scale=2.0):
        h, w = img.shape[:2]
        if w * h <= area_thresh:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return img

    def _load_and_prepare(self, crop_bgr):
        img = crop_bgr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = self._preprocess_crop(img)
        img = self._align_to_long_axis(img)
        img = self._upscale_if_small(img)
        return img

    def _run_detection(self, img):
        result = list(self.det_model.predict(img))
        if not result:
            return []
        item = result[0]
        for key in ('dt_polys', 'det_polys', 'boxes'):
            if key in item:
                return item[key]
        return []

    @staticmethod
    def _long_axis_angle_deg(pts):
        e12, e23 = pts[1] - pts[0], pts[2] - pts[1]
        l12, l23 = float(np.hypot(*e12)), float(np.hypot(*e23))
        long_e, L, S = (e12, l12, l23) if l12 >= l23 else (e23, l23, l12)
        theta = float(np.degrees(np.arctan2(long_e[1], long_e[0])))
        return theta, L, S

    def _rotate_only(self, img):
        polys = self._run_detection(img)
        n = len(polys)
        if n == 0:
            for a in (90, 180, 270):
                rimg = self._rotate_image(img, a)
                rp = self._run_detection(rimg)
                if len(rp) > 0:
                    img, polys, n = rimg, rp, len(rp)
                    break
        if n == 0:
            return img
        polys = [np.asarray(p, dtype=np.float64) for p in polys]
        angles_rad, areas = [], []
        for p in polys:
            theta, L, S = self._long_axis_angle_deg(p)
            angles_rad.append(np.radians(theta))
            areas.append(max(L * S, 1e-6))
        areas = np.asarray(areas)
        mean_sin = float(np.sum(np.sin(angles_rad) * areas) / areas.sum())
        mean_cos = float(np.sum(np.cos(angles_rad) * areas) / areas.sum())
        theta = float(np.degrees(np.arctan2(mean_sin, mean_cos)))
        rot = (theta + 90.0) % 180.0 - 90.0
        pts_all = np.concatenate(polys, axis=0)
        H, W = img.shape[:2]
        xs0, ys0 = pts_all[:, 0], pts_all[:, 1]
        cx, cy = float((xs0.min() + xs0.max()) / 2), float((ys0.min() + ys0.max()) / 2)
        if abs(rot) > 1.0:
            M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img

    def _resolve_180_flip(self, crop):
        if self._ori is None or crop is None:
            return crop
        try:
            result = list(self._ori.predict(crop))
            if not result:
                return crop
            item = result[0]
            labels = None
            for key in ('label_names', 'class_ids', 'scores'):
                if key in item:
                    labels = item[key]
                    break
            label_str = str(labels[0]) if isinstance(labels, (list, tuple)) and labels else str(labels)
            return self._rotate_image(crop, 180) if '180' in label_str else crop
        except Exception:
            return crop

    @staticmethod
    def _extract_raw_conf(page):
        texts = [str(t) for t in page.get('rec_texts', [])]
        confs = [float(c) for c in page.get('rec_scores', [])]
        polys = page.get('rec_polys', [])
        if not texts:
            return '', float('nan')
        if polys and len(polys) == len(texts):
            def _cy(poly): return sum(p[1] for p in poly) / len(poly)
            def _cx(poly): return sum(p[0] for p in poly) / len(poly)
            heights = [max(p[1] for p in poly) - min(p[1] for p in poly) for poly in polys]
            row_thresh = (sum(heights) / len(heights)) / 2 if heights else 1
            items = sorted(zip(texts, polys, confs),
                           key=lambda x: (round(_cy(x[1]) / row_thresh), _cx(x[1])))
            texts = [t for t, _, _ in items]
            confs = [c for _, _, c in items]
        raw = ''.join(texts).strip()
        conf = float(np.mean(confs)) if confs else float('nan')
        return raw, conf

    def read(self, crop_bgr):
        """crop(BGR) → (raw_text, conf). raw_text 는 정규화 전 원문."""
        base = self._load_and_prepare(crop_bgr)

        if self.mode == 'fast':
            res = self.ocr.predict(base, text_det_thresh=0.3)
            if not res:
                return '', 0.0
            raw, conf = self._extract_raw_conf(res[0])
            return raw, (0.0 if np.isnan(conf) else conf)

        best_conf, best_page = -1.0, None
        for angle in range(0, 360, self.angle_step):
            rimg = self._rotate_image(base, angle) if angle else base
            refined = self._rotate_only(rimg)
            refined = self._resolve_180_flip(refined)
            res = self.ocr.predict(refined, text_det_thresh=0.3)
            if not res:
                continue
            raw, conf = self._extract_raw_conf(res[0])
            if not np.isnan(conf) and conf > best_conf:
                best_conf, best_page = conf, res[0]
        if best_page is None:
            res = self.ocr.predict(base, text_det_thresh=0.3)
            if not res:
                return '', 0.0
            best_page = res[0]
        raw, conf = self._extract_raw_conf(best_page)
        return raw, (0.0 if np.isnan(conf) else conf)


# ============================================================ ④ Fusion (단건, 학습 w)
class FusionInferencer:
    """학습된 새식 w(weights.json) 로드 → 단건 재순위."""

    def __init__(self, weights_json, db):
        with open(weights_json) as f:
            wj = json.load(f)
        assert list(wj['feat_names']) == list(FEAT_NAMES), "피처 불일치 — pill_fusion 버전 확인"
        self.w = np.array(wj['w'])
        self.gate_mode = wj['gate_mode']
        self.gate_params = wj.get('gate_params')   # calibrated 곡선/커스텀 thr — run_apply_weights와 동일 로딩
        self.eps = wj['eps']
        self.topk_combo = wj['topk_combo']
        self.combo_to_items = db['combo_to_items']
        self.cand_info = db['cand_info']
        self.CC = db['COLOR_CLASSES']
        self.SC = db['SHAPE_CLASSES']

    def rank(self, p_color, p_shape, ocr_raw_text, ocr_conf, k=3):
        cand = compress_candidates(p_color, p_shape, self.combo_to_items,
                                   self.CC, self.SC, self.topk_combo)
        if not cand:
            return []
        txt = '' if str(ocr_raw_text).strip().upper() in {'', 'NAN', 'NONE'} else str(ocr_raw_text)
        qf = [txt] if txt else []
        g = gate(ocr_conf, self.gate_mode, self.gate_params)

        scored = []
        for seq in cand:
            info = self.cand_info.get(seq)
            if info is None:
                continue
            f = cand_features(qf, g, p_color, p_shape, info, self.eps)
            score = float(self.w @ f)
            ci, si = info['ci'], info['si']
            tb = (p_color[ci] if ci is not None else 0.0) + (p_shape[si] if si is not None else 0.0)
            scored.append((score, tb, seq))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [(seq, sc) for sc, tb, seq in scored[:k]]


# ============================================================ ① 검출 (YOLO, 단건)
class Detector:
    """원본 씬(BGR) → 알약 bbox 리스트. 다른 모듈처럼 로드·추론 분리.
    축정렬 bbox (OBB 아님) — 학습된 yolo11n best.pt 사용."""

    def __init__(self, yolo_pt):
        from ultralytics import YOLO
        self.model = YOLO(yolo_pt)
        print(f"[Detector] YOLO 로드: {yolo_pt}")

    def detect(self, scene_bgr, conf_thres=0.25):
        """씬 → [(cx, cy, bw, bh, det_conf), ...] 픽셀 좌표."""
        res = self.model.predict(scene_bgr, conf=conf_thres, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf[0])))
        return boxes


# ============================================================ crop (앞단 ②)
def bbox_crop(img_bgr, cx, cy, bw, bh, margin=CROP_MARGIN):
    """축정렬 bbox 중심에서 margin 배수 정사각 crop (회전 없음).
    obb_upright_crop.py 무회전 경로 + _crop_exact 와 동일 규격 → crop_v7 재현.
    cx,cy,bw,bh: 픽셀 좌표 (YOLO 출력)."""
    side = int(round(max(bw, bh) * margin))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    H, W = img_bgr.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    if pad_l or pad_t or pad_r or pad_b:
        img_bgr = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; y0 += pad_t
    return img_bgr[y0:y0 + side, x0:x0 + side]


# ============================================================ 뒷단 파이프라인 (crop 1장 → top-k)
class PillPipeline:
    """crop 1장 → top-k item_seq. (③④⑤. ScenePipeline 이 내부적으로 이걸 씀)"""

    def __init__(self, ckpt, ts_path, encoders, weights_json, db,
                 ocr_gpu=False, ocr_mode='accurate'):
        self.clf = Classifier(ckpt, ts_path, encoders)
        self.ocr = OCRReader(use_gpu=ocr_gpu, mode=ocr_mode)
        self.fusion = FusionInferencer(weights_json, db)
        self.db = db
        self._dm = db['dm_valid'].set_index('item_seq')

    def _name(self, seq):
        if seq in self._dm.index and 'dl_name' in self._dm.columns:
            return self._dm.loc[seq]['dl_name']
        return str(seq)

    def run(self, crop_bgr, k=3):
        pc, ps = self.clf.predict(crop_bgr)                 # ③a
        raw, conf = self.ocr.read(crop_bgr)                 # ③b
        topk = self.fusion.rank(pc, ps, raw, conf, k=k)     # ④
        results = [{'item_seq': seq, 'score': score, 'name': self._name(seq)}
                   for seq, score in topk]                  # ⑤
        return {'topk': results, 'ocr_raw': raw, 'ocr_conf': conf,
                'p_color': pc, 'p_shape': ps}


# ============================================================ 앞단+뒷단 통합 (원본 씬 → 알약별 top-k)
class ScenePipeline:
    """4개 모듈(Detector + Classifier/OCR/Fusion=PillPipeline)을 조립.  ★ 데모 최상위 진입점.

    사용:
      pipe = ScenePipeline(ckpt=.., ts_path=.., encoders=.., weights_json=.., yolo_pt=.., db=db)
      results = pipe.run_scene(scene_bgr)      # 알약별 top-k 리스트
      vis = pipe.visualize(scene_bgr, results) # bbox + top-1 이름 그린 RGB 이미지
    """

    def __init__(self, ckpt, ts_path, encoders, weights_json, yolo_pt, db,
                 ocr_gpu=False, ocr_mode='accurate'):
        self.detector = Detector(yolo_pt)                          # ①
        self.backend = PillPipeline(ckpt, ts_path, encoders,       # ③④⑤
                                    weights_json, db,
                                    ocr_gpu=ocr_gpu, ocr_mode=ocr_mode)

    def run_scene(self, scene_bgr, k=3, det_conf=0.25):
        """원본 씬 1장 → 검출된 알약마다 top-k. 다중 알약을 순차 집계
        (알약끼리 독립 → 순차/병렬 결과 동일. 데모 규모엔 순차로 충분)."""
        boxes = self.detector.detect(scene_bgr, det_conf)          # ①
        results = []
        for i, (cx, cy, bw, bh, dconf) in enumerate(boxes):
            crop = bbox_crop(scene_bgr, cx, cy, bw, bh)
            out = self.backend.run(crop, k=k)
            results.append({
                'pill_idx': i,
                'bbox': (int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)),
                'det_conf': dconf,
                'ocr_raw': out['ocr_raw'], 'ocr_conf': out['ocr_conf'],
                'topk': out['topk'],
            })
        return results

    def visualize(self, scene_bgr, results):
        """bbox + top-1 이름 그린 RGB 이미지 반환 (plt.imshow 로 띄우면 됨)."""
        vis = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
        for r in results:
            x, y, w, h = r['bbox']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            top1 = r['topk'][0]['name'] if r['topk'] else '?'
            cv2.putText(vis, f"#{r['pill_idx']} {top1}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis