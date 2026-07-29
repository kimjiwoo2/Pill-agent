"""OCR 파트 — 알약 각인 인식 (final, pretrained only).

Fine-tuning 없이 PP-OCRv5 pretrained 모델 + 추론 단계 전처리/회전 보정 설계만으로 구성된
최종 채택 파이프라인. 전처리 -> confidence 기반 multi-angle 회전 보정 -> detection/recognition
-> (DB 매칭용 CER 기반 채점) 순서로 동작한다. 섹션 구성은 README.md 참고.

두 가지 진입점을 제공한다:
1. `resolve_rotation_by_confidence(img, ocr, angle_step=30)` -- 기존 모듈 평가 노트북과 동일한
   시그니처의 독립 함수. e2e/fusion 쪽 설계 문서가 이 이름으로 이식 계획을 세워둔 상태라,
   기존 코드/문서를 그대로 재사용할 수 있도록 유지한다.
2. `OCRPipeline` -- 위 함수 + 전처리 + fallback + 결과 파싱까지 한 번에 묶은 상위 래퍼.
   새로 통합하는 코드는 이쪽을 쓰는 걸 권장(README 참고).

두 진입점은 동일한 detection/orientation 모델 캐시를 공유하므로, 같은 프로세스에서 둘 다 써도
모델이 중복 로드되지 않는다.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

# =============================================================================
# 1. Preprocessing -- 명암 대비/선명도 보정 + 방향 정렬 + 저해상도 업스케일
#
# rotation_label_deg 같은 정답 참고 라벨을 전혀 쓰지 않는 '블라인드' 전처리 --
# 실제 배포(새 알약 사진)와 동일한 조건.
# =============================================================================

def preprocess_crop(
    image_bgr: np.ndarray,
    clahe_clip: float = 2.5,
    clahe_tile: int = 8,
    unsharp_strength: float = 1.0,
    unsharp_sigma: float = 1.5,
) -> np.ndarray:
    """CLAHE(명암 대비) + Unsharp Masking(선명화). Grayscale로 변환해 처리 후 BGR로 복원."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), unsharp_sigma)
        enhanced = cv2.addWeighted(enhanced, 1 + unsharp_strength, blurred, -unsharp_strength, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """이미지를 angle(도)만큼 회전. 잘리지 않도록 캔버스를 확장하고, 검정 패딩 대신
    경계 픽셀을 복제(BORDER_REPLICATE)해 엣지 오인식을 방지한다."""
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
    return cv2.warpAffine(
        img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def align_to_long_axis(img: np.ndarray) -> np.ndarray:
    """세로로 긴 이미지 -> 90도 회전해 가로로 맞춤 (실제 이미지 크기 기준 단순 종횡비 휴리스틱,
    라벨 불필요). 방향(+90 고정, 반대 방향 검증 없음)의 잔여 오차는 이후 0/180도 보정 단계가
    상쇄한다."""
    h, w = img.shape[:2]
    if h > w * 1.2:
        return rotate_image(img, 90)
    return img


def upscale_if_small(img: np.ndarray, area_thresh: float = 71818, scale: float = 2.0) -> np.ndarray:
    """작은 이미지는 2배 확대(Cubic 보간) — 실제 이미지 픽셀 크기 기준으로 판단(manifest 불필요)."""
    h, w = img.shape[:2]
    if w * h <= area_thresh:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def load_and_prepare(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """원본 이미지 로드(경로인 경우) + 전체 전처리(CLAHE+Unsharp -> 방향 정렬 -> 업스케일).
    이미 로드된 ndarray(예: 데모의 실시간 카메라 프레임)를 넘겨도 전처리는 동일하게 적용된다 --
    "이미 배열이면 전처리를 통째로 건너뛴다"는 실수를 방지하기 위해 입력 종류와 무관하게
    전처리 단계는 항상 실행한다. 회전각 라벨을 쓰지 않는 블라인드 전처리 -- 실제 배포와 동일한 조건."""
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = preprocess_crop(img)
    img = align_to_long_axis(img)
    img = upscale_if_small(img)
    return img


# =============================================================================
# 2. Detection -- 텍스트 poly 기반 회전각 미세 보정 (크롭 없음)
#
# Fix 25: 크롭 없이 텍스트 poly 각도로 회전만 보정하는 방식(rotate_only)이, 크롭까지 하는 방식이나
# 정제를 아예 안 하는 방식보다 우수함이 별도 ablation으로 확인되어 최종 채택됨
# (crop 0.615 EM / 정제없음 0.610 EM / rotate_only 0.733 EM).
# =============================================================================

def get_det_polys(det_result_item: dict) -> list:
    for key in ("dt_polys", "det_polys", "boxes"):
        if key in det_result_item:
            return det_result_item[key]
    return []


def run_detection(det_model, img: np.ndarray) -> list:
    """det_model: paddleocr.TextDetection 인스턴스."""
    result = list(det_model.predict(img))
    if not result:
        return []
    return get_det_polys(result[0])


def _long_axis_angle_deg(pts: np.ndarray) -> Tuple[float, float, float]:
    e12, e23 = pts[1] - pts[0], pts[2] - pts[1]
    l12, l23 = float(np.hypot(*e12)), float(np.hypot(*e23))
    long_e, L, S = (e12, l12, l23) if l12 >= l23 else (e23, l23, l12)
    theta = float(np.degrees(np.arctan2(long_e[1], long_e[0])))
    return theta, L, S


def rotate_only(det_model, img: np.ndarray, min_area: float = 64) -> Tuple[np.ndarray, int]:
    """검출된 텍스트 poly들의 면적 가중 평균 장축각으로 회전만 보정(크롭 없음).
    최초 detection이 박스 0개로 실패하면 90/180/270도로 재시도.
    반환: (회전 적용된 전체 이미지, 검출된 박스 수)"""
    polys = run_detection(det_model, img)
    n = len(polys)

    if n == 0:
        for retry_angle in (90, 180, 270):
            rimg = rotate_image(img, retry_angle)
            rpolys = run_detection(det_model, rimg)
            if len(rpolys) > 0:
                img, polys, n = rimg, rpolys, len(rpolys)
                break

    if n == 0:
        return img, n

    polys_np: List[np.ndarray] = [np.asarray(p, dtype=np.float64) for p in polys]
    angles_rad, areas = [], []
    for p in polys_np:
        theta, L, S = _long_axis_angle_deg(p)
        angles_rad.append(np.radians(theta))
        areas.append(max(L * S, 1e-6))
    areas_arr = np.asarray(areas)
    mean_sin = float(np.sum(np.sin(angles_rad) * areas_arr) / areas_arr.sum())
    mean_cos = float(np.sum(np.cos(angles_rad) * areas_arr) / areas_arr.sum())
    theta = float(np.degrees(np.arctan2(mean_sin, mean_cos)))
    rot = (theta + 90.0) % 180.0 - 90.0

    if abs(rot) > 1.0:
        pts_all = np.concatenate(polys_np, axis=0)
        H, W = img.shape[:2]
        xs0, ys0 = pts_all[:, 0], pts_all[:, 1]
        cx, cy = float((xs0.min() + xs0.max()) / 2), float((ys0.min() + ys0.max()) / 2)
        Mrot = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        img = cv2.warpAffine(img, Mrot, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img, n


# =============================================================================
# 3. Orientation -- 0도/180도 방향 보정 (textline orientation 분류기)
#
# 회전각 미세보정(rotate_only)까지 마쳐도 텍스트가 180도 뒤집힌 채로 남을 수 있어(각도 계산만으로는
# 0/180 방향까지는 구분 못 함), 별도 분류기로 방향을 최종 확인/보정한다.
# =============================================================================

def _get_ori_label(item: dict):
    for key in ("label_names", "class_ids", "scores"):
        if key in item:
            return item[key]
    return None


def resolve_180_flip(ori_model, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """ori_model: paddleocr.TextLineOrientationClassification 인스턴스 (없으면 그대로 반환).
    모델 로드/추론 실패 시에도 조용히 원본을 반환 -- 180도 보정 없이 나머지 파이프라인은
    정상 진행되도록 방어."""
    if ori_model is None or crop is None:
        return crop
    try:
        result = list(ori_model.predict(crop))
        if not result:
            return crop
        labels = _get_ori_label(result[0])
        label_str = str(labels[0]) if isinstance(labels, (list, tuple)) and labels else str(labels)
        if "180" in label_str:
            return rotate_image(crop, 180)
        return crop
    except Exception:
        return crop


# =============================================================================
# 4. Matching -- 각인 텍스트 정규화 + CER 기반 DB 후보 매칭 점수
#
# score_one은 OCR 인식 결과를 DB 후보군(약제 각인 후보 리스트)과 대조해 랭킹을 매기는 데
# 그대로 쓰인다(추론 시점 사용, 정답 라벨 불필요) -- Late Fusion/DB 매칭 단계 입력으로 연결됨.
# 최종 후보 선택(top-3 랭킹)은 fusion 모듈이 이 점수를 가져다 수행 -- 여기는 채점만 담당.
# =============================================================================

IGNORE_IMPRINT_TOKENS = {"", "NAN", "NONE", "NULL", "마크", "분할선", "없음", "무", "-", "십자"}
_RE_STRIP_TOKENS = re.compile(r"\s+|분할선|마크|\|")
_RE_ALLOWED = re.compile(r"[^0-9A-Z가-힣+\-/]")


def _is_missing(text) -> bool:
    if text is None:
        return True
    if isinstance(text, float) and text != text:  # NaN
        return True
    return False


def normalize_imprint(text) -> str:
    """각인 텍스트 표기를 통일: 대문자화, 공백/분할선/마크/`|` 제거, 허용 문자만 남김."""
    if _is_missing(text):
        return ""
    text = str(text).strip().upper()
    if text in IGNORE_IMPRINT_TOKENS:
        return ""
    text = _RE_STRIP_TOKENS.sub("", text)
    text = _RE_ALLOWED.sub("", text)
    return "" if text in IGNORE_IMPRINT_TOKENS else text


def levenshtein(pred: str, target: str) -> int:
    p, t = list(pred), list(target)
    dp = list(range(len(t) + 1))
    for pc in p:
        ndp = [dp[0] + 1]
        for j, tc in enumerate(t):
            ndp.append(min(dp[j] + (pc != tc), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[len(t)]


def score_one(pred: str, candidates: Iterable[str]) -> Optional[float]:
    """pred(OCR 결과)와 candidates(DB 후보군 각인 텍스트) 간 CER 기반 유사도.
    score = max(0, 1 - min_c EditDist(pred, c) / max(len(c), 1))
    candidates가 전부 빈 값이면 None 반환. 관대 매칭(substring escape) 없음 -- 순수 CER 기준."""
    pred_norm = normalize_imprint(pred)
    valid = [c for c in candidates if c]
    if not valid:
        return None
    best_cer = min(levenshtein(pred_norm, c) / max(len(c), 1) for c in valid)
    return max(0.0, 1.0 - best_cer)


def exact_match(pred: str, candidates: Iterable[str]) -> bool:
    """pred가 candidates 중 하나와 정확히 일치하는지 (평가/검증용 -- 실시간 매칭에는 score_one 사용)."""
    pred_norm = normalize_imprint(pred)
    return any(pred_norm == c for c in candidates if c)


def build_candidates(print_front, print_back) -> List[str]:
    """DB의 앞면/뒷면 인쇄 텍스트로 후보 리스트 생성 (drug_dir 라벨에 의존하지 않고 항상 둘 다
    포함 -- drug_dir 라벨 오류가 채점 오류로 전이되는 것을 방지)."""
    front = normalize_imprint(print_front)
    back = normalize_imprint(print_back)
    return list(dict.fromkeys(t for t in [front, back] if t))


# =============================================================================
# 5. Pipeline -- 최종 추론 진입점 (모델 로드/전처리/회전탐색/fallback/결과파싱)
# =============================================================================

# 모델 캐시 (지연 로드, 프로세스당 1회) -- resolve_rotation_by_confidence와 OCRPipeline이 공유.
_det_model = None
_ori_model = None
_ori_model_load_attempted = False


def get_det_model():
    global _det_model
    if _det_model is None:
        from paddleocr import TextDetection

        _det_model = TextDetection(model_name="PP-OCRv5_server_det")
    return _det_model


def get_ori_model():
    """0/180도 방향 분류기. 로드 실패해도 None을 반환해 나머지 파이프라인은 정상 동작."""
    global _ori_model, _ori_model_load_attempted
    if not _ori_model_load_attempted:
        _ori_model_load_attempted = True
        try:
            from paddleocr import TextLineOrientationClassification

            _ori_model = TextLineOrientationClassification(model_name="PP-LCNet_x1_0_textline_ori")
        except Exception as exc:
            print(f"[WARN] TextLineOrientationClassification 로드 실패({exc}) -- 180도 보정 없이 진행합니다.")
    return _ori_model


def create_ocr(device: str = "gpu"):
    """최종 채택 설정으로 PaddleOCR pretrained 인스턴스 생성(recognition은 fine-tune 안 됨)."""
    from paddleocr import PaddleOCR

    return PaddleOCR(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
    )


@dataclass
class OCRResult:
    """필드명은 기존 CSV 산출물(ocr_result_*.csv) 컬럼명과 동일하게 맞춤
    (ocr_text_raw/ocr_text_norm/ocr_conf) -- e2e/fusion 쪽이 이 이름을 기준으로
    이미 설계돼 있어서, CSV 배치 진입점과 실시간 predict() 진입점의 필드명이 일치해야 함."""
    ocr_text_raw: str = ""
    ocr_text_norm: str = ""
    ocr_conf: float = float("nan")
    rec_polys: List = field(default_factory=list)
    rec_confs: List[float] = field(default_factory=list)
    used_fallback: bool = False
    error: str = ""


def _extract_ocr(page: dict):
    """PaddleOCR predict() 결과 1건을 (raw_text, norm_text, mean_conf, polys, confs)로 변환.
    다중 텍스트 라인은 위->아래, 왼쪽->오른쪽(사람이 읽는 순서)으로 재정렬한다."""
    texts = [str(t) for t in page.get("rec_texts", [])]
    confs = [float(c) for c in page.get("rec_scores", [])]
    polys = page.get("rec_polys", [])
    if not texts:
        return "", "", float("nan"), [], []
    if polys and len(polys) == len(texts):
        def _cy(poly):
            return sum(p[1] for p in poly) / len(poly)

        def _cx(poly):
            return sum(p[0] for p in poly) / len(poly)

        heights = [max(p[1] for p in poly) - min(p[1] for p in poly) for poly in polys]
        row_thresh = (sum(heights) / len(heights)) / 2 if heights else 1
        items = sorted(zip(texts, polys, confs), key=lambda x: (round(_cy(x[1]) / row_thresh), _cx(x[1])))
        texts = [t for t, _, _ in items]
        polys = [p for _, p, _ in items]
        confs = [c for _, _, c in items]
    return (
        " | ".join(texts),
        normalize_imprint("".join(texts).strip()),
        float(np.mean(confs)) if confs else float("nan"),
        polys,
        confs,
    )


def resolve_rotation_by_confidence(img: np.ndarray, ocr, angle_step: int = 30, det_thresh: float = 0.3):
    """0~330도를 angle_step 간격으로 돌려보고, 매 후보마다 rotate_only(poly 각도 미세보정,
    크롭 없음) + resolve_180_flip을 적용한 뒤 recognition confidence가 가장 높은 각도를 채택.

    기존 모듈 평가 노트북과 동일한 시그니처(img, ocr, angle_step) -- e2e 쪽에서 별도 이식 작업
    없이 `from final.ocr import resolve_rotation_by_confidence`로 그대로 가져다 쓸 수 있다.
    det_model/ori_model은 내부적으로 지연 로드되는 공유 캐시를 사용한다(직접 안 넘겨도 됨).

    반환: (선택된 이미지 또는 None, 그 각도의 OCR page 결과 또는 None, best_conf)
    """
    det_model = get_det_model()
    ori_model = get_ori_model()

    best_conf, best_img, best_page = -1.0, None, None
    for angle in range(0, 360, angle_step):
        rimg = rotate_image(img, angle) if angle else img
        refined_img, _n_boxes = rotate_only(det_model, rimg)
        refined_img = resolve_180_flip(ori_model, refined_img)
        ocr_result = ocr.predict(refined_img, text_det_thresh=det_thresh)
        if not ocr_result:
            continue
        _, _, conf, _, _ = _extract_ocr(ocr_result[0])
        if not np.isnan(conf) and conf > best_conf:
            best_conf, best_img, best_page = conf, refined_img, ocr_result[0]
    return best_img, best_page, best_conf


class OCRPipeline:
    """모델을 한 번 로드해두고 여러 이미지에 대해 predict()를 반복 호출하는 용도
    (데모/서버에서 요청마다 재로드하지 않도록). 내부적으로 resolve_rotation_by_confidence()와
    동일한 공유 det/orientation 모델 캐시를 사용한다."""

    def __init__(self, device: str = "gpu", angle_step: int = 30, det_thresh: float = 0.3):
        self.angle_step = angle_step
        self.det_thresh = det_thresh
        get_det_model()  # 공유 캐시 warm-up (없으면 첫 predict() 호출 시 지연 로드됨)
        get_ori_model()
        self.ocr = create_ocr(device=device)

    def predict(self, image: Union[str, Path, np.ndarray]) -> OCRResult:
        """단일 알약 crop 이미지(경로 또는 이미 로드된 BGR ndarray)에 대해 각인 텍스트를 인식한다."""
        result = OCRResult()
        try:
            base_img = load_and_prepare(image)
            _refined_img, best_page, _best_conf = resolve_rotation_by_confidence(
                base_img, self.ocr, self.angle_step, self.det_thresh
            )

            if best_page is None:
                # multi-angle 탐색 전부 실패해도 전체 이미지로 마지막 시도 (완전 포기 안 함)
                ocr_result = self.ocr.predict(base_img, text_det_thresh=self.det_thresh)
                if not ocr_result:
                    result.error = "multi-angle 탐색 + 전체 이미지 시도 모두 실패"
                    return result
                best_page = ocr_result[0]
                result.used_fallback = True

            raw, norm, conf, polys, confs = _extract_ocr(best_page)
            result.ocr_text_raw = raw
            result.ocr_text_norm = norm
            result.ocr_conf = conf
            result.rec_polys = [p.tolist() if hasattr(p, "tolist") else p for p in polys]
            result.rec_confs = confs
        except Exception as exc:
            result.error = str(exc)
        return result
