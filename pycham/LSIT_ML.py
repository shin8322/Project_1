import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass
class Detection:
    idx: int
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int
    area: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_path(path: str) -> str:
    if path is None:
        return ""
    path = str(path).strip().strip('"').strip("'")
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(path))


def read_image(image_path: str) -> np.ndarray:
    image_path = normalize_path(image_path)

    if not image_path:
        raise FileNotFoundError("이미지 경로가 비어 있습니다.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다:\n{image_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"지정한 경로가 파일이 아닙니다:\n{image_path}")

    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"이미지 파일을 읽을 수 없습니다:\n{image_path}")

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 디코딩할 수 없습니다:\n{image_path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    path = normalize_path(path)
    ensure_dir(str(Path(path).parent))

    ext = Path(path).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        ext = ".png"
        path = str(Path(path).with_suffix(ext))

    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"이미지 인코딩 실패: {path}")
    encoded.tofile(path)


def save_detections_csv(csv_path: str, detections: List[Detection]) -> None:
    csv_path = normalize_path(csv_path)
    ensure_dir(str(Path(csv_path).parent))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "cx", "cy", "x1", "y1", "x2", "y2", "area"])
        for d in detections:
            writer.writerow([d.idx, d.cx, d.cy, d.x1, d.y1, d.x2, d.y2, f"{d.area:.2f}"])


def ensure_odd(n: int, default: int = 5, minimum: int = 3) -> int:
    if n is None or n <= 0:
        n = default
    if n < minimum:
        n = minimum
    if n % 2 == 0:
        n += 1
    return n


def preprocess_gray(gray: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    blur_ksize = ensure_odd(blur_ksize, default=5, minimum=1)
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def threshold_candidates(
    gray: np.ndarray,
    method: str = "adaptive",
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
) -> np.ndarray:
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        adaptive_block_size = ensure_odd(adaptive_block_size, default=31, minimum=3)
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block_size,
            adaptive_c,
        )
    return th


def apply_morphology(mask: np.ndarray, morph_kernel: int = 3, open_iter: int = 1, close_iter: int = 1) -> np.ndarray:
    morph_kernel = max(1, morph_kernel)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)

    out = mask.copy()
    if open_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return out


def detect_saturated_right_band(
    gray: np.ndarray,
    sat_thresh: int = 250,
    col_ratio_thresh: float = 0.90,
    min_band_width: int = 2,
) -> int:
    h, w = gray.shape[:2]
    saturated_cols = []

    for x in range(w - 1, -1, -1):
        col = gray[:, x]
        ratio = float(np.mean(col >= sat_thresh))
        if ratio >= col_ratio_thresh:
            saturated_cols.append(x)
        else:
            break

    if len(saturated_cols) >= min_band_width:
        return min(saturated_cols)
    return w


def non_max_suppression_boxes(boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.3) -> List[int]:
    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / np.maximum(areas[idxs[:last]], 1)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return pick


def detect_diffraction_patterns(
    image_bgr: np.ndarray,
    crop_size: int = 30,
    min_area: int = 20,
    max_area: Optional[int] = None,
    blur_ksize: int = 5,
    thresh_method: str = "adaptive",
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
    morph_kernel: int = 3,
    open_iter: int = 1,
    close_iter: int = 1,
    nms_overlap: float = 0.3,
    remove_right_saturation: bool = True,
    sat_thresh: int = 250,
    sat_col_ratio: float = 0.90,
    sat_min_band_width: int = 2,
    border_margin: int = 2,
) -> Tuple[List[Detection], np.ndarray, int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = preprocess_gray(gray, blur_ksize=blur_ksize)

    mask = threshold_candidates(
        gray=gray,
        method=thresh_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )

    mask = apply_morphology(mask, morph_kernel=morph_kernel, open_iter=open_iter, close_iter=close_iter)

    h, w = gray.shape[:2]
    valid_xmax = w

    if remove_right_saturation:
        valid_xmax = detect_saturated_right_band(
            gray=gray,
            sat_thresh=sat_thresh,
            col_ratio_thresh=sat_col_ratio,
            min_band_width=sat_min_band_width,
        )
        if valid_xmax < w:
            mask[:, valid_xmax:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crop_size = max(1, int(crop_size))
    half = crop_size // 2
    border_margin = max(0, int(border_margin))

    boxes = []
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)

        if border_margin > 0:
            if x <= border_margin or y <= border_margin or x + ww >= w - border_margin or y + hh >= h - border_margin:
                continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)

        x1 = max(0, x2 - crop_size)
        y1 = max(0, y2 - crop_size)

        if border_margin > 0:
            if x1 <= border_margin or y1 <= border_margin or x2 >= w - border_margin or y2 >= h - border_margin:
                continue

        boxes.append((x1, y1, x2, y2))
        candidates.append((cx, cy, x1, y1, x2, y2, area))

    keep = non_max_suppression_boxes(boxes, overlap_thresh=nms_overlap)

    detections: List[Detection] = []
    for new_idx, old_idx in enumerate(keep, start=1):
        cx, cy, x1, y1, x2, y2, area = candidates[old_idx]
        detections.append(Detection(new_idx, int(cx), int(cy), int(x1), int(y1), int(x2), int(y2), float(area)))

    detections.sort(key=lambda d: (d.y1, d.x1))
    for i, d in enumerate(detections, start=1):
        d.idx = i

    return detections, mask, valid_xmax


def crop_square(image_bgr: np.ndarray, det: Detection, crop_size: int) -> np.ndarray:
    crop = image_bgr[det.y1:det.y2, det.x1:det.x2].copy()
    h, w = crop.shape[:2]

    if h != crop_size or w != crop_size:
        pad_bottom = max(0, crop_size - h)
        pad_right = max(0, crop_size - w)
        crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop


def is_bad_crop_by_right_saturation(
    crop_bgr: np.ndarray,
    edge_width: int = 3,
    sat_thresh: int = 250,
    ratio_thresh: float = 0.85,
) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return True

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    edge_width = max(1, min(edge_width, w))

    right_band = gray[:, w - edge_width:w]
    sat_ratio = float(np.mean(right_band >= sat_thresh))
    return sat_ratio >= ratio_thresh


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Detection],
    valid_xmax: Optional[int] = None,
    selected_idx: Optional[int] = None,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if valid_xmax is not None and valid_xmax < image_bgr.shape[1]:
        cv2.line(canvas, (valid_xmax, 0), (valid_xmax, image_bgr.shape[0] - 1), (0, 165, 255), 1)
        cv2.putText(
            canvas,
            "saturation cutoff",
            (max(5, valid_xmax - 130), 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    for idx, det in enumerate(detections):
        is_selected = selected_idx is not None and idx == selected_idx
        color = (0, 255, 255) if is_selected else (0, 255, 0)
        thickness = 2 if is_selected else 1
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
        cv2.circle(canvas, (det.cx, det.cy), 3 if is_selected else 2, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            str(det.idx),
            (det.x1, max(12, det.y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if is_selected:
            cv2.putText(
                canvas,
                "selected",
                (det.x1, min(image_bgr.shape[0] - 6, det.y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if len(mask.shape) == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def add_title_bar(img_bgr: np.ndarray, title: str, height: int = 28) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((height, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (10, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def make_side_by_side(original_bgr: np.ndarray, annotated_bgr: np.ndarray) -> np.ndarray:
    target_h = max(original_bgr.shape[0], annotated_bgr.shape[0])

    def resize_h(img: np.ndarray, target_h_: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_h_ / h
        new_w = max(1, int(w * scale))
        return cv2.resize(img, (new_w, target_h_), interpolation=cv2.INTER_AREA)

    left = resize_h(original_bgr, target_h)
    right = resize_h(annotated_bgr, target_h)

    left = add_title_bar(left, "Original")
    right = add_title_bar(right, "Annotated / Crop Regions")

    gap = np.full((left.shape[0], 20, 3), 255, dtype=np.uint8)
    return np.hstack([left, gap, right])


def process_image(
    image_path: str,
    output_dir: str,
    crop_size: int = 30,
    min_area: int = 20,
    max_area: Optional[int] = None,
    blur_ksize: int = 5,
    thresh_method: str = "adaptive",
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
    morph_kernel: int = 3,
    open_iter: int = 1,
    close_iter: int = 1,
    nms_overlap: float = 0.3,
    remove_right_saturation: bool = True,
    sat_thresh: int = 250,
    sat_col_ratio: float = 0.90,
    sat_min_band_width: int = 2,
    border_margin: int = 2,
    reject_bad_right_crop: bool = True,
    crop_right_edge_width: int = 3,
    crop_sat_ratio: float = 0.85,
) -> Dict[str, Any]:
    image_path = normalize_path(image_path)
    output_dir = normalize_path(output_dir)

    if not image_path:
        raise ValueError("이미지 경로가 비어 있습니다.")
    if not output_dir:
        raise ValueError("출력 폴더가 비어 있습니다.")

    ensure_dir(output_dir)
    crops_dir = os.path.join(output_dir, "crops")
    ensure_dir(crops_dir)

    image_bgr = read_image(image_path)

    detections, mask, valid_xmax = detect_diffraction_patterns(
        image_bgr=image_bgr,
        crop_size=crop_size,
        min_area=min_area,
        max_area=max_area,
        blur_ksize=blur_ksize,
        thresh_method=thresh_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_kernel=morph_kernel,
        open_iter=open_iter,
        close_iter=close_iter,
        nms_overlap=nms_overlap,
        remove_right_saturation=remove_right_saturation,
        sat_thresh=sat_thresh,
        sat_col_ratio=sat_col_ratio,
        sat_min_band_width=sat_min_band_width,
        border_margin=border_margin,
    )

    saved_detections: List[Detection] = []
    saved_idx = 1

    for det in detections:
        crop = crop_square(image_bgr, det, crop_size)

        if reject_bad_right_crop and is_bad_crop_by_right_saturation(
            crop_bgr=crop,
            edge_width=crop_right_edge_width,
            sat_thresh=sat_thresh,
            ratio_thresh=crop_sat_ratio,
        ):
            continue

        det.idx = saved_idx
        saved_detections.append(det)
        save_image(os.path.join(crops_dir, f"crop_{saved_idx:03d}.png"), crop)
        saved_idx += 1

    detections = saved_detections

    annotated = draw_detections(image_bgr, detections, valid_xmax=valid_xmax)
    overview = make_side_by_side(image_bgr, annotated)

    original_path = os.path.join(output_dir, "original.png")
    mask_path = os.path.join(output_dir, "mask.png")
    annotated_path = os.path.join(output_dir, "annotated.png")
    overview_path = os.path.join(output_dir, "overview_original_vs_annotated.png")
    csv_path = os.path.join(output_dir, "detections.csv")

    save_image(original_path, image_bgr)
    save_image(mask_path, mask)
    save_image(annotated_path, annotated)
    save_image(overview_path, overview)
    save_detections_csv(csv_path, detections)

    parameter_info = perform_parameter_analysis(
        crops_dir=crops_dir,
        output_dir=output_dir,
        smooth_win=5,
        start_idx=2,
        blur_ksize=5,
        bg_margin=3,
        outer_border_margin=2,
        local_std_ksize=5,
        quantile_keep=0.60,
    )

    return {
        "input_image": image_path,
        "output_dir": output_dir,
        "crop_count": len(detections),
        "original_path": original_path,
        "mask_path": mask_path,
        "annotated_path": annotated_path,
        "overview_path": overview_path,
        "csv_path": csv_path,
        "crops_dir": crops_dir,
        "original_image": image_bgr,
        "mask_image": mask_to_bgr(mask),
        "annotated_image": annotated,
        "overview_image": overview,
        "detections": detections,
        "valid_xmax": valid_xmax,
        **parameter_info,
    }



# =========================================================
# Parameter analysis for saved crop images
# - 1-line center / axis parameters
# - external background pixels only baseline
# - band pixel counts
# =========================================================

PARAMETER_SUMMARY_FIELDS = [
    "image_name", "image_path", "status", "baseline_status", "band_status", "baseline_error", "band_error",
    "analysis_overlay_path",
    "center_x", "center_y", "CMV",
    "PPD_x_plus", "PPD_x_minus", "PPD_y_plus", "PPD_y_minus", "PPD_avg", "PPD_std",
    "MMD_x_plus", "MMD_x_minus", "MMD_y_plus", "MMD_y_minus", "MMD_avg", "MMD_std",
    "WCM_x_plus", "WCM_x_minus", "WCM_y_plus", "WCM_y_minus", "WCM_x", "WCM_y",
    "WSM_x_plus", "WSM_x_minus", "WSM_y_plus", "WSM_y_minus", "WSM_x", "WSM_y",
    "pattern_end_radius", "background_pixel_count", "background_r_start", "background_r_end",
    "baseline_external_pixels", "baseline_sigma", "baseline_tol", "grad_thr", "local_std_thr",
    "cross_1", "cross_2", "cross_3", "cross_4",
    "center_bright_pixel_count", "ring_dark_1_pixel_count", "ring_bright_1_pixel_count", "ring_dark_2_pixel_count",
]


def save_summary_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(str(Path(csv_path).parent))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PARAMETER_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            safe_row = {k: row.get(k, "") for k in PARAMETER_SUMMARY_FIELDS}
            writer.writerow(safe_row)


def save_summary_json(json_path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(str(Path(json_path).parent))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def read_gray_for_analysis(image_path: str) -> np.ndarray:
    image_path = normalize_path(image_path)
    data = np.fromfile(image_path, dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return gray.astype(np.float32)


def smooth_1d(arr: np.ndarray, win: int = 5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if win <= 1:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(padded, kernel, mode="valid")


def robust_mad_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad)


def local_minima_indices(profile: np.ndarray) -> np.ndarray:
    d = np.diff(profile)
    return np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1


def local_maxima_indices(profile: np.ndarray) -> np.ndarray:
    d = np.diff(profile)
    return np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1


def find_first_min_then_first_max(profile: np.ndarray, start_idx: int = 2) -> Tuple[Optional[int], Optional[int]]:
    mins = local_minima_indices(profile)
    mins = mins[mins >= start_idx]
    if len(mins) == 0:
        return None, None
    min_idx = int(mins[0])

    maxs = local_maxima_indices(profile)
    maxs = maxs[maxs > min_idx]
    if len(maxs) == 0:
        return min_idx, None
    return min_idx, int(maxs[0])


def find_center_1line_analysis(
    gray: np.ndarray,
    blur_ksize: int = 5,
    smooth_win: int = 5,
    n_iter: int = 3,
    search_half: Optional[int] = None,
) -> Tuple[int, int]:
    if blur_ksize < 1:
        blur_ksize = 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    h, w = blur.shape
    cx, cy = w // 2, h // 2

    if search_half is None:
        search_half = max(6, min(h, w) // 4)

    for _ in range(n_iter):
        row = smooth_1d(blur[cy, :], smooth_win)
        x0 = max(0, cx - search_half)
        x1 = min(w, cx + search_half + 1)
        if x1 > x0:
            cx = int(x0 + np.argmax(row[x0:x1]))

        col = smooth_1d(blur[:, cx], smooth_win)
        y0 = max(0, cy - search_half)
        y1 = min(h, cy + search_half + 1)
        if y1 > y0:
            cy = int(y0 + np.argmax(col[y0:y1]))

    return cx, cy


def split_profiles_1line_analysis(gray: np.ndarray, cx: int, cy: int, smooth_win: int = 5) -> Dict[str, np.ndarray]:
    row = smooth_1d(gray[cy, :], smooth_win)
    col = smooth_1d(gray[:, cx], smooth_win)
    return {
        "row_full": row,
        "col_full": col,
        "x+": row[cx:],
        "x-": row[:cx + 1][::-1],
        "y+": col[cy:],
        "y-": col[:cy + 1][::-1],
    }


def compute_axis_parameters_analysis(
    gray: np.ndarray,
    cx: int,
    cy: int,
    smooth_win: int = 5,
    start_idx: int = 2,
) -> Dict[str, Any]:
    profiles = split_profiles_1line_analysis(gray, cx, cy, smooth_win=smooth_win)
    I_center = float(gray[cy, cx])

    directional: Dict[str, Dict[str, Any]] = {}
    for key in ["x+", "x-", "y+", "y-"]:
        prof = profiles[key]
        min_idx, max_idx = find_first_min_then_first_max(prof, start_idx=start_idx)

        if min_idx is None:
            directional[key] = {
                "I_center": I_center, "I_min": np.nan, "I_max": np.nan,
                "PPD": np.nan, "MMD": np.nan, "WCM": np.nan, "WSM": np.nan,
                "min_idx": None, "max_idx": None,
            }
            continue

        I_min = float(prof[min_idx])
        if max_idx is None:
            I_max = np.nan
            MMD = np.nan
            WSM = np.nan
        else:
            I_max = float(prof[max_idx])
            MMD = I_max - I_min
            WSM = float(max_idx)

        directional[key] = {
            "I_center": I_center,
            "I_min": I_min,
            "I_max": I_max,
            "PPD": I_center - I_min,
            "MMD": MMD,
            "WCM": float(min_idx),
            "WSM": WSM,
            "min_idx": int(min_idx),
            "max_idx": (None if max_idx is None else int(max_idx)),
        }

    ppd_vals = np.array([directional[k]["PPD"] for k in ["x+", "x-", "y+", "y-"]], dtype=np.float32)
    mmd_vals = np.array([directional[k]["MMD"] for k in ["x+", "x-", "y+", "y-"]], dtype=np.float32)

    return {
        "center_x": int(cx),
        "center_y": int(cy),
        "CMV": float(I_center),
        "directional": directional,
        "PPD_avg": float(np.nanmean(ppd_vals)),
        "PPD_std": float(np.nanstd(ppd_vals)),
        "MMD_avg": float(np.nanmean(mmd_vals)),
        "MMD_std": float(np.nanstd(mmd_vals)),
        "WCM_x": float(np.nansum([directional["x+"]["WCM"], directional["x-"]["WCM"]])),
        "WCM_y": float(np.nansum([directional["y+"]["WCM"], directional["y-"]["WCM"]])),
        "WSM_x": float(np.nansum([directional["x+"]["WSM"], directional["x-"]["WSM"]])),
        "WSM_y": float(np.nansum([directional["y+"]["WSM"], directional["y-"]["WSM"]])),
        "profiles": profiles,
    }


def radial_profile_analysis(gray: np.ndarray, cx: int, cy: int, max_r: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if max_r is None:
        max_r = int(np.floor(min(cx, cy, w - 1 - cx, h - 1 - cy)))

    r_int = np.floor(rr).astype(np.int32)
    sums = np.bincount(r_int.ravel(), weights=gray.ravel(), minlength=max_r + 1)
    counts = np.bincount(r_int.ravel(), minlength=max_r + 1)
    prof = sums / np.maximum(counts, 1)
    return prof[:max_r + 1], counts[:max_r + 1]


def estimate_pattern_end_radius_analysis(
    gray: np.ndarray,
    cx: int,
    cy: int,
    smooth_win: int = 5,
    stable_len: int = 4,
    bg_fraction: float = 0.25,
    sigma_factor: float = 1.5,
) -> Dict[str, Any]:
    rp, _ = radial_profile_analysis(gray, cx, cy, None)
    rp_s = smooth_1d(rp, smooth_win)

    n = len(rp_s)
    start_bg = max(3, int(n * (1.0 - bg_fraction)))
    outer = rp_s[start_bg:]
    outer_med = float(np.median(outer))
    outer_std = robust_mad_std(outer)
    thr = max(outer_std * sigma_factor, 0.5)

    r_end = None
    for i in range(3, n - stable_len):
        seg = rp_s[i:i + stable_len]
        if np.all(np.abs(seg - outer_med) <= thr):
            r_end = i
            break

    if r_end is None:
        r_end = int(n * 0.55)

    return {
        "pattern_end_radius": float(r_end),
        "outer_profile_median": outer_med,
        "outer_profile_std": outer_std,
        "outer_profile_thr": thr,
        "radial_profile_raw": rp,
        "radial_profile_smooth": rp_s,
    }


def box_local_std_analysis(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize < 1:
        ksize = 1
    mean = cv2.blur(gray, (ksize, ksize))
    mean2 = cv2.blur(gray * gray, (ksize, ksize))
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var)


def build_background_mask_analysis(
    gray: np.ndarray,
    cx: int,
    cy: int,
    pattern_end_radius: float,
    bg_margin: int = 3,
    outer_border_margin: int = 2,
    local_std_ksize: int = 5,
    quantile_keep: float = 0.60,
) -> Dict[str, Any]:
    h, w = gray.shape
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    rmax = float(np.floor(min(cx, cy, w - 1 - cx, h - 1 - cy)))
    r_bg_start = min(rmax - 2.0, float(pattern_end_radius) + float(bg_margin))
    r_bg_end = max(r_bg_start + 1.0, rmax - float(outer_border_margin))

    base_mask = (rr >= r_bg_start) & (rr <= r_bg_end)

    border_mask = np.zeros_like(base_mask, dtype=bool)
    border_mask[outer_border_margin:h - outer_border_margin, outer_border_margin:w - outer_border_margin] = True
    base_mask &= border_mask

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lstd = box_local_std_analysis(gray, local_std_ksize)

    if np.sum(base_mask) < 20:
        return {
            "background_mask": base_mask, "base_mask": base_mask,
            "r_bg_start": float(r_bg_start), "r_bg_end": float(r_bg_end),
            "grad_thr": np.nan, "local_std_thr": np.nan,
            "background_pixel_count": int(np.sum(base_mask)),
        }

    grad_vals = grad[base_mask]
    std_vals = lstd[base_mask]
    grad_thr = float(np.quantile(grad_vals, quantile_keep))
    std_thr = float(np.quantile(std_vals, quantile_keep))

    bg_mask = base_mask & (grad <= grad_thr) & (lstd <= std_thr)

    if np.sum(bg_mask) < 20:
        grad_thr = float(np.quantile(grad_vals, 0.80))
        std_thr = float(np.quantile(std_vals, 0.80))
        bg_mask = base_mask & (grad <= grad_thr) & (lstd <= std_thr)

    if np.sum(bg_mask) < 20:
        bg_mask = base_mask.copy()

    return {
        "background_mask": bg_mask,
        "base_mask": base_mask,
        "r_bg_start": float(r_bg_start),
        "r_bg_end": float(r_bg_end),
        "grad_thr": grad_thr,
        "local_std_thr": std_thr,
        "background_pixel_count": int(np.sum(bg_mask)),
    }


def estimate_background_baseline_from_external_pixels_analysis(
    gray: np.ndarray,
    cx: int,
    cy: int,
    pattern_end_radius: float,
    bg_margin: int = 3,
    outer_border_margin: int = 2,
    local_std_ksize: int = 5,
    quantile_keep: float = 0.60,
) -> Dict[str, Any]:
    bg = build_background_mask_analysis(
        gray=gray,
        cx=cx,
        cy=cy,
        pattern_end_radius=pattern_end_radius,
        bg_margin=bg_margin,
        outer_border_margin=outer_border_margin,
        local_std_ksize=local_std_ksize,
        quantile_keep=quantile_keep,
    )
    bg_mask = bg["background_mask"]
    bg_vals = gray[bg_mask]
    if bg_vals.size < 5:
        raise ValueError("baseline 계산용 외부 배경 픽셀이 너무 적습니다.")

    baseline = float(np.median(bg_vals))
    bg_sigma = robust_mad_std(bg_vals)
    tol = max(0.5, 1.5 * bg_sigma)

    return {
        "background_mask": bg_mask,
        "base_mask": bg["base_mask"],
        "r_bg_start": bg["r_bg_start"],
        "r_bg_end": bg["r_bg_end"],
        "grad_thr": bg["grad_thr"],
        "local_std_thr": bg["local_std_thr"],
        "background_pixel_count": bg["background_pixel_count"],
        "baseline": baseline,
        "baseline_sigma": float(bg_sigma),
        "baseline_tol": float(tol),
    }


def find_baseline_crossings_analysis(radial_prof: np.ndarray, baseline: float, smooth_win: int = 5, min_start: int = 1) -> Dict[str, Any]:
    rp_s = smooth_1d(radial_prof, smooth_win)
    diff = rp_s - baseline
    crossings: List[float] = []

    for i in range(max(0, min_start), len(diff) - 1):
        if diff[i] == 0:
            crossings.append(float(i))
        elif diff[i] * diff[i + 1] < 0:
            t = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
            crossings.append(float(i + t))

    if len(crossings) < 4:
        raise ValueError("배경 기준 교차점 4개를 찾지 못했습니다.")

    return {
        "radial_profile_smooth": rp_s,
        "cross_1": float(crossings[0]),
        "cross_2": float(crossings[1]),
        "cross_3": float(crossings[2]),
        "cross_4": float(crossings[3]),
        "all_crossings": [float(x) for x in crossings],
    }


def compute_band_pixel_counts_analysis(gray: np.ndarray, cx: int, cy: int, crossings: Dict[str, Any]) -> Dict[str, Any]:
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    c1 = crossings["cross_1"]
    c2 = crossings["cross_2"]
    c3 = crossings["cross_3"]
    c4 = crossings["cross_4"]

    center_bright_mask = rr <= c1
    ring_dark_1_mask = (rr > c1) & (rr <= c2)
    ring_bright_1_mask = (rr > c2) & (rr <= c3)
    ring_dark_2_mask = (rr > c3) & (rr <= c4)

    return {
        "center_bright_mask": center_bright_mask,
        "ring_dark_1_mask": ring_dark_1_mask,
        "ring_bright_1_mask": ring_bright_1_mask,
        "ring_dark_2_mask": ring_dark_2_mask,
        "center_bright_pixel_count": int(np.sum(center_bright_mask)),
        "ring_dark_1_pixel_count": int(np.sum(ring_dark_1_mask)),
        "ring_bright_1_pixel_count": int(np.sum(ring_bright_1_mask)),
        "ring_dark_2_pixel_count": int(np.sum(ring_dark_2_mask)),
    }


def draw_axis_points_analysis(gray: np.ndarray, axis_params: Dict[str, Any]) -> np.ndarray:
    vis = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cx = axis_params["center_x"]
    cy = axis_params["center_y"]

    cv2.line(vis, (0, cy), (vis.shape[1] - 1, cy), (255, 255, 255), 1)
    cv2.line(vis, (cx, 0), (cx, vis.shape[0] - 1), (255, 255, 255), 1)
    cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)

    d = axis_params["directional"]
    for key, dx, dy in [("x+", 1, 0), ("x-", -1, 0), ("y+", 0, 1), ("y-", 0, -1)]:
        if d[key]["min_idx"] is not None:
            cv2.circle(vis, (cx + dx * d[key]["min_idx"], cy + dy * d[key]["min_idx"]), 2, (255, 0, 0), -1)
        if d[key]["max_idx"] is not None:
            cv2.circle(vis, (cx + dx * d[key]["max_idx"], cy + dy * d[key]["max_idx"]), 2, (0, 165, 255), -1)
    return vis


def draw_region_overlay_analysis(gray: np.ndarray, band_info: Optional[Dict[str, Any]], bg_info: Optional[Dict[str, Any]], cx: int, cy: int) -> np.ndarray:
    base = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    if bg_info is not None and "background_mask" in bg_info:
        bg_mask = bg_info["background_mask"]
        overlay[bg_mask] = (0.7 * overlay[bg_mask] + 0.3 * np.array([255, 0, 255])).astype(np.uint8)

    if band_info:
        for mask_key, color in [
            ("center_bright_mask", np.array([0, 255, 255])),
            ("ring_dark_1_mask", np.array([255, 0, 0])),
            ("ring_bright_1_mask", np.array([0, 165, 255])),
            ("ring_dark_2_mask", np.array([0, 0, 255])),
        ]:
            if mask_key in band_info:
                overlay[band_info[mask_key]] = (0.55 * overlay[band_info[mask_key]] + 0.45 * color).astype(np.uint8)
        for key in ["cross_1", "cross_2", "cross_3", "cross_4"]:
            if key in band_info and np.isfinite(band_info[key]):
                r = int(round(float(band_info[key])))
                cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 1)

    cv2.circle(overlay, (cx, cy), 2, (0, 255, 255), -1)
    return overlay


def save_analysis_figure_analysis(
    out_png: str,
    gray_raw: np.ndarray,
    bg_info: Optional[Dict[str, Any]],
    axis_params: Dict[str, Any],
    band_info: Optional[Dict[str, Any]],
    pattern_est: Optional[Dict[str, Any]],
) -> None:
    cx = axis_params["center_x"]
    cy = axis_params["center_y"]

    raw_vis = cv2.cvtColor(np.clip(gray_raw, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    region_vis = draw_region_overlay_analysis(gray_raw, band_info, bg_info, cx, cy)
    axis_vis = draw_axis_points_analysis(gray_raw, axis_params)

    baseline = np.nan
    if bg_info is not None:
        baseline = bg_info.get("baseline", np.nan)

    rp_raw_s = None
    if pattern_est is not None:
        rp_raw_s = pattern_est["radial_profile_smooth"]
    else:
        rp_raw_s = smooth_1d(radial_profile_analysis(gray_raw, cx, cy, None)[0], 5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(raw_vis[..., ::-1], cmap="gray")
    axes[0, 0].set_title("raw crop")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(region_vis[..., ::-1])
    axes[0, 1].set_title("bands + selected external bg pixels")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(axis_vis[..., ::-1])
    axes[1, 0].set_title("1-line axis points")
    axes[1, 0].axis("off")

    row = axis_params["profiles"]["row_full"]
    col = axis_params["profiles"]["col_full"]
    axes[1, 1].plot(row, label="row(y=cy)")
    axes[1, 1].plot(col, label="col(x=cx)")
    axes[1, 1].plot(rp_raw_s, label="radial smooth")
    if np.isfinite(baseline):
        axes[1, 1].axhline(baseline, color="m", linestyle="--", label=f"baseline={baseline:.2f}")
    if pattern_est is not None:
        axes[1, 1].axvline(pattern_est["pattern_end_radius"], color="c", linestyle=":", label="pattern end")
    if band_info:
        for key in ["cross_1", "cross_2", "cross_3", "cross_4"]:
            if key in band_info and np.isfinite(band_info[key]):
                axes[1, 1].axvline(band_info[key], color="k", linestyle="--", alpha=0.4)
    axes[1, 1].set_title("profiles")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    ensure_dir(str(Path(out_png).parent))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def empty_parameter_result(image_path: str, overlay_path: str, status: str = "error", baseline_status: str = "error", band_status: str = "error", baseline_error: str = "", band_error: str = "") -> Dict[str, Any]:
    row = {k: np.nan for k in PARAMETER_SUMMARY_FIELDS}
    row.update({
        "image_name": os.path.basename(image_path),
        "image_path": normalize_path(image_path),
        "status": status,
        "baseline_status": baseline_status,
        "band_status": band_status,
        "baseline_error": baseline_error,
        "band_error": band_error,
        "analysis_overlay_path": overlay_path,
    })
    return row


def analyze_single_crop(
    image_path: str,
    overlay_dir: str,
    smooth_win: int = 5,
    start_idx: int = 2,
    blur_ksize: int = 5,
    bg_margin: int = 3,
    outer_border_margin: int = 2,
    local_std_ksize: int = 5,
    quantile_keep: float = 0.60,
) -> Dict[str, Any]:
    stem = Path(image_path).stem
    overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")

    try:
        gray_raw = read_gray_for_analysis(image_path)
    except Exception as e:
        return empty_parameter_result(image_path, overlay_path, status="error", baseline_error=str(e), band_error=str(e))

    cx, cy = find_center_1line_analysis(gray_raw, blur_ksize=blur_ksize, smooth_win=smooth_win, n_iter=3)
    axis_params = compute_axis_parameters_analysis(gray_raw, cx, cy, smooth_win=smooth_win, start_idx=start_idx)

    baseline_status = "ok"
    band_status = "ok"
    baseline_error = ""
    band_error = ""

    pattern_est = None
    bg_info = None
    band_info = {}

    try:
        pattern_est = estimate_pattern_end_radius_analysis(gray_raw, cx, cy, smooth_win=smooth_win)
        bg_info = estimate_background_baseline_from_external_pixels_analysis(
            gray=gray_raw, cx=cx, cy=cy,
            pattern_end_radius=pattern_est["pattern_end_radius"],
            bg_margin=bg_margin, outer_border_margin=outer_border_margin,
            local_std_ksize=local_std_ksize, quantile_keep=quantile_keep,
        )
    except Exception as e:
        baseline_status = "error"
        baseline_error = str(e)

    if baseline_status == "ok":
        try:
            rp_raw, _ = radial_profile_analysis(gray_raw, cx, cy, None)
            crossings = find_baseline_crossings_analysis(
                radial_prof=rp_raw, baseline=bg_info["baseline"], smooth_win=smooth_win, min_start=max(1, start_idx)
            )
            band_info = compute_band_pixel_counts_analysis(gray_raw, cx, cy, crossings)
            band_info.update(crossings)
        except Exception as e:
            band_status = "error"
            band_error = str(e)

    try:
        save_analysis_figure_analysis(
            out_png=overlay_path,
            gray_raw=gray_raw,
            bg_info=bg_info,
            axis_params=axis_params,
            band_info=(band_info if band_status == "ok" else None),
            pattern_est=pattern_est,
        )
    except Exception:
        pass

    d = axis_params["directional"]
    result = {
        "image_name": os.path.basename(image_path),
        "image_path": normalize_path(image_path),
        "status": "ok" if (baseline_status == "ok" and band_status == "ok") else "partial",
        "baseline_status": baseline_status,
        "band_status": band_status,
        "baseline_error": baseline_error,
        "band_error": band_error,
        "analysis_overlay_path": overlay_path,

        "center_x": axis_params["center_x"],
        "center_y": axis_params["center_y"],
        "CMV": axis_params["CMV"],

        "PPD_x_plus": d["x+"]["PPD"],
        "PPD_x_minus": d["x-"]["PPD"],
        "PPD_y_plus": d["y+"]["PPD"],
        "PPD_y_minus": d["y-"]["PPD"],
        "PPD_avg": axis_params["PPD_avg"],
        "PPD_std": axis_params["PPD_std"],

        "MMD_x_plus": d["x+"]["MMD"],
        "MMD_x_minus": d["x-"]["MMD"],
        "MMD_y_plus": d["y+"]["MMD"],
        "MMD_y_minus": d["y-"]["MMD"],
        "MMD_avg": axis_params["MMD_avg"],
        "MMD_std": axis_params["MMD_std"],

        "WCM_x_plus": d["x+"]["WCM"],
        "WCM_x_minus": d["x-"]["WCM"],
        "WCM_y_plus": d["y+"]["WCM"],
        "WCM_y_minus": d["y-"]["WCM"],
        "WCM_x": axis_params["WCM_x"],
        "WCM_y": axis_params["WCM_y"],

        "WSM_x_plus": d["x+"]["WSM"],
        "WSM_x_minus": d["x-"]["WSM"],
        "WSM_y_plus": d["y+"]["WSM"],
        "WSM_y_minus": d["y-"]["WSM"],
        "WSM_x": axis_params["WSM_x"],
        "WSM_y": axis_params["WSM_y"],

        "pattern_end_radius": np.nan,
        "background_pixel_count": np.nan,
        "background_r_start": np.nan,
        "background_r_end": np.nan,
        "baseline_external_pixels": np.nan,
        "baseline_sigma": np.nan,
        "baseline_tol": np.nan,
        "grad_thr": np.nan,
        "local_std_thr": np.nan,

        "cross_1": np.nan,
        "cross_2": np.nan,
        "cross_3": np.nan,
        "cross_4": np.nan,
        "center_bright_pixel_count": np.nan,
        "ring_dark_1_pixel_count": np.nan,
        "ring_bright_1_pixel_count": np.nan,
        "ring_dark_2_pixel_count": np.nan,
    }

    if pattern_est is not None:
        result["pattern_end_radius"] = pattern_est["pattern_end_radius"]

    if bg_info is not None:
        result.update({
            "background_pixel_count": bg_info["background_pixel_count"],
            "background_r_start": bg_info["r_bg_start"],
            "background_r_end": bg_info["r_bg_end"],
            "baseline_external_pixels": bg_info["baseline"],
            "baseline_sigma": bg_info["baseline_sigma"],
            "baseline_tol": bg_info["baseline_tol"],
            "grad_thr": bg_info["grad_thr"],
            "local_std_thr": bg_info["local_std_thr"],
        })

    if band_status == "ok":
        result.update({
            "cross_1": band_info["cross_1"],
            "cross_2": band_info["cross_2"],
            "cross_3": band_info["cross_3"],
            "cross_4": band_info["cross_4"],
            "center_bright_pixel_count": band_info["center_bright_pixel_count"],
            "ring_dark_1_pixel_count": band_info["ring_dark_1_pixel_count"],
            "ring_bright_1_pixel_count": band_info["ring_bright_1_pixel_count"],
            "ring_dark_2_pixel_count": band_info["ring_dark_2_pixel_count"],
        })

    return result


def perform_parameter_analysis(
    crops_dir: str,
    output_dir: str,
    smooth_win: int = 5,
    start_idx: int = 2,
    blur_ksize: int = 5,
    bg_margin: int = 3,
    outer_border_margin: int = 2,
    local_std_ksize: int = 5,
    quantile_keep: float = 0.60,
) -> Dict[str, Any]:
    crops_dir = normalize_path(crops_dir)
    output_dir = normalize_path(output_dir)

    param_dir = os.path.join(output_dir, "parameter_analysis")
    overlay_dir = os.path.join(param_dir, "cell_overlays")
    ensure_dir(param_dir)
    ensure_dir(overlay_dir)

    crop_files = sorted(
        [
            os.path.join(crops_dir, name)
            for name in os.listdir(crops_dir)
            if os.path.isfile(os.path.join(crops_dir, name)) and Path(name).suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ]
    )

    rows: List[Dict[str, Any]] = []
    for crop_path in crop_files:
        row = analyze_single_crop(
            image_path=crop_path,
            overlay_dir=overlay_dir,
            smooth_win=smooth_win,
            start_idx=start_idx,
            blur_ksize=blur_ksize,
            bg_margin=bg_margin,
            outer_border_margin=outer_border_margin,
            local_std_ksize=local_std_ksize,
            quantile_keep=quantile_keep,
        )
        rows.append(row)

    csv_path = os.path.join(param_dir, "cell_parameter_summary_bgpixels_only.csv")
    json_path = os.path.join(param_dir, "cell_parameter_summary_bgpixels_only.json")
    save_summary_csv(csv_path, rows)
    save_summary_json(json_path, rows)

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    partial_count = sum(1 for r in rows if r.get("status") == "partial")
    error_count = sum(1 for r in rows if r.get("status") == "error")

    return {
        "parameter_analysis_dir": param_dir,
        "parameter_overlay_dir": overlay_dir,
        "parameter_summary_csv": csv_path,
        "parameter_summary_json": json_path,
        "parameter_row_count": len(rows),
        "parameter_ok_count": ok_count,
        "parameter_partial_count": partial_count,
        "parameter_error_count": error_count,
    }



def save_processed_result(result: Dict[str, Any], output_dir: str, crop_size: int) -> Dict[str, Any]:
    """
    현재 미리보기/편집 결과(last_result)를 그대로 저장한다.
    - 사용자가 이동/삭제한 detections 기준으로 crop 저장
    - 기존 crop 파일은 먼저 비워서 삭제된 박스의 잔여 파일이 남지 않게 함
    - 저장용 annotated는 선택 강조 없이 생성
    """
    if result is None:
        raise ValueError("저장할 결과가 없습니다. 먼저 Preview를 실행하세요.")

    output_dir = normalize_path(output_dir)
    if not output_dir:
        raise ValueError("출력 폴더가 비어 있습니다.")

    ensure_dir(output_dir)
    crops_dir = os.path.join(output_dir, "crops")
    ensure_dir(crops_dir)

    # 기존 crop 삭제
    for name in os.listdir(crops_dir):
        fpath = os.path.join(crops_dir, name)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    original_image = result["original_image"]
    detections = [Detection(d.idx, d.cx, d.cy, d.x1, d.y1, d.x2, d.y2, d.area) for d in result.get("detections", [])]
    valid_xmax = result.get("valid_xmax", original_image.shape[1])
    input_image = result.get("input_image", "")

    # 번호 재정렬 후 crop 저장
    for i, det in enumerate(detections, start=1):
        det.idx = i
        crop = crop_square(original_image, det, crop_size)
        save_image(os.path.join(crops_dir, f"crop_{i:03d}.png"), crop)

    annotated = draw_detections(original_image, detections, valid_xmax=valid_xmax, selected_idx=None)
    overview = make_side_by_side(original_image, annotated)

    original_path = os.path.join(output_dir, "original.png")
    mask_path = os.path.join(output_dir, "mask.png")
    annotated_path = os.path.join(output_dir, "annotated.png")
    overview_path = os.path.join(output_dir, "overview_original_vs_annotated.png")
    csv_path = os.path.join(output_dir, "detections.csv")

    save_image(original_path, original_image)

    # mask_image는 last_result에 BGR 형태로 저장되어 있으므로 그대로 저장
    if "mask_image" in result and result["mask_image"] is not None:
        save_image(mask_path, result["mask_image"])

    save_image(annotated_path, annotated)
    save_image(overview_path, overview)
    save_detections_csv(csv_path, detections)

    parameter_info = perform_parameter_analysis(
        crops_dir=crops_dir,
        output_dir=output_dir,
        smooth_win=5,
        start_idx=2,
        blur_ksize=5,
        bg_margin=3,
        outer_border_margin=2,
        local_std_ksize=5,
        quantile_keep=0.60,
    )

    new_result = dict(result)
    new_result.update({
        "input_image": input_image,
        "output_dir": output_dir,
        "crop_count": len(detections),
        "original_path": original_path,
        "mask_path": mask_path,
        "annotated_path": annotated_path,
        "overview_path": overview_path,
        "csv_path": csv_path,
        "crops_dir": crops_dir,
        "annotated_image": annotated,
        "overview_image": overview,
        "detections": detections,
        "valid_xmax": valid_xmax,
        **parameter_info,
    })
    return new_result


class DiffractionCropApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LSIT / Lens-free Diffraction Crop Tool")
        self.root.geometry("1500x920")
        self.root.minsize(1280, 760)

        self.canvas_photo = None
        self.last_result = None
        self.preview_items: List[Tuple[str, str, np.ndarray]] = []
        self.preview_index = 0
        self.zoom_scale = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 10.0
        self.last_params: Optional[Dict[str, Any]] = None
        self.selected_detection_idx: Optional[int] = None
        self.dragging_box = False
        self.drag_offset = (0.0, 0.0)

        self._build_variables()
        self._build_ui()
        self._bind_keys()

    def _build_variables(self):
        self.image_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="output_diffraction_gui")

        self.crop_size_var = tk.StringVar(value="30")
        self.min_area_var = tk.StringVar(value="20")
        self.max_area_var = tk.StringVar(value="0")
        self.blur_var = tk.StringVar(value="5")
        self.thresh_var = tk.StringVar(value="adaptive")

        self.adaptive_block_var = tk.StringVar(value="31")
        self.adaptive_c_var = tk.StringVar(value="7")

        self.morph_kernel_var = tk.StringVar(value="3")
        self.open_iter_var = tk.StringVar(value="1")
        self.close_iter_var = tk.StringVar(value="1")
        self.nms_overlap_var = tk.StringVar(value="0.30")

        self.remove_right_saturation_var = tk.BooleanVar(value=True)
        self.sat_thresh_var = tk.StringVar(value="250")
        self.sat_col_ratio_var = tk.StringVar(value="0.90")
        self.sat_min_band_width_var = tk.StringVar(value="2")
        self.border_margin_var = tk.StringVar(value="2")

        self.reject_bad_right_crop_var = tk.BooleanVar(value=True)
        self.crop_right_edge_width_var = tk.StringVar(value="3")
        self.crop_sat_ratio_var = tk.StringVar(value="0.85")

        self.status_var = tk.StringVar(value="이미지를 선택하고 Preview 또는 Run & Save를 누르세요.")
        self.preview_title_var = tk.StringVar(value="Preview")
        self.zoom_info_var = tk.StringVar(value="100%")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=430)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # 파일 설정
        file_frame = ttk.LabelFrame(left, text="파일 설정", padding=8)
        file_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(file_frame, text="입력 이미지").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.image_path_var, width=34).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(file_frame, text="찾기", width=8, command=self.browse_image).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Label(file_frame, text="출력 폴더").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=34).grid(
            row=3, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(file_frame, text="찾기", width=8, command=self.browse_output).grid(
            row=3, column=1, sticky="ew"
        )

        file_frame.columnconfigure(0, weight=1)

        # 탭형 설정 영역
        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True, pady=(0, 6))

        tab_basic = ttk.Frame(notebook, padding=8)
        tab_tune = ttk.Frame(notebook, padding=8)
        tab_sat = ttk.Frame(notebook, padding=8)
        tab_help = ttk.Frame(notebook, padding=8)

        notebook.add(tab_basic, text="기본")
        notebook.add(tab_tune, text="튜닝")
        notebook.add(tab_sat, text="포화/필터")
        notebook.add(tab_help, text="가이드")

        self._build_basic_tab(tab_basic)
        self._build_tune_tab(tab_tune)
        self._build_sat_tab(tab_sat)
        self._build_help_tab(tab_help)

        # 실행 버튼
        btn_frame = ttk.LabelFrame(left, text="실행 / 보기", padding=8)
        btn_frame.pack(fill="x", pady=(0, 6))

        row1 = ttk.Frame(btn_frame)
        row1.pack(fill="x", pady=2)
        ttk.Button(row1, text="Preview", command=self.on_preview).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(row1, text="Run & Save", command=self.on_run_save).pack(side="left", fill="x", expand=True, padx=(3, 0))

        row2 = ttk.Frame(btn_frame)
        row2.pack(fill="x", pady=2)
        ttk.Button(row2, text="Original", command=lambda: self.show_preview_key("original_image")).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(row2, text="Annotated", command=lambda: self.show_preview_key("annotated_image")).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(row2, text="선택 박스 삭제", command=self.delete_selected_detection).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # 상태
        status_frame = ttk.LabelFrame(left, text="상태", padding=8)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=390, justify="left").pack(anchor="w")

        # 우측 미리보기
        preview_frame = ttk.LabelFrame(right, text="단일 미리보기", padding=8)
        preview_frame.pack(fill="both", expand=True)

        topbar = ttk.Frame(preview_frame)
        topbar.pack(fill="x", pady=(0, 8))

        ttk.Button(topbar, text="◀ Prev", width=8, command=self.prev_preview).pack(side="left")
        ttk.Button(topbar, text="Next ▶", width=8, command=self.next_preview).pack(side="left", padx=(4, 0))
        ttk.Button(topbar, text="Fit", width=7, command=self.fit_current_preview).pack(side="left", padx=(14, 0))
        ttk.Button(topbar, text="100%", width=7, command=self.reset_zoom_100).pack(side="left", padx=(4, 0))
        ttk.Button(topbar, text="Zoom In", width=9, command=lambda: self.adjust_zoom(1.25)).pack(side="left", padx=(14, 0))
        ttk.Button(topbar, text="Zoom Out", width=9, command=lambda: self.adjust_zoom(0.8)).pack(side="left", padx=(4, 0))

        ttk.Label(topbar, textvariable=self.preview_title_var, font=("Arial", 10, "bold")).pack(side="left", padx=(18, 0))
        ttk.Label(topbar, textvariable=self.zoom_info_var).pack(side="right")

        canvas_wrap = ttk.Frame(preview_frame)
        canvas_wrap.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_wrap, bg="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        vbar = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        vbar.pack(side="right", fill="y")
        hbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        hbar.pack(fill="x")

        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)
        self.canvas.bind("<Control-MouseWheel>", self.on_mousewheel_zoom)
        self.canvas.bind("<Button-4>", self.on_mousewheel_zoom_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_zoom_linux)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def _build_basic_tab(self, parent):
        box1 = ttk.LabelFrame(parent, text="검출 / Crop 설정", padding=8)
        box1.pack(fill="x", pady=(0, 6))
        self._add_labeled_entry(box1, "Crop Size", self.crop_size_var, 0)
        self._add_labeled_entry(box1, "Min Area", self.min_area_var, 1)
        self._add_labeled_entry(box1, "Max Area (0=No Limit)", self.max_area_var, 2)
        self._add_labeled_entry(box1, "Gaussian Blur", self.blur_var, 3)

        box2 = ttk.LabelFrame(parent, text="Threshold", padding=8)
        box2.pack(fill="x")
        ttk.Label(box2, text="Threshold Method").grid(row=0, column=0, sticky="w", pady=2)
        thresh_combo = ttk.Combobox(
            box2,
            textvariable=self.thresh_var,
            values=["adaptive", "otsu"],
            state="readonly",
            width=14,
        )
        thresh_combo.grid(row=0, column=1, sticky="ew", pady=2)
        box2.columnconfigure(1, weight=1)

    def _build_tune_tab(self, parent):
        upper = ttk.Frame(parent)
        upper.pack(fill="x", pady=(0, 6))

        left_box = ttk.LabelFrame(upper, text="Adaptive", padding=8)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 3))
        self._add_labeled_entry(left_box, "Block Size", self.adaptive_block_var, 0)
        self._add_labeled_entry(left_box, "C", self.adaptive_c_var, 1)

        right_box = ttk.LabelFrame(upper, text="Morphology", padding=8)
        right_box.pack(side="left", fill="both", expand=True, padx=(3, 0))
        self._add_labeled_entry(right_box, "Kernel", self.morph_kernel_var, 0)
        self._add_labeled_entry(right_box, "Open Iter", self.open_iter_var, 1)
        self._add_labeled_entry(right_box, "Close Iter", self.close_iter_var, 2)

        nms_box = ttk.LabelFrame(parent, text="NMS", padding=8)
        nms_box.pack(fill="x")
        self._add_labeled_entry(nms_box, "Overlap Threshold", self.nms_overlap_var, 0)

    def _build_sat_tab(self, parent):
        sat_box = ttk.LabelFrame(parent, text="오른쪽 끝 포화 밴드 제거", padding=8)
        sat_box.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(
            sat_box,
            text="Remove Right Saturation",
            variable=self.remove_right_saturation_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._add_labeled_entry(sat_box, "Sat Threshold", self.sat_thresh_var, 1)
        self._add_labeled_entry(sat_box, "Column Ratio", self.sat_col_ratio_var, 2)
        self._add_labeled_entry(sat_box, "Min Band Width", self.sat_min_band_width_var, 3)
        self._add_labeled_entry(sat_box, "Border Margin", self.border_margin_var, 4)

        crop_box = ttk.LabelFrame(parent, text="Crop 저장 전 추가 필터", padding=8)
        crop_box.pack(fill="x")
        ttk.Checkbutton(
            crop_box,
            text="Reject crop if right edge is saturated",
            variable=self.reject_bad_right_crop_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._add_labeled_entry(crop_box, "Edge Width", self.crop_right_edge_width_var, 1)
        self._add_labeled_entry(crop_box, "Crop Sat Ratio", self.crop_sat_ratio_var, 2)

    def _build_help_tab(self, parent):
        help_text = (
            "빠른 시작 권장값\n"
            "• Crop Size = 30\n"
            "• Min Area = 20\n"
            "• Blur = 5\n"
            "• Threshold = adaptive\n"
            "• Block Size = 31, C = 7\n"
            "• Kernel = 3, Open = 1, Close = 1\n"
            "• NMS = 0.30\n"
            "• Right Saturation 제거 = ON\n"
            "• Sat Threshold = 250, Column Ratio = 0.90\n\n"
            "튜닝 팁\n"
            "• 검출이 너무 많으면: Min Area↑, Blur↑, C↑\n"
            "• 검출이 너무 적으면: Min Area↓, Blur↓, C↓\n"
            "• 오른쪽 끝 과검출이면: Sat Threshold↓, Column Ratio↓, Border Margin↑\n\n"
            "박스 편집\n"
            "• Annotated 화면에서 박스를 클릭하면 선택\n"
            "• 선택한 박스를 드래그하면 위치 수정\n"
            "• Delete 키 또는 우클릭으로 박스 삭제"
        )
        ttk.Label(parent, text=help_text, justify="left").pack(anchor="w", fill="x")

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.prev_preview())
        self.root.bind("<Right>", lambda e: self.next_preview())
        self.root.bind("<plus>", lambda e: self.adjust_zoom(1.25))
        self.root.bind("<minus>", lambda e: self.adjust_zoom(0.8))
        self.root.bind("<KeyPress-equal>", lambda e: self.adjust_zoom(1.25))
        self.root.bind("<Control-0>", lambda e: self.fit_current_preview())
        self.root.bind("<Control-1>", lambda e: self.reset_zoom_100())
        self.root.bind("<Delete>", lambda e: self.delete_selected_detection())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected_detection())

    def _add_labeled_entry(self, parent, text, var, row):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", pady=2, padx=(6, 0))
        parent.columnconfigure(1, weight=1)

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.image_path_var.set(normalize_path(path))

    def browse_output(self):
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if path:
            self.output_dir_var.set(normalize_path(path))

    def get_params(self) -> Dict[str, Any]:
        crop_size = int(self.crop_size_var.get())
        min_area = int(self.min_area_var.get())
        max_area_input = int(self.max_area_var.get())
        max_area = None if max_area_input == 0 else max_area_input
        blur = int(self.blur_var.get())
        thresh_method = self.thresh_var.get().strip()
        adaptive_block_size = int(self.adaptive_block_var.get())
        adaptive_c = int(self.adaptive_c_var.get())
        morph_kernel = int(self.morph_kernel_var.get())
        open_iter = int(self.open_iter_var.get())
        close_iter = int(self.close_iter_var.get())
        nms_overlap = float(self.nms_overlap_var.get())

        remove_right_saturation = bool(self.remove_right_saturation_var.get())
        sat_thresh = int(self.sat_thresh_var.get())
        sat_col_ratio = float(self.sat_col_ratio_var.get())
        sat_min_band_width = int(self.sat_min_band_width_var.get())
        border_margin = int(self.border_margin_var.get())

        reject_bad_right_crop = bool(self.reject_bad_right_crop_var.get())
        crop_right_edge_width = int(self.crop_right_edge_width_var.get())
        crop_sat_ratio = float(self.crop_sat_ratio_var.get())

        if crop_size <= 0:
            raise ValueError("Crop Size는 1 이상이어야 합니다.")
        if min_area < 0:
            raise ValueError("Min Area는 0 이상이어야 합니다.")
        if max_area is not None and max_area < min_area:
            raise ValueError("Max Area는 Min Area보다 크거나 같아야 합니다.")
        if not (0.0 <= nms_overlap <= 1.0):
            raise ValueError("NMS Overlap Threshold는 0.0 ~ 1.0 범위여야 합니다.")
        if not (0.0 <= sat_col_ratio <= 1.0):
            raise ValueError("Column Ratio Threshold는 0.0 ~ 1.0 범위여야 합니다.")
        if not (0.0 <= crop_sat_ratio <= 1.0):
            raise ValueError("Crop Saturation Ratio는 0.0 ~ 1.0 범위여야 합니다.")
        if sat_thresh < 0 or sat_thresh > 255:
            raise ValueError("Saturation Threshold는 0~255 범위여야 합니다.")

        return {
            "image_path": normalize_path(self.image_path_var.get()),
            "output_dir": normalize_path(self.output_dir_var.get()),
            "crop_size": crop_size,
            "min_area": min_area,
            "max_area": max_area,
            "blur_ksize": blur,
            "thresh_method": thresh_method,
            "adaptive_block_size": adaptive_block_size,
            "adaptive_c": adaptive_c,
            "morph_kernel": morph_kernel,
            "open_iter": open_iter,
            "close_iter": close_iter,
            "nms_overlap": nms_overlap,
            "remove_right_saturation": remove_right_saturation,
            "sat_thresh": sat_thresh,
            "sat_col_ratio": sat_col_ratio,
            "sat_min_band_width": sat_min_band_width,
            "border_margin": border_margin,
            "reject_bad_right_crop": reject_bad_right_crop,
            "crop_right_edge_width": crop_right_edge_width,
            "crop_sat_ratio": crop_sat_ratio,
        }

    def _set_preview_items_from_result(self, result: Dict[str, Any]):
        self.preview_items = [
            ("original_image", "Original", result["original_image"]),
            ("annotated_image", "Annotated", result["annotated_image"]),
        ]
        self.preview_index = min(self.preview_index, max(0, len(self.preview_items) - 1))

    def _current_preview_item(self) -> Optional[Tuple[str, str, np.ndarray]]:
        if not self.preview_items:
            return None
        return self.preview_items[self.preview_index]

    def _render_current_preview(self):
        if not PIL_AVAILABLE:
            self.status_var.set("Pillow가 없어서 GUI 미리보기를 표시할 수 없습니다. pip install pillow 후 사용하세요.")
            return

        item = self._current_preview_item()
        if item is None:
            self.canvas.delete("all")
            self.preview_title_var.set("Preview")
            self.zoom_info_var.set("100%")
            return

        _, title, img_bgr = item
        self.preview_title_var.set(f"{title}  ({self.preview_index + 1}/{len(self.preview_items)})")

        h, w = img_bgr.shape[:2]
        disp_w = max(1, int(w * self.zoom_scale))
        disp_h = max(1, int(h * self.zoom_scale))
        disp = cv2.resize(
            img_bgr,
            (disp_w, disp_h),
            interpolation=cv2.INTER_LINEAR if self.zoom_scale >= 1.0 else cv2.INTER_AREA,
        )

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.canvas_photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        self.zoom_info_var.set(f"{self.zoom_scale * 100:.0f}%   {disp_w} x {disp_h}")

    def fit_current_preview(self):
        item = self._current_preview_item()
        if item is None:
            return
        _, _, img_bgr = item
        h, w = img_bgr.shape[:2]

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if canvas_w <= 1 or canvas_h <= 1:
            self.root.update_idletasks()
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())

        self.zoom_scale = min(canvas_w / w, canvas_h / h)
        self.zoom_scale = min(max(self.zoom_scale, self.zoom_min), self.zoom_max)
        self._render_current_preview()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def reset_zoom_100(self):
        if self._current_preview_item() is None:
            return
        self.zoom_scale = 1.0
        self._render_current_preview()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def adjust_zoom(self, factor: float):
        if self._current_preview_item() is None:
            return
        new_zoom = self.zoom_scale * factor
        new_zoom = min(max(new_zoom, self.zoom_min), self.zoom_max)
        if abs(new_zoom - self.zoom_scale) < 1e-9:
            return
        self.zoom_scale = new_zoom
        self._render_current_preview()

    def prev_preview(self):
        if not self.preview_items:
            return
        self.preview_index = (self.preview_index - 1) % len(self.preview_items)
        self.fit_current_preview()

    def next_preview(self):
        if not self.preview_items:
            return
        self.preview_index = (self.preview_index + 1) % len(self.preview_items)
        self.fit_current_preview()

    def show_preview_key(self, key: str):
        if self.last_result is None:
            self.on_preview()
            if self.last_result is None:
                return
        for i, item in enumerate(self.preview_items):
            if item[0] == key:
                self.preview_index = i
                self.fit_current_preview()
                return

    def _current_preview_key(self) -> Optional[str]:
        item = self._current_preview_item()
        return None if item is None else item[0]

    def _canvas_to_image_xy(self, event) -> Tuple[int, int]:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return int(round(x / max(self.zoom_scale, 1e-9))), int(round(y / max(self.zoom_scale, 1e-9)))

    def _find_detection_index_at(self, img_x: int, img_y: int) -> Optional[int]:
        if self.last_result is None:
            return None
        detections = self.last_result.get("detections", [])
        for idx in range(len(detections) - 1, -1, -1):
            det = detections[idx]
            if det.x1 <= img_x <= det.x2 and det.y1 <= img_y <= det.y2:
                return idx
        return None

    def _update_detection_box_from_center(self, det: Detection, cx: int, cy: int, crop_size: int, img_w: int, img_h: int):
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img_w, x1 + crop_size)
        y2 = min(img_h, y1 + crop_size)
        x1 = max(0, x2 - crop_size)
        y1 = max(0, y2 - crop_size)
        det.x1, det.y1, det.x2, det.y2 = int(x1), int(y1), int(x2), int(y2)
        det.cx = int((det.x1 + det.x2) // 2)
        det.cy = int((det.y1 + det.y2) // 2)

    def _refresh_result_images(self):
        if self.last_result is None:
            return
        self.last_result["annotated_image"] = draw_detections(
            self.last_result["original_image"],
            self.last_result["detections"],
            valid_xmax=self.last_result.get("valid_xmax"),
            selected_idx=self.selected_detection_idx,
        )
        self.last_result["overview_image"] = make_side_by_side(
            self.last_result["original_image"],
            self.last_result["annotated_image"],
        )
        self._set_preview_items_from_result(self.last_result)

    def _renumber_detections(self):
        if self.last_result is None:
            return
        for i, det in enumerate(self.last_result.get("detections", []), start=1):
            det.idx = i

    def delete_selected_detection(self):
        if self.last_result is None or self.selected_detection_idx is None:
            return
        detections = self.last_result.get("detections", [])
        if not (0 <= self.selected_detection_idx < len(detections)):
            self.selected_detection_idx = None
            return
        detections.pop(self.selected_detection_idx)
        if not detections:
            self.selected_detection_idx = None
        else:
            self.selected_detection_idx = min(self.selected_detection_idx, len(detections) - 1)
        self._renumber_detections()
        self._refresh_result_images()
        self._render_current_preview()
        self.status_var.set(f"선택 박스 삭제 완료: 남은 검출 개수 = {len(detections)}")

    def on_canvas_press(self, event):
        if self._current_preview_key() == "annotated_image" and self.last_result is not None:
            img_x, img_y = self._canvas_to_image_xy(event)
            found_idx = self._find_detection_index_at(img_x, img_y)
            self.selected_detection_idx = found_idx
            if found_idx is not None:
                det = self.last_result["detections"][found_idx]
                self.dragging_box = True
                self.drag_offset = (img_x - det.cx, img_y - det.cy)
                self._refresh_result_images()
                self._render_current_preview()
                return
            self._refresh_result_images()
            self._render_current_preview()
        self.canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.dragging_box and self.last_result is not None and self.selected_detection_idx is not None:
            img_x, img_y = self._canvas_to_image_xy(event)
            cx = int(round(img_x - self.drag_offset[0]))
            cy = int(round(img_y - self.drag_offset[1]))
            crop_size = int(self.last_params["crop_size"]) if self.last_params else int(self.crop_size_var.get())
            img_h, img_w = self.last_result["original_image"].shape[:2]
            self._update_detection_box_from_center(
                self.last_result["detections"][self.selected_detection_idx],
                cx,
                cy,
                crop_size,
                img_w,
                img_h,
            )
            self._refresh_result_images()
            self._render_current_preview()
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_release(self, event):
        self.dragging_box = False

    def on_canvas_right_click(self, event):
        if self._current_preview_key() != "annotated_image" or self.last_result is None:
            return
        img_x, img_y = self._canvas_to_image_xy(event)
        found_idx = self._find_detection_index_at(img_x, img_y)
        if found_idx is None:
            return
        self.selected_detection_idx = found_idx
        self.delete_selected_detection()

    def on_mousewheel_zoom(self, event):
        if self._current_preview_item() is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        self.adjust_zoom(factor)

    def on_mousewheel_zoom_linux(self, event):
        if self._current_preview_item() is None:
            return
        factor = 1.1 if event.num == 4 else 0.9
        self.adjust_zoom(factor)

    def on_canvas_configure(self, event):
        if self.canvas_photo is None and self.preview_items:
            self.fit_current_preview()

    def on_preview(self):
        try:
            params = self.get_params()
            if not params["image_path"]:
                raise ValueError("입력 이미지를 선택하세요.")

            image_bgr = read_image(params["image_path"])
            detections, mask, valid_xmax = detect_diffraction_patterns(
                image_bgr=image_bgr,
                crop_size=params["crop_size"],
                min_area=params["min_area"],
                max_area=params["max_area"],
                blur_ksize=params["blur_ksize"],
                thresh_method=params["thresh_method"],
                adaptive_block_size=params["adaptive_block_size"],
                adaptive_c=params["adaptive_c"],
                morph_kernel=params["morph_kernel"],
                open_iter=params["open_iter"],
                close_iter=params["close_iter"],
                nms_overlap=params["nms_overlap"],
                remove_right_saturation=params["remove_right_saturation"],
                sat_thresh=params["sat_thresh"],
                sat_col_ratio=params["sat_col_ratio"],
                sat_min_band_width=params["sat_min_band_width"],
                border_margin=params["border_margin"],
            )

            preview_detections = []
            if params["reject_bad_right_crop"]:
                new_idx = 1
                for det in detections:
                    crop = crop_square(image_bgr, det, params["crop_size"])
                    if is_bad_crop_by_right_saturation(
                        crop_bgr=crop,
                        edge_width=params["crop_right_edge_width"],
                        sat_thresh=params["sat_thresh"],
                        ratio_thresh=params["crop_sat_ratio"],
                    ):
                        continue
                    det.idx = new_idx
                    preview_detections.append(det)
                    new_idx += 1
            else:
                preview_detections = detections

            self.last_result = {
                "original_image": image_bgr,
                "mask_image": mask_to_bgr(mask),
                "annotated_image": image_bgr.copy(),
                "overview_image": image_bgr.copy(),
                "detections": preview_detections,
                "valid_xmax": valid_xmax,
            }
            self.last_params = params.copy()
            self.selected_detection_idx = None
            self._refresh_result_images()
            self.fit_current_preview()
            self.status_var.set(
                f"Preview 완료: 검출 개수 = {len(preview_detections)} / 유효 끝 x = {valid_xmax} / Annotated에서 박스 선택·이동·삭제 가능"
            )

        except Exception as e:
            messagebox.showerror("오류", str(e))
            self.status_var.set(f"오류: {e}")

    def on_run_save(self):
        try:
            params = self.get_params()
            if not params["image_path"]:
                raise ValueError("입력 이미지를 선택하세요.")
            if not params["output_dir"]:
                raise ValueError("출력 폴더를 입력하세요.")

            comparable_keys = [k for k in params.keys() if k != "output_dir"]
            can_save_current = (
                self.last_result is not None
                and self.last_params is not None
                and all(self.last_params.get(k) == params.get(k) for k in comparable_keys)
            )

            if can_save_current:
                result = save_processed_result(self.last_result, params["output_dir"], params["crop_size"])
                self.last_result = result
                self.last_params = params.copy()
                self.selected_detection_idx = None
                self._refresh_result_images()
            else:
                result = process_image(**params)
                self.last_result = result
                self.last_params = params.copy()
                self.selected_detection_idx = None
                self._refresh_result_images()

            self.fit_current_preview()
            self.status_var.set(
                f"저장 완료: crop {result['crop_count']}개 / 유효 끝 x = {result['valid_xmax']}\n저장 위치: {result['output_dir']}"
            )
            messagebox.showinfo(
                "완료",
                f"저장 완료\n\nCrop 개수: {result['crop_count']}\n유효 끝 x: {result['valid_xmax']}\n폴더: {result['output_dir']}"
            )

        except Exception as e:
            messagebox.showerror("오류", str(e))
            self.status_var.set(f"오류: {e}")

def build_argparser():
    parser = argparse.ArgumentParser(description="Lens-free diffraction crop tool")
    parser.add_argument("--image", type=str, default="", help="입력 이미지 경로")
    parser.add_argument("--output", type=str, default="output_diffraction_cli", help="출력 폴더")
    parser.add_argument("--crop-size", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--thresh", type=str, default="adaptive", choices=["adaptive", "otsu"])
    parser.add_argument("--adaptive-block", type=int, default=31)
    parser.add_argument("--adaptive-c", type=int, default=7)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--open-iter", type=int, default=1)
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--nms-overlap", type=float, default=0.30)
    parser.add_argument("--remove-right-saturation", action="store_true")
    parser.add_argument("--sat-thresh", type=int, default=250)
    parser.add_argument("--sat-col-ratio", type=float, default=0.90)
    parser.add_argument("--sat-min-band-width", type=int, default=2)
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--reject-bad-right-crop", action="store_true")
    parser.add_argument("--crop-right-edge-width", type=int, default=3)
    parser.add_argument("--crop-sat-ratio", type=float, default=0.85)
    parser.add_argument("--no-gui", action="store_true", help="GUI 없이 CLI로 실행")
    return parser


def run_cli(args):
    max_area = None if args.max_area == 0 else args.max_area

    result = process_image(
        image_path=args.image,
        output_dir=args.output,
        crop_size=args.crop_size,
        min_area=args.min_area,
        max_area=max_area,
        blur_ksize=args.blur,
        thresh_method=args.thresh,
        adaptive_block_size=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        morph_kernel=args.morph_kernel,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
        nms_overlap=args.nms_overlap,
        remove_right_saturation=args.remove_right_saturation,
        sat_thresh=args.sat_thresh,
        sat_col_ratio=args.sat_col_ratio,
        sat_min_band_width=args.sat_min_band_width,
        border_margin=args.border_margin,
        reject_bad_right_crop=args.reject_bad_right_crop,
        crop_right_edge_width=args.crop_right_edge_width,
        crop_sat_ratio=args.crop_sat_ratio,
    )

    print("[완료]")
    print(f"입력 이미지 : {result['input_image']}")
    print(f"출력 폴더   : {result['output_dir']}")
    print(f"검출 개수   : {result['crop_count']}")
    print(f"유효 끝 x   : {result['valid_xmax']}")
    print(f"overview    : {result['overview_path']}")
    print(f"csv         : {result['csv_path']}")
    print(f"crops       : {result['crops_dir']}")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.no_gui:
        if not args.image:
            raise SystemExit("--no-gui 사용 시 --image 경로가 필요합니다.")
        run_cli(args)
    else:
        root = tk.Tk()
        app = DiffractionCropApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
