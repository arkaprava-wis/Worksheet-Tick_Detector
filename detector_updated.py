"""Dynamic worksheet tick detector.

Pipeline:
1. Illumination normalization
2. Detect four L-bracket anchor marks and perspective-warp the solved photo
3. Align the blank template to the canonical canvas
4. Inverted Otsu threshold + template subtraction
5. Infer question rows from the template's answer-box horizontal rules
6. Score A/B/C checkbox compartments and save JSON/debug images

Important: The input photo must contain four high-contrast L-bracket anchors,
one near each image corner. This is a heuristic baseline; inspect debug outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Update these two names to match files in your samples/ directory.
SOLVED_IMAGE_PATH = SAMPLES_DIR / "solved_clicked_worksheet.jpeg"
BLANK_TEMPLATE_PATH = SAMPLES_DIR / "blank_worksheet.jpeg"

OPTIONS = ("A", "B", "C")

# The four L-brackets must be located within these corner regions.
ANCHOR_CORNER_FRACTION = 0.18
ANCHOR_MIN_COMPONENT_AREA = 35
ANCHOR_MAX_COMPONENT_AREA_FRACTION = 0.025

# Relative x-ranges for each full printed answer box in the canonical template.
# Adjust only if your blank template's layout changes.
OPTION_X_RANGES = {
    "A": (0.585, 0.688),
    "B": (0.716, 0.820),
    "C": (0.847, 0.949),
}
ANSWER_AREA_X_RANGE = (0.58, 0.97)

# Tick scoring: the inner checkbox is the right-hand compartment of each box.
CHECKBOX_LEFT_FRACTION = 0.63
CHECKBOX_RIGHT_MARGIN_FRACTION = 0.035
TICK_DENSITY_MIN = 0.025
TICK_MIN_PIXELS = 8

# Illumination normalization strength.
ILLUMINATION_BLUR_FRACTION = 0.045

# New output naming convention.
OUT_THRESHOLD = OUTPUT_DIR / "adaptive_threshold.png"
OUT_LINE_MAP = OUTPUT_DIR / "horizontal_border_detection.png"
OUT_LINE_OVERLAY = OUTPUT_DIR / "horizontal_border_preview.jpg"
OUT_BOX_PREVIEW = OUTPUT_DIR / "answer_box_detection_preview.jpg"
OUT_TICK_DEBUG = OUTPUT_DIR / "tick_classification_debug.jpg"
OUT_ANSWERS_JSON = OUTPUT_DIR / "detected_student_answers.json"
OUT_SCORES_JSON = OUTPUT_DIR / "tick_classification_scores.json"
OUT_WARPED = OUTPUT_DIR / "perspective_corrected_worksheet.jpg"
OUT_SUBTRACTED = OUTPUT_DIR / "template_subtraction.png"
OUT_TEMPLATE_WARP = OUTPUT_DIR / "aligned_blank_template.png"


def load_image(path: Path, description: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Cannot load {description}: {path}\n"
            "Check the filename and ensure the image is inside samples/."
        )
    return image


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Normalize slow background-lighting changes while retaining thin strokes."""
    k = max(31, int(min(gray.shape) * ILLUMINATION_BLUR_FRACTION))
    if k % 2 == 0:
        k += 1
    background = cv2.GaussianBlur(gray, (k, k), 0)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def find_l_bracket_anchor(gray: np.ndarray, corner: str) -> tuple[float, float]:
    """Find an L-shaped dark marker in a corner ROI and estimate its page-corner point.

    This expects high-contrast L brackets (two perpendicular arms), not filled squares.
    The estimate uses the outside corner of the detected component's bounding box.
    """
    h, w = gray.shape
    rw = max(1, int(w * ANCHOR_CORNER_FRACTION))
    rh = max(1, int(h * ANCHOR_CORNER_FRACTION))
    if corner == "tl":
        x0, y0, x1, y1 = 0, 0, rw, rh
    elif corner == "tr":
        x0, y0, x1, y1 = w - rw, 0, w, rh
    elif corner == "bl":
        x0, y0, x1, y1 = 0, h - rh, rw, h
    else:
        x0, y0, x1, y1 = w - rw, h - rh, w, h

    roi = gray[y0:y1, x0:x1]
    # Dark-marker threshold is applied locally, tolerating tinted paper/background.
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    max_area = int(gray.size * ANCHOR_MAX_COMPONENT_AREA_FRACTION)
    candidates = []
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if area < ANCHOR_MIN_COMPONENT_AREA or area > max_area:
            continue
        if bw < 5 or bh < 5:
            continue
        ratio = bw / float(bh)
        if not 0.25 <= ratio <= 4.0:
            continue
        # Prefer the largest plausible component near the corresponding image corner.
        cx, cy = centroids[i]
        corner_distance = {
            "tl": np.hypot(cx, cy),
            "tr": np.hypot((roi.shape[1] - 1) - cx, cy),
            "bl": np.hypot(cx, (roi.shape[0] - 1) - cy),
            "br": np.hypot((roi.shape[1] - 1) - cx, (roi.shape[0] - 1) - cy),
        }[corner]
        candidates.append((area / (1.0 + corner_distance), bx, by, bw, bh))

    if not candidates:
        raise RuntimeError(
            f"Could not find the {corner.upper()} L-bracket in the image corner. "
            "Make sure all four markers are dark, visible, and inside the image."
        )

    _, bx, by, bw, bh = max(candidates, key=lambda v: v[0])
    # Map the outer end of the L marker to the corresponding canonical corner.
    if corner == "tl":
        point = (x0 + bx, y0 + by)
    elif corner == "tr":
        point = (x0 + bx + bw - 1, y0 + by)
    elif corner == "bl":
        point = (x0 + bx, y0 + by + bh - 1)
    else:
        point = (x0 + bx + bw - 1, y0 + by + bh - 1)
    return float(point[0]), float(point[1])


def perspective_correct(solved: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Warp the photographed sheet using four detected corner L-brackets."""
    gray = cv2.cvtColor(solved, cv2.COLOR_BGR2GRAY)
    names = ("tl", "tr", "br", "bl")
    source = np.array([find_l_bracket_anchor(gray, n) for n in names], dtype=np.float32)
    target_w, target_h = target_size
    # Small inset keeps the marker strokes just inside the output canvas.
    inset = 2.0
    target = np.array([
        [inset, inset],
        [target_w - 1 - inset, inset],
        [target_w - 1 - inset, target_h - 1 - inset],
        [inset, target_h - 1 - inset],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        solved, matrix, (target_w, target_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def infer_rows_from_template(template_gray: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Infer question-row centers dynamically from the repeated shaded row bands."""
    h, w = template_gray.shape
    normalized = normalize_illumination(template_gray)
    _, binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    xa = int(w * ANSWER_AREA_X_RANGE[0])
    xb = int(w * ANSWER_AREA_X_RANGE[1])
    answer_area = binary[:, xa:xb]
    kernel_width = max(15, int((xb - xa) * 0.20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(answer_area, cv2.MORPH_OPEN, kernel)

    # Use the mean grayscale signal in the mostly-unprinted pink strip just to
    # the right of the serial-number band. It has one repeated valley per row.
    band_x1, band_x2 = int(w * 0.08), int(w * 0.16)
    signal = np.mean(template_gray[:, band_x1:band_x2], axis=1).astype(np.float32)
    smooth_size = max(5, int(h * 0.006))
    if smooth_size % 2 == 0:
        smooth_size += 1
    signal = cv2.GaussianBlur(signal.reshape(-1, 1), (1, smooth_size), 0).ravel()
    minima = [
        y for y in range(2, h - 2)
        if signal[y] <= signal[y - 1] and signal[y] <= signal[y + 1]
    ]
    min_distance = max(8, int(h * 0.018))
    selected: list[int] = []
    for y in sorted(minima, key=lambda yy: signal[yy]):
        if all(abs(y - old) >= min_distance for old in selected):
            selected.append(y)
    selected = sorted(y for y in selected if int(h * 0.07) < y < int(h * 0.955))
    if len(selected) < 2:
        raise RuntimeError("Could not infer question rows from the blank template.")

    # The first and last band minima can sit near the outside edge of their row.
    # Nudge only those edge estimates inward by a fraction of the observed pitch.
    pitch = float(np.median(np.diff(selected)))
    selected[0] = int(selected[0] + pitch * 0.40)
    selected[-1] = int(selected[-1] - pitch * 0.50)
    return selected, horizontal


def remove_template_lines(solved_gray: np.ndarray, template_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract aligned print/template pixels, retaining likely student strokes."""
    solved_norm = normalize_illumination(solved_gray)
    template_norm = normalize_illumination(template_gray)
    _, solved_bin = cv2.threshold(
        solved_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, template_bin = cv2.threshold(
        template_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Slight dilation makes template subtraction tolerant of one-pixel alignment noise.
    template_dilated = cv2.dilate(
        template_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    residual = cv2.bitwise_and(solved_bin, cv2.bitwise_not(template_dilated))
    return residual, solved_bin


def score_tick(residual: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    x, y, w, h = box
    # Exclude box borders and inspect only the right-hand checkbox compartment.
    rx1 = x + int(w * CHECKBOX_LEFT_FRACTION)
    rx2 = x + w - max(2, int(w * CHECKBOX_RIGHT_MARGIN_FRACTION))
    ry1 = y + max(2, int(h * 0.16))
    ry2 = y + h - max(2, int(h * 0.16))
    if rx2 <= rx1 or ry2 <= ry1:
        return {"nonzero_pixels": 0, "density": 0.0, "marked": False, "error": "Invalid ROI"}
    roi = residual[ry1:ry2, rx1:rx2]
    # Residual is white foreground on black background.
    count = int(cv2.countNonZero(roi))
    density = count / float(max(1, roi.size))
    return {
        "nonzero_pixels": count,
        "density": round(density, 4),
        "marked": bool(count >= TICK_MIN_PIXELS and density >= TICK_DENSITY_MIN),
        "roi": [int(rx1), int(ry1), int(rx2), int(ry2)],
    }


def main() -> None:
    solved = load_image(SOLVED_IMAGE_PATH, "solved worksheet photo")
    template = load_image(BLANK_TEMPLATE_PATH, "blank worksheet template")
    template_h, template_w = template.shape[:2]
    print(f"Solved image size: {solved.shape[1]} x {solved.shape[0]}")
    print(f"Blank template size: {template_w} x {template_h}")

    corrected = perspective_correct(solved, (template_w, template_h))
    cv2.imwrite(str(OUT_WARPED), corrected)

    # The blank template is the canonical layout and already has no student marks.
    # Resize is used only to ensure exact pixel dimensions for subtraction.
    template_canonical = cv2.resize(template, (template_w, template_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_TEMPLATE_WARP), template_canonical)

    solved_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_canonical, cv2.COLOR_BGR2GRAY)
    residual, solved_binary = remove_template_lines(solved_gray, template_gray)
    cv2.imwrite(str(OUT_THRESHOLD), solved_binary)
    cv2.imwrite(str(OUT_SUBTRACTED), residual)

    row_centers, horizontal_lines = infer_rows_from_template(template_gray)
    cv2.imwrite(str(OUT_LINE_MAP), horizontal_lines)
    overlay = corrected.copy()
    # Map template row centers onto the aligned photo and visualize row guides.
    for q_num, cy in enumerate(row_centers, start=1):
        cv2.line(overlay, (int(template_w * ANSWER_AREA_X_RANGE[0]), cy),
                 (int(template_w * ANSWER_AREA_X_RANGE[1]), cy), (0, 255, 0), 1)
        cv2.putText(overlay, str(q_num), (int(template_w * 0.035), cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_LINE_OVERLAY), overlay)

    if len(row_centers) > 1:
        pitch = float(np.median(np.diff(row_centers)))
    else:
        pitch = template_h * 0.025
    box_h = max(8, int(pitch * 0.62))

    preview = corrected.copy()
    tick_debug = corrected.copy()
    scores: dict[str, dict] = {}
    question_results: dict[str, Optional[str]] = {}
    warnings: list[str] = []

    for q_num, center_y in enumerate(row_centers, start=1):
        y1 = max(0, int(center_y - box_h / 2))
        y2 = min(template_h - 1, int(center_y + box_h / 2))
        option_marked = []
        for option, (left_frac, right_frac) in OPTION_X_RANGES.items():
            x1 = int(template_w * left_frac)
            x2 = int(template_w * right_frac)
            box = (x1, y1, x2 - x1, y2 - y1)
            label = f"{q_num}{option}"
            scores[label] = score_tick(residual, box)
            if scores[label]["marked"]:
                option_marked.append(option)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
            rx = scores[label].get("roi")
            if rx:
                color = (0, 0, 255) if scores[label]["marked"] else (255, 0, 0)
                cv2.rectangle(tick_debug, (rx[0], rx[1]), (rx[2], rx[3]), color, 1)
                cv2.putText(tick_debug, f"{option}:{scores[label]['density']:.2f}",
                            (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.28, color, 1, cv2.LINE_AA)
        question_results[str(q_num)] = option_marked[0] if len(option_marked) == 1 else None
        if len(option_marked) != 1:
            warnings.append(
                f"Question {q_num}: detected {len(option_marked)} marked options; expected 1."
            )

    cv2.imwrite(str(OUT_BOX_PREVIEW), preview)
    cv2.imwrite(str(OUT_TICK_DEBUG), tick_debug)
    OUT_ANSWERS_JSON.write_text(
        json.dumps({"student_tick_mark": question_results}, indent=2), encoding="utf-8"
    )
    OUT_SCORES_JSON.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print("\nRectangle detection summary")
    print("Candidate boxes:", len(row_centers) * len(OPTIONS))
    print("Detected rows:", len(row_centers))
    print("Labeled boxes:", len(row_centers) * len(OPTIONS))
    for warning in warnings:
        print("WARNING:", warning)

    one_selection = sum(value is not None for value in question_results.values())
    print("\nTick detection completed.")
    print("Student selections:", OUT_ANSWERS_JSON)
    print("Tick scores:", OUT_SCORES_JSON)
    print("Debug preview:", OUT_TICK_DEBUG)
    print("Questions inferred from template rows:", len(row_centers))
    print("Questions with one detected selection:", one_selection)
    print("Blank/multiple/undetected selections:", len(question_results) - one_selection)
    print("\nOther outputs:")
    for path in (OUT_THRESHOLD, OUT_LINE_MAP, OUT_LINE_OVERLAY, OUT_BOX_PREVIEW,
                 OUT_WARPED, OUT_SUBTRACTED, OUT_TEMPLATE_WARP):
        print("-", path)
    print("\nNote: This is a heuristic baseline. Inspect previews and scores before relying on results.")


if __name__ == "__main__":
    main()
