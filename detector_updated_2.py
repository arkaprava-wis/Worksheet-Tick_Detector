"""Dynamic worksheet tick detector.

Pipeline:
1. Illumination normalization
2. Detect four L-bracket anchor marks and perspective-warp the solved photo
3. Align the blank template to the canonical canvas
4. Inverted Otsu threshold + template subtraction
5. Infer question rows from the template's answer-box horizontal rules
6. Score A/B/C checkbox compartments and save JSON/debug images

The blank template is the canonical layout. The solved image is aligned to it
using ECC registration, so the blank template does not need anchor marks. This
remains a heuristic detector and must be validated on the supplied images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

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

# Registration and tick-detection settings.
ECC_ITERATIONS = 250
ECC_EPSILON = 1e-6
# Grayscale residual threshold for non-blue pencil/pen marks after alignment.
RESIDUAL_DIFFERENCE_THRESHOLD = 32
# Blue tick mask in OpenCV HSV hue units (0-179).
BLUE_HUE_MIN = 90
BLUE_HUE_MAX = 140
BLUE_SATURATION_MIN = 45
BLUE_VALUE_MAX = 245

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
TICK_DENSITY_MIN = 0.012
TICK_MIN_PIXELS = 5

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


def align_solved_to_template(solved: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Align the solved digital worksheet to the blank template without anchors.

    The blank template is the canonical coordinate system. ECC estimates a small
    affine correction from the printed worksheet content; corner anchor marks are
    not used for registration and therefore need not exist in the blank image.
    """
    target_h, target_w = template.shape[:2]
    if solved.shape[:2] != (target_h, target_w):
        solved = cv2.resize(solved, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    solved_gray = cv2.cvtColor(solved, cv2.COLOR_BGR2GRAY)
    template_gray = normalize_illumination(template_gray)
    solved_gray = normalize_illumination(solved_gray)

    # Exclude the outer margin, where added corner markers can bias registration.
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mx, my = int(target_w * 0.06), int(target_h * 0.035)
    mask[my:target_h-my, mx:target_w-mx] = 255

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_ITERATIONS, ECC_EPSILON)
    try:
        cv2.findTransformECC(template_gray, solved_gray, warp,
                             cv2.MOTION_AFFINE, criteria, inputMask=mask, gaussFiltSize=5)
        aligned = cv2.warpAffine(
            solved, warp, (target_w, target_h),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        print("Template registration: ECC affine alignment completed.")
        print("Registration matrix:", np.round(warp, 5).tolist())
        return aligned
    except cv2.error as exc:
        # Exact digital exports may already be aligned; avoid failing on ECC
        # when the correlation optimizer cannot improve an already aligned pair.
        print("WARNING: ECC registration did not converge; using resized solved image.")
        print("Registration detail:", str(exc).splitlines()[0])
        return solved


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


def remove_template_lines(solved_bgr: np.ndarray, template_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a mark mask using color evidence and aligned template differences.

    Blue ink is detected directly in HSV. A grayscale residual is also retained for
    darker pencil/pen marks. Printed template pixels are suppressed from that residual.
    """
    solved_gray = cv2.cvtColor(solved_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    solved_norm = normalize_illumination(solved_gray)
    template_norm = normalize_illumination(template_gray)

    _, solved_bin = cv2.threshold(
        solved_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, template_bin = cv2.threshold(
        template_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Only suppress a very small neighborhood of printed template ink. The previous
    # 3x3 dilation was too aggressive around ticks touching a box border.
    template_neighborhood = cv2.dilate(
        template_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )
    gray_difference = cv2.absdiff(solved_norm, template_norm)
    _, difference_mask = cv2.threshold(
        gray_difference, RESIDUAL_DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    dark_residual = cv2.bitwise_and(difference_mask, solved_bin)
    dark_residual = cv2.bitwise_and(dark_residual, cv2.bitwise_not(template_neighborhood))

    hsv = cv2.cvtColor(solved_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        np.array([BLUE_HUE_MIN, BLUE_SATURATION_MIN, 25], dtype=np.uint8),
        np.array([BLUE_HUE_MAX, 255, BLUE_VALUE_MAX], dtype=np.uint8),
    )
    # Combine color-sensitive detection with residual detection for pencil/black ink.
    marks = cv2.bitwise_or(blue_mask, dark_residual)
    return marks, solved_bin, blue_mask


def score_tick(mark_mask: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    x, y, w, h = box
    # Inspect the right-hand checkbox compartment, excluding printed borders.
    rx1 = x + int(w * CHECKBOX_LEFT_FRACTION)
    rx2 = x + w - max(2, int(w * CHECKBOX_RIGHT_MARGIN_FRACTION))
    ry1 = y + max(2, int(h * 0.16))
    ry2 = y + h - max(2, int(h * 0.16))
    if rx2 <= rx1 or ry2 <= ry1:
        return {"nonzero_pixels": 0, "density": 0.0, "marked": False,
                "error": "Invalid ROI"}
    roi = mark_mask[ry1:ry2, rx1:rx2]
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

    # The blank worksheet has no anchors; align using its printed content instead.
    template_canonical = template.copy()
    corrected = align_solved_to_template(solved, template_canonical)
    cv2.imwrite(str(OUT_WARPED), corrected)
    cv2.imwrite(str(OUT_TEMPLATE_WARP), template_canonical)

    template_gray = cv2.cvtColor(template_canonical, cv2.COLOR_BGR2GRAY)
    mark_mask, solved_binary, blue_mask = remove_template_lines(corrected, template_canonical)
    cv2.imwrite(str(OUT_THRESHOLD), solved_binary)
    cv2.imwrite(str(OUT_SUBTRACTED), mark_mask)
    cv2.imwrite(str(OUTPUT_DIR / "blue_ink_mask.png"), blue_mask)

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
    question_results: dict[str, Union[Optional[str], list[str]]] = {}
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
            scores[label] = score_tick(mark_mask, box)
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
        question_results[str(q_num)] = (option_marked[0] if len(option_marked) == 1
                                        else option_marked if len(option_marked) > 1 else None)
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

    one_selection = sum(isinstance(value, str) for value in question_results.values())
    multiple_selections = sum(isinstance(value, list) for value in question_results.values())
    print("\nTick detection completed.")
    print("Student selections:", OUT_ANSWERS_JSON)
    print("Tick scores:", OUT_SCORES_JSON)
    print("Debug preview:", OUT_TICK_DEBUG)
    print("Questions inferred from template rows:", len(row_centers))
    print("Questions with one detected selection:", one_selection)
    print("Questions with multiple detected selections:", multiple_selections)
    print("Blank/undetected selections:", sum(value is None for value in question_results.values()))
    print("\nOther outputs:")
    for path in (OUT_THRESHOLD, OUT_LINE_MAP, OUT_LINE_OVERLAY, OUT_BOX_PREVIEW,
                 OUT_WARPED, OUT_SUBTRACTED, OUT_TEMPLATE_WARP, OUTPUT_DIR / "blue_ink_mask.png"):
        print("-", path)
    print("\nNote: This is a heuristic baseline. Inspect previews and scores before relying on results.")


if __name__ == "__main__":
    main()
