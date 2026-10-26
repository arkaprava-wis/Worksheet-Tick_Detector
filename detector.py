import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "samples"
    / "solved_clicked_worksheet.jpeg"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_QUESTIONS = 34
OPTIONS = ["A", "B", "C"]
EXPECTED_BOXES = EXPECTED_QUESTIONS * len(OPTIONS)


# --------------------------------------------------
# 2. Load the original worksheet
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Cannot load worksheet image: {IMAGE_PATH}"
    )

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

print(f"Image size: {width} x {height}")


# --------------------------------------------------
# 3. Apply Otsu's thresholding
# --------------------------------------------------

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

threshold_path = OUTPUT_DIR / "threshold.png"
cv2.imwrite(str(threshold_path), binary)

print("Threshold image saved:", threshold_path)


# --------------------------------------------------
# 4. Extract horizontal and vertical lines
# --------------------------------------------------

horizontal_length = max(15, width // 35)
vertical_length = max(10, height // 100)

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (horizontal_length, 1),
)

vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, vertical_length),
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel,
)

vertical_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel,
)


# --------------------------------------------------
# 5. Combine and close small gaps
# --------------------------------------------------

line_map = cv2.bitwise_or(
    horizontal_lines,
    vertical_lines,
)

closing_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 3),
)

line_map = cv2.morphologyEx(
    line_map,
    cv2.MORPH_CLOSE,
    closing_kernel,
)

line_map_path = OUTPUT_DIR / "line_map.png"
cv2.imwrite(str(line_map_path), line_map)

print("Line map saved:", line_map_path)


# --------------------------------------------------
# 6. Save the line-detection overlay
# --------------------------------------------------

line_overlay = image.copy()
line_overlay[line_map > 0] = (0, 255, 0)

overlay_path = OUTPUT_DIR / "line_detection_preview.jpg"

cv2.imwrite(str(overlay_path), line_overlay)

print("Line overlay saved:", overlay_path)


# --------------------------------------------------
# 7. Find rectangular candidates
# --------------------------------------------------

contours, _ = cv2.findContours(
    line_map,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE,
)

candidates = []

min_box_width = int(width * 0.065)
max_box_width = int(width * 0.18)

min_box_height = max(12, int(height * 0.008))
max_box_height = int(height * 0.04)

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    # Answer options are on the right-hand side.
    if x < width * 0.55:
        continue

    # Ignore the heading area.
    if y < height * 0.065:
        continue

    if not (
        min_box_width <= w <= max_box_width
        and min_box_height <= h <= max_box_height
    ):
        continue

    aspect_ratio = w / float(h)

    if not 2.0 <= aspect_ratio <= 7.0:
        continue

    candidates.append((x, y, w, h))


# --------------------------------------------------
# 8. Remove duplicate detections
# --------------------------------------------------

def intersection_over_union(box_a, box_b):
    """Calculate overlap between two rectangles."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection = (
        max(0, right - left)
        * max(0, bottom - top)
    )

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


# Keep larger rectangles first.
candidates.sort(
    key=lambda box: box[2] * box[3],
    reverse=True,
)

boxes = []

for candidate in candidates:

    duplicate = False

    for existing in boxes:

        if (
            intersection_over_union(
                candidate,
                existing,
            ) > 0.65
        ):
            duplicate = True
            break

    if not duplicate:
        boxes.append(candidate)


# --------------------------------------------------
# 9. Group rectangles into question rows
# --------------------------------------------------

boxes.sort(key=lambda box: box[1])

rows = []

for box in boxes:

    x, y, w, h = box
    center_y = y + h / 2

    matching_row = None

    for row in rows:

        row_center = np.mean([
            item[1] + item[3] / 2
            for item in row
        ])

        tolerance = max(8, int(h * 0.45))

        if abs(center_y - row_center) <= tolerance:
            matching_row = row
            break

    if matching_row is None:
        rows.append([box])
    else:
        matching_row.append(box)


# Sort rows from top to bottom.
rows.sort(
    key=lambda row: np.mean([
        item[1] + item[3] / 2
        for item in row
    ])
)

# Sort boxes in each row from left to right.
for row in rows:
    row.sort(key=lambda box: box[0])


# --------------------------------------------------
# 10. Assign question-option labels
# --------------------------------------------------

labeled_boxes = []
validation_errors = []

for row_index, row in enumerate(rows, start=1):

    if len(row) != 3:
        validation_errors.append(
            f"Row {row_index}: found {len(row)} boxes; "
            "expected 3."
        )

    for option_index, box in enumerate(row):

        # Do not assign A/B/C beyond the third box.
        if option_index >= len(OPTIONS):
            continue

        question_number = row_index
        option = OPTIONS[option_index]
        label = f"{question_number}{option}"

        labeled_boxes.append({
            "label": label,
            "question": question_number,
            "option": option,
            "box": box,
        })


# --------------------------------------------------
# 11. Draw labeled detection boxes
# --------------------------------------------------

preview = image.copy()

for item in labeled_boxes:

    x, y, w, h = item["box"]
    label = item["label"]

    # Draw the detected rectangle.
    cv2.rectangle(
        preview,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    # Put labels above the rectangles.
    cv2.putText(
        preview,
        label,
        (x, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


preview_path = OUTPUT_DIR / "detected_boxes_preview.jpg"

cv2.imwrite(str(preview_path), preview)


# --------------------------------------------------
# 12. Validate detection results
# --------------------------------------------------

print("\nDetection completed.")
print("Total candidate boxes:", len(boxes))
print("Detected question rows:", len(rows))
print("Labeled boxes:", len(labeled_boxes))

print("\nOutput files:")
print("1.", threshold_path)
print("2.", line_map_path)
print("3.", overlay_path)
print("4.", preview_path)

if validation_errors:
    print("\nVALIDATION WARNINGS:")

    for error in validation_errors:
        print("-", error)

else:
    print(
        "\nEvery detected row contains exactly "
        "three boxes."
    )

if len(boxes) == EXPECTED_BOXES and not validation_errors:
    print(
        "PASS: 102 boxes detected across 34 rows."
    )
else:
    print(
        f"CHECK REQUIRED: expected {EXPECTED_BOXES} "
        "boxes arranged in 34 rows of 3."
    )

print(
    "\nImportant: Labels depend on the detected row "
    "grouping being correct. Inspect the preview "
    "before using these coordinates for tick detection."
)


# --------------------------------------------------
# 13. Detect ticks inside option checkbox compartments
# --------------------------------------------------

import json

# Tune these thresholds after inspecting the debug image.
DARK_PIXEL_THRESHOLD = 100
MIN_TICK_PIXELS = 15

# Keep diagnostics for every detected option.
tick_scores = {}

debug_preview = image.copy()

for item in labeled_boxes:

    label = item["label"]
    x, y, w, h = item["box"]

    # The number occupies the left part of each option.
    # Inspect only the right-hand checkbox compartment.
    checkbox_x1 = x + int(w * 0.65)
    checkbox_x2 = x + w - 4

    checkbox_y1 = y + 3
    checkbox_y2 = y + h - 3

    if (
        checkbox_x2 <= checkbox_x1
        or checkbox_y2 <= checkbox_y1
    ):
        tick_scores[label] = {
            "dark_pixels": 0,
            "marked": False,
            "error": "Invalid checkbox crop",
        }
        continue

    checkbox_roi = gray[
        checkbox_y1:checkbox_y2,
        checkbox_x1:checkbox_x2,
    ]

    # Dark pencil/pen strokes become white.
    _, checkbox_binary = cv2.threshold(
        checkbox_roi,
        DARK_PIXEL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )

    dark_pixels = int(
        cv2.countNonZero(checkbox_binary)
    )

    # Initial heuristic; validate against the debug image.
    marked = dark_pixels >= MIN_TICK_PIXELS

    tick_scores[label] = {
        "dark_pixels": dark_pixels,
        "marked": marked,
    }

    # Draw the inspected checkbox compartment.
    color = (0, 0, 255) if marked else (255, 0, 0)

    cv2.rectangle(
        debug_preview,
        (checkbox_x1, checkbox_y1),
        (checkbox_x2, checkbox_y2),
        color,
        1,
    )

    cv2.putText(
        debug_preview,
        str(dark_pixels),
        (x, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        color,
        1,
        cv2.LINE_AA,
    )


# --------------------------------------------------
# 14. Classify the selected option per question
# --------------------------------------------------

student_tick_mark = {}

for question_number in range(1, EXPECTED_QUESTIONS + 1):

    marked_options = []

    for option in OPTIONS:

        label = f"{question_number}{option}"
        score = tick_scores.get(label, {})

        if score.get("marked", False):
            marked_options.append(option)

    # Exactly one marked option is required.
    # Blank or multiple-mark questions become null.
    if len(marked_options) == 1:
        student_tick_mark[str(question_number)] = (
            marked_options[0]
        )
    else:
        student_tick_mark[str(question_number)] = None


# --------------------------------------------------
# 15. Save JSON output
# --------------------------------------------------

json_path = OUTPUT_DIR / "student_tick_marks.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        {"student_tick_mark": student_tick_mark},
        file,
        indent=2,
    )


# --------------------------------------------------
# 16. Save tick-detection diagnostics
# --------------------------------------------------

scores_path = OUTPUT_DIR / "tick_detection_scores.json"

with open(scores_path, "w", encoding="utf-8") as file:
    json.dump(tick_scores, file, indent=2)

debug_path = OUTPUT_DIR / "tick_detection_debug.jpg"

cv2.imwrite(str(debug_path), debug_preview)


# --------------------------------------------------
# 17. Print results
# --------------------------------------------------

print("\nTick detection completed.")
print("Student selections:", json_path)
print("Tick scores:", scores_path)
print("Debug preview:", debug_path)

selected_count = sum(
    value is not None
    for value in student_tick_mark.values()
)

ambiguous_count = EXPECTED_QUESTIONS - selected_count

print("Questions with one detected selection:", selected_count)
print("Blank or multiple-mark questions:", ambiguous_count)

print(
    "\nIMPORTANT: The pixel threshold is an initial "
    "heuristic, not a validated tick classifier."
)