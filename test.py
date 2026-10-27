"""
Robust Corner Anchor Worksheet Tick Detector
Designed for high-accuracy processing of skewed smartphone captures.
"""

import json
from pathlib import Path
import cv2
import numpy as np

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File Targets
SOLVED_IMAGE_PATH = BASE_DIR / "samples" / "solved_clicked_worksheet.jpeg"
OUT_WARPED = OUTPUT_DIR / "perfect_aligned_worksheet.jpg"
OUT_DEBUG = OUTPUT_DIR / "tick_detection_debug.jpg"
OUT_RESULTS = OUTPUT_DIR / "detected_student_answers.json"

# --- STEP 5: Target Canvas Coordinates Configuration ---
# Once transformed, the image will be exactly 1000x1400 pixels.
CANONICAL_W = 1000
CANONICAL_H = 1400

# Strict Row Pitch coordinates for the 34-item sheet layout
ROW_START_Y = 180      # Exact Y-coordinate where row 1 starts
ROW_HEIGHT = 34        # Pixel height of each row compartment

# Step 5 Column coordinate zones for Option Bounding Boxes
COLUMN_X_MAPPING = {
    "A": (590, 680),   # (xmin, xmax)
    "B": (715, 805),
    "C": (840, 930)
}

# Sub-slice inside the bounding box to isolate the tick and ignore borders
TICK_PADDING_X = 12
TICK_PADDING_Y = 6
MIN_TICK_PIXEL_THRESHOLD = 95  # Minimum active white pixels to count as a checkmark


def get_anchor_corners(image: np.ndarray) -> np.ndarray:
    """
    Locates the 4 corner L-bracket anchor marks using structural contour filtering.
    Orders them strictly: [Top-Left, Top-Right, Bottom-Left, Bottom-Right].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing to pull out high-contrast black corner markers
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_points = []
    img_h, img_w = gray.shape
    
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
        area = cv2.contourArea(c)
        
        # Filter out tiny noise and gigantic elements like sheet borders
        if 150 < area < (img_w * img_h * 0.05):
            # Calculate center point mass of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_points.append([cx, cy])
                
    if len(detected_points) < 4:
        print(f"Warning: Only found {len(detected_points)} anchor marks. Falling back to default corner edges.")
        return np.array([[0, 0], [img_w - 1, 0], [0, img_h - 1], [img_w - 1, img_h - 1]], dtype="float32")
        
    # Sort points based on their sum and difference vectors to map corners accurately
    pts = np.array(detected_points)
    sorted_pts = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    sorted_pts[0] = pts[np.argmin(s)]  # Top-Left
    sorted_pts[3] = pts[np.argmax(s)]  # Bottom-Right
    
    diff = np.diff(pts, axis=1)
    sorted_pts[1] = pts[np.argmin(diff)] # Top-Right
    sorted_pts[2] = pts[np.argmax(diff)] # Bottom-Left
    
    return sorted_pts


def main():
    # 1. Read input image
    img = cv2.imread(str(SOLVED_IMAGE_PATH))
    if img is None:
        print(f"Error: Could not load file from {SOLVED_IMAGE_PATH}")
        return

    # 2. STEP 2: Find anchor marks and remove smartphone skew/tilt distortions
    src_corners = get_anchor_corners(img)
    dst_corners = np.array([
        [0, 0],
        [CANONICAL_W - 1, 0],
        [0, CANONICAL_H - 1],
        [CANONICAL_W - 1, CANONICAL_H - 1]
    ], dtype="float32")
    
    transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(img, transform_matrix, (CANONICAL_W, CANONICAL_H))
    cv2.imwrite(str(OUT_WARPED), warped)
    print(f"Transformation Complete: Rectified worksheet saved to {OUT_WARPED}")

    # 3. STEP 3 & 4: Equalization & Clean Inverted Binarization
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Background light flattening filter
    bg_smooth = cv2.GaussianBlur(warped_gray, (51, 51), 0)
    normalized = cv2.divide(warped_gray, np.maximum(bg_smooth, 1), scale=255)
    
    # Otsu thresholding creates white strokes (255) on black spaces (0)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. STEP 5 & 6: Grid Segmentation Loop & Ink Evaluation
    student_selections = {}
    debug_preview = warped.copy()
    
    print("\n--- PROCESSING SELECTIONS (STEP 5 ROW ANALYSIS) ---")
    for row_idx in range(34):
        q_num = row_idx + 1
        ymin = ROW_START_Y + (row_idx * ROW_HEIGHT)
        ymax = ymin + ROW_HEIGHT
        
        row_scores = {}
        
        # Check Options A, B, and C systematically inside the target row array slice
        for option, (xmin, xmax) in COLUMN_X_MAPPING.items():
            # Slice down to the specific letter compartment bounding box array
            # Apply padding margins to throw away outer printed bounding lines completely
            roi_x1 = xmin + TICK_PADDING_X
            roi_x2 = xmax - TICK_PADDING_X
            roi_y1 = ymin + TICK_PADDING_Y
            roi_y2 = ymax - TICK_PADDING_Y
            
            # Slice matrix
            cell_roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Count the active white pen pixels in the targeted slot space
            ink_pixel_count = int(cv2.countNonZero(cell_roi))
            row_scores[option] = ink_pixel_count
            
            # Render visual box guidelines onto debug output image
            is_marked = ink_pixel_count >= MIN_TICK_PIXEL_THRESHOLD
            box_color = (0, 0, 255) if is_marked else (255, 0, 0) # Red if ticked, Blue if empty
            cv2.rectangle(debug_preview, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, 1)
            
        # Determine the winner based on highest stroke density count
        best_option = max(row_scores, key=row_scores.get)
        highest_score = row_scores[best_option]
        
        if highest_score >= MIN_TICK_PIXEL_THRESHOLD:
            student_selections[str(q_num)] = best_option
            print(f"Question {q_num:02d}: Ticked -> {best_option} (Pixel Count: {highest_score})")
            
            # Draw marked confirmation marker text on the visual dashboard export
            win_xmin, _ = COLUMN_X_MAPPING[best_option]
            cv2.putText(debug_preview, "CHECK", (win_xmin, ymax - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            student_selections[str(q_num)] = None
            print(f"Question {q_num:02d}: Blank / Unanswered Row")

    # 5. Export results to local output directory files
    cv2.imwrite(str(OUT_DEBUG), debug_preview)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump({"detected_student_answers": student_selections}, f, indent=2)
        
    print(f"\nProcessing Complete! Check output directory files:")
    print(f"- Debug Marks Visualization: {OUT_DEBUG}")
    print(f"- Extracted Data (JSON): {OUT_RESULTS}")


if __name__ == "__main__":
    main()
