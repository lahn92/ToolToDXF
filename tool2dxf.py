import cv2
import numpy as np
import sys
import os
import ezdxf
from shapely.geometry import Polygon

# ------------------------------
# Helper functions
# ------------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def save_debug_image(img, folder, step, name):
    filename = os.path.join(folder, f"{step:02d}_{name}.jpg")
    cv2.imwrite(filename, img)
    print(f"Step {step}: saved {filename}")

def contour_to_dxf(contour, output_path="tool.dxf", offset_mm=1.0, ppm_width=1.0, ppm_height=1.0):
    points_mm = []
    for pt in contour.reshape(-1, 2):
        x_mm = pt[0] / ppm_width
        y_mm = pt[1] / ppm_height
        points_mm.append((x_mm, y_mm))
    poly = Polygon(points_mm)
    poly_offset = poly.buffer(offset_mm, join_style=2)
    exterior_coords = list(poly_offset.exterior.coords)
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    msp.add_lwpolyline(exterior_coords, close=True)
    doc.saveas(output_path)
    print(f"✅ DXF saved to {output_path}")

# ------------------------------
# Main detection function
# ------------------------------

def detect_paper(image_path, paper_size='A4', offset_mm=1.0):
    debug_folder = "debug"
    os.makedirs(debug_folder, exist_ok=True)
    step = 1

    sizes = {
        'A0': (841, 1189), 'A1': (594, 841), 'A2': (420, 594),
        'A3': (297, 420), 'A4': (210, 297), 'A5': (148, 210),
        'A6': (105, 148), 'A7': (74, 105), 'A8': (52, 74),
    }

    paper_size = paper_size.upper()
    if paper_size not in sizes:
        print(f"⚠️ Unknown paper size '{paper_size}', defaulting to A4.")
        paper_size = 'A4'

    portrait_width_mm, portrait_height_mm = sizes[paper_size]
    expected_aspect = portrait_height_mm / portrait_width_mm
    DPI = 300

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_debug_image(gray, debug_folder, step, "gray")
    step += 1

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug_image(blurred, debug_folder, step, "blurred")
    step += 1

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug_image(thresh, debug_folder, step, "thresh")
    step += 1

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_contour = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                paper_contour = approx
                max_area = area

    if paper_contour is not None:
        x, y, w, h = cv2.boundingRect(paper_contour)
        if w >= img.shape[1] * 0.95 and h >= img.shape[0] * 0.95:
            paper_contour = None
            print("⚠️ Detected contour is nearly the entire image; ignoring.")
        else:
            rect = cv2.minAreaRect(paper_contour)
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height)
            if not (expected_aspect - 0.1 < aspect_ratio < expected_aspect + 0.1):
                print("⚠️ Found rectangle, but aspect ratio doesn't match the specified paper size.")
                paper_contour = None

    if paper_contour is not None:
        contoured_img = img.copy()
        cv2.drawContours(contoured_img, [paper_contour], -1, (0, 255, 0), 8)
        save_debug_image(contoured_img, debug_folder, step, "contoured")
        step += 1

        src_pts = order_points(paper_contour.reshape(4, 2))
        tl, tr, br, bl = src_pts

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        avg_width_pix = (width_top + width_bottom) / 2
        avg_height_pix = (height_left + height_right) / 2

        if avg_width_pix > avg_height_pix:
            dst_width_mm = portrait_height_mm
            dst_height_mm = portrait_width_mm
        else:
            dst_width_mm = portrait_width_mm
            dst_height_mm = portrait_height_mm

        output_width = int(dst_width_mm / 25.4 * DPI)
        output_height = int(dst_height_mm / 25.4 * DPI)

        ppm_width = output_width / dst_width_mm
        ppm_height = output_height / dst_height_mm
        print(f"Calibrated: {ppm_width:.2f} px/mm width, {ppm_height:.2f} px/mm height")

        dst_pts = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype="float32")

        H, _ = cv2.findHomography(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, H, (output_width, output_height))
        save_debug_image(warped, debug_folder, step, "warped")
        step += 1

        # Draw grid
        gridded = warped.copy()
        for x_mm in range(0, int(dst_width_mm) + 1, 10):
            x_pix = int(x_mm * ppm_width)
            cv2.line(gridded, (x_pix, 0), (x_pix, output_height - 1), (0, 0, 255), 2)
        for y_mm in range(0, int(dst_height_mm) + 1, 10):
            y_pix = int(y_mm * ppm_height)
            cv2.line(gridded, (0, y_pix), (output_width - 1, y_pix), (0, 0, 255), 2)
        save_debug_image(gridded, debug_folder, step, "gridded")
        step += 1

        # ------------------------------
        # Robust Tool detection on top of warped paper (more detailed)
        # ------------------------------
        gray_tool = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        save_debug_image(gray_tool, debug_folder, step, "tool_gray")
        step += 1

        # Threshold to create mask
        _, tool_mask = cv2.threshold(gray_tool, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional: slight blur to smooth edges but keep detail
        tool_mask = cv2.GaussianBlur(tool_mask, (3, 3), 0)
        save_debug_image(tool_mask, debug_folder, step, "tool_mask_blurred")
        step += 1

        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tool_mask = cv2.morphologyEx(tool_mask, cv2.MORPH_CLOSE, kernel)
        save_debug_image(tool_mask, debug_folder, step, "tool_closed")
        step += 1

        # Find contours with all points preserved
        tool_contours, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if tool_contours:
            largest_tool = max(tool_contours, key=cv2.contourArea)

            # Small epsilon for minimal simplification
            epsilon = 0.0001 * cv2.arcLength(largest_tool, True)
            largest_tool = cv2.approxPolyDP(largest_tool, epsilon, True)

            # Draw final tool contour
            tool_outline = warped.copy()
            cv2.drawContours(tool_outline, [largest_tool], -1, (0, 255, 0), 2)
            save_debug_image(tool_outline, debug_folder, step, "tool_detected_detailed")
            step += 1

            # Export DXF
            os.makedirs(os.path.join(debug_folder, "dxf"), exist_ok=True)
            dxf_path = os.path.join(debug_folder, "dxf", "tool_contour.dxf")
            contour_to_dxf(largest_tool, dxf_path, offset_mm, ppm_width, ppm_height)
        else:
            print("⚠️ No tool contours found.")
    else:
        print("⚠️ No rectangular contour found.")

# ------------------------------
# Command-line interface
# ------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        paper_size = sys.argv[2] if len(sys.argv) > 2 else 'A4'
        offset_mm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        detect_paper(image_path, paper_size, offset_mm)
    else:
        print("Usage: python script.py <image_path> [paper_size] [offset_mm]")
