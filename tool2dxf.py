import cv2
import numpy as np
import sys
import os
import shutil
import ezdxf
from shapely.geometry import Polygon
import tkinter as tk
from tkinter import filedialog, messagebox

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
    os.makedirs(folder, exist_ok=True)
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
# GUI polyline refinement (OpenCV)
# ------------------------------

EDITOR_WINDOW = "Refine Tool Contour"

def _cnt_to_list(cnt):
    # cnt: (N,1,2) -> [(x,y), ...]
    return [tuple(map(int, p)) for p in cnt.reshape(-1, 2)]

def _list_to_cnt(pts):
    # [(x,y), ...] -> (N,1,2) int32
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

def _closest_vertex_idx(point, poly_pts):
    # poly_pts: list[(x,y)]
    px, py = point
    d2 = [(px - x) ** 2 + (py - y) ** 2 for (x, y) in poly_pts]
    return int(np.argmin(d2))

def _integrate_polyline(contour_cnt, polyline_pts):
    """
    Replace the shortest arc between the two closest vertices to the polyline's
    endpoints with the polyline.
    contour_cnt: (N,1,2)
    polyline_pts: list[(x,y)] length >= 2
    """
    if len(polyline_pts) < 2:
        return contour_cnt  # nothing to do

    poly = _cnt_to_list(contour_cnt)
    n = len(poly)
    start_idx = _closest_vertex_idx(polyline_pts[0], poly)
    end_idx   = _closest_vertex_idx(polyline_pts[-1], poly)

    if n < 3 or start_idx == end_idx:
        # Degenerate or nonsense; just insert after the start
        new_poly = poly[:start_idx+1] + polyline_pts + poly[start_idx+1:]
        return _list_to_cnt(new_poly)

    # Compute number of vertices removed for each direction
    if start_idx < end_idx:
        removed_forward = end_idx - start_idx - 1
    else:
        removed_forward = n - (start_idx - end_idx) - 1  # wrap removal

    # Forward path: keep [0..start_idx], add polyline, keep [end_idx..end]
    if start_idx <= end_idx:
        forward_poly = poly[:start_idx+1] + polyline_pts + poly[end_idx:]
    else:
        forward_poly = poly[:start_idx+1] + polyline_pts + poly[end_idx:]  # wrap case identical

    # Backward: swap start/end and reverse polyline
    start2, end2 = end_idx, start_idx
    polyline_rev = list(reversed(polyline_pts))
    if start2 <= end2:
        backward_poly = poly[:start2+1] + polyline_rev + poly[end2:]
        removed_backward = end2 - start2 - 1
    else:
        backward_poly = poly[:start2+1] + polyline_rev + poly[end2:]
        removed_backward = n - (start2 - end2) - 1

    # Choose the integration that removes fewer original vertices
    new_poly = forward_poly if removed_forward <= removed_backward else backward_poly

    # Optional tiny cleanup: merge very-close neighbors (avoid duplicate spikes)
    cleaned = [new_poly[0]]
    for p in new_poly[1:]:
        if (p[0] - cleaned[-1][0])**2 + (p[1] - cleaned[-1][1])**2 > 1:  # >1px apart
            cleaned.append(p)

    # Light simplify to keep contour reasonable
    cnt = _list_to_cnt(cleaned)
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)
    return cnt

def draw_instructions(canvas):
    """Overlay key instructions on the OpenCV canvas."""
    instructions = [
        "Controls:",
        "Left-click: add polyline point",
        "Right-click: insert polyline into contour",
        "u: undo last point",
        "c: clear current polyline",
        "Enter: finish editing"
    ]
    x, y0 = 10, 20
    for i, text in enumerate(instructions):
        y = y0 + i * 22
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)

def refine_contour_gui(img, contour_cnt):
    base = img.copy()
    polygon_cnt = contour_cnt.copy()
    current_polyline = []
    last_scale = None
    last_offset = (0, 0)

    def to_display(pt, scale, offset):
        return (int(pt[0] * scale[0] + offset[0]),
                int(pt[1] * scale[1] + offset[1]))

    def to_original(pt, scale, offset):
        return (int((pt[0] - offset[0]) / scale[0]),
                int((pt[1] - offset[1]) / scale[1]))

    def redraw():
        nonlocal last_scale, last_offset
        # Get current window size
        win_x, win_y, win_w, win_h = cv2.getWindowImageRect(EDITOR_WINDOW)
        h, w = base.shape[:2]

        # Compute scale while keeping aspect ratio
        scale = min(win_w / w, win_h / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        # Resize image
        resized = cv2.resize(base, new_size, interpolation=cv2.INTER_AREA)

        # White background for the unused window area
        canvas = np.ones((win_h, win_w, 3), dtype=np.uint8) * 255
        offset_x = (win_w - new_size[0]) // 2
        offset_y = (win_h - new_size[1]) // 2
        canvas[offset_y:offset_y+new_size[1], offset_x:offset_x+new_size[0]] = resized

        last_scale = (new_size[0] / w, new_size[1] / h)
        last_offset = (offset_x, offset_y)

        # Draw polygon (scaled + centered)
        scaled_cnt = np.array(
            [to_display(p[0], last_scale, last_offset) for p in polygon_cnt],
            dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.drawContours(canvas, [scaled_cnt], -1, (0, 255, 0), 2)

        # Draw current polyline
        if len(current_polyline) > 1:
            pts = [to_display(p, last_scale, last_offset) for p in current_polyline]
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 2)
        for p in current_polyline:
            cv2.circle(canvas, to_display(p, last_scale, last_offset), 3, (0, 0, 255), -1)

        draw_instructions(canvas)
        cv2.imshow(EDITOR_WINDOW, canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal polygon_cnt, current_polyline, last_scale, last_offset
        if last_scale is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polyline.append(to_original((x, y), last_scale, last_offset))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polyline) > 1:
                polygon_cnt = _integrate_polyline(polygon_cnt, current_polyline)
            current_polyline = []
            redraw()

    print("Editor: Left-click to add polyline points, Right-click to insert into contour.")
    print("        'u' undo point, 'c' clear polyline, Enter to finish.")

    cv2.namedWindow(EDITOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(EDITOR_WINDOW, on_mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        redraw()
        if key == 13:   # Enter
            break
        elif key == ord('u'):
            if current_polyline:
                current_polyline.pop()
        elif key == ord('c'):
            current_polyline = []

    cv2.destroyWindow(EDITOR_WINDOW)
    return polygon_cnt

# ------------------------------
# Output directory helpers
# ------------------------------

def make_output_dirs_for_image(image_path, output_name):
    """
    Returns (base_dir, debug_dir, dxf_path) based on the chosen output_name.
    Ensures a clean (overwritten) folder for each run.
    """
    # Use provided output_name (sanitized)
    safe_name = "".join(c for c in output_name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_name:
        safe_name = os.path.splitext(os.path.basename(image_path))[0]
    base_dir = os.path.join("output", safe_name)
    debug_dir = os.path.join(base_dir, "debug")
    # Overwrite behavior: remove existing folder entirely
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(debug_dir, exist_ok=True)
    dxf_path = os.path.join(base_dir, f"{safe_name}.dxf")
    return base_dir, debug_dir, dxf_path

# ------------------------------
# Main detection function
# ------------------------------

def detect_paper(image_path, paper_size='A4', offset_mm=1.0, output_name=None):
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(image_path))[0]
    base_dir, debug_dir, dxf_out_path = make_output_dirs_for_image(image_path, output_name)
    step = 1

    sizes = {
        'A0': (841, 1189),
        'A1': (594, 841),
        'A2': (420, 594),
        'A3': (297, 420),
        'A4': (210, 297),
        'A5': (148, 210),
        'A6': (105, 148),
        'A7': (74, 105),
        'A8': (52, 74),
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
    save_debug_image(gray, debug_dir, step, "gray")
    step += 1

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug_image(blurred, debug_dir, step, "blurred")
    step += 1

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug_image(thresh, debug_dir, step, "thresh")
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
        save_debug_image(contoured_img, debug_dir, step, "contoured")
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
        save_debug_image(warped, debug_dir, step, "warped")
        step += 1

        # Draw grid
        gridded = warped.copy()
        for x_mm in range(0, int(dst_width_mm) + 1, 10):
            x_pix = int(x_mm * ppm_width)
            cv2.line(gridded, (x_pix, 0), (x_pix, output_height - 1), (0, 0, 255), 2)
        for y_mm in range(0, int(dst_height_mm) + 1, 10):
            y_pix = int(y_mm * ppm_height)
            cv2.line(gridded, (0, y_pix), (output_width - 1, y_pix), (0, 0, 255), 2)
        save_debug_image(gridded, debug_dir, step, "gridded")
        step += 1

        # ------------------------------
        # Robust Tool detection on top of warped paper
        # ------------------------------
        gray_tool = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        save_debug_image(gray_tool, debug_dir, step, "tool_gray")
        step += 1

        _, tool_mask = cv2.threshold(gray_tool, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        save_debug_image(tool_mask, debug_dir, step, "tool_mask")
        step += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tool_mask = cv2.morphologyEx(tool_mask, cv2.MORPH_CLOSE, kernel)
        save_debug_image(tool_mask, debug_dir, step, "tool_closed")
        step += 1

        tool_contours, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if tool_contours:
            largest_tool = max(tool_contours, key=cv2.contourArea)

            # === OpenCV editor to refine contour by inserting polylines ===
            largest_tool = refine_contour_gui(warped, largest_tool)

            tool_outline = warped.copy()
            cv2.drawContours(tool_outline, [largest_tool], -1, (0, 255, 0), 3)
            save_debug_image(tool_outline, debug_dir, step, "tool_detected_smooth")
            step += 1

            contour_to_dxf(largest_tool, dxf_out_path, offset_mm, ppm_width, ppm_height)
        else:
            print("⚠️ No tool contours found.")
    else:
        print("⚠️ No rectangular contour found.")

# ------------------------------
# GUI for input instead of CLI
# ------------------------------
class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, default_name):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("300x220")
        self.result = None

        # Paper size dropdown
        tk.Label(self, text="Paper size:").pack(pady=(10,0))
        self.paper_var = tk.StringVar(value="A4")
        options = ["A0","A1","A2","A3","A4","A5","A6","A7","A8"]
        tk.OptionMenu(self, self.paper_var, *options).pack()

        # Offset text input
        tk.Label(self, text="Offset (mm):").pack(pady=(10,0))
        self.offset_entry = tk.Entry(self)
        self.offset_entry.insert(0, "1.0")
        self.offset_entry.pack()

        # Output name input
        tk.Label(self, text="Output name:").pack(pady=(10,0))
        self.name_entry = tk.Entry(self)
        self.name_entry.insert(0, default_name)
        self.name_entry.pack()

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", width=10, command=self.on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10, command=self.on_cancel).pack(side=tk.LEFT, padx=5)

        self.grab_set()
        self.wait_window()

    def on_ok(self):
        try:
            offset = float(self.offset_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Offset must be a number.")
            return
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Invalid input", "Output name cannot be empty.")
            return
        self.result = (self.paper_var.get(), offset, name)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide root window

    # Ask for image file
    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
    )
    if not image_path:
        messagebox.showerror("Error", "No file selected.")
        sys.exit(1)

    # Default output name = file stem
    default_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ask for paper size + offset + output name in dialog
    dialog = SettingsDialog(root, default_name)
    if dialog.result is None:
        sys.exit(1)

    paper_size, offset_mm, output_name = dialog.result
    return image_path, paper_size, offset_mm, output_name

if __name__ == "__main__":
    image_path, paper_size, offset_mm, output_name = get_user_input()
    detect_paper(image_path, paper_size, offset_mm, output_name)
