# Tool2DXF

This Python project automatically detects a tool on a photographed paper sheet, allows interactive contour refinement, and exports the contour as a DXF file. Ideal for CNC machining, laser cutting, or CAD workflows.

---

## Features

- Automatic detection of rectangular paper sheets in images.
- Perspective correction (warping) and size calibration based on selected paper size.
- Tool contour detection on top of the warped sheet.
- Interactive OpenCV GUI to refine contours:
  - Add polyline points
  - Insert polylines into the contour
  - Undo or clear edits
- Export contours to DXF format with configurable offset.
- Step-by-step debug images saved for verification.

---

![Demo](pictures/animation.webp)