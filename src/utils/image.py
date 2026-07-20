"""Low-level image helpers used across preprocessing stages."""
import cv2
import numpy as np
from pathlib import Path

def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR. Raises FileNotFoundError if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_max_dim(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize image so its longest side equals `max_dim`, preserving aspect ratio.

    Returns:
        (resized_image, scale) where scale = new_size / original_size.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if not already."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered as: top-left, top-right, bottom-right, bottom-left.

    Standard trick: sum(x+y) is smallest at TL and largest at BR;
                    diff(y-x) is smallest at TR and largest at BL.

    Args:
        pts: (4, 2) array of points.
    Returns:
        (4, 2) float32 array in TL, TR, BR, BL order.
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]     # top-right
    rect[3] = pts[np.argmax(d)]     # bottom-left
    return rect


def adaptive_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """Compute Canny lower/upper thresholds from the image's median intensity."""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper

def save_debug_image(image: np.ndarray, stage: str, output_dir: str | Path) -> Path:
    """Save an intermediate image to disk for debugging.

    Args:
        image: Image (BGR or grayscale) to save.
        stage: Short label for this stage, e.g. "01_edges".
        output_dir: Directory to save to (created if missing).

    Returns:
        Path where the image was written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{stage}.png"
    cv2.imwrite(str(out), image)
    return out


def draw_contour(image: np.ndarray, contour: np.ndarray, color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    """Return a copy of `image` with the contour drawn on it."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, color, thickness)
    return vis


def draw_grid(image: np.ndarray, rows: int = 3, cols: int = 3, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Overlay an N×M grid on the image and label each cell (1-indexed)."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    for r in range(1, rows):
        cv2.line(vis, (0, r * cell_h), (w, r * cell_h), color, thickness)
    for c in range(1, cols):
        cv2.line(vis, (c * cell_w, 0), (c * cell_w, h), color, thickness)

    n = 1
    for r in range(rows):
        for c in range(cols):
            cx, cy = c * cell_w + 20, r * cell_h + 40
            cv2.putText(vis, str(n), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
            n += 1
    return vis