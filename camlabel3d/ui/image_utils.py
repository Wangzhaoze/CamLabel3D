"""Qt image conversion helpers safe to run outside the GUI thread."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage


def pil_to_qimage(image: Image.Image) -> QImage:
    """Return an owning QImage; the temporary PIL byte buffer may be released."""

    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()


def rgb_array_to_qimage(frame_rgb: np.ndarray) -> QImage:
    """Create an owning QImage directly from an RGB uint8 array."""

    frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB frame, got shape {frame.shape!r}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    qimage = QImage(
        frame.data,
        width,
        height,
        int(frame.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()
