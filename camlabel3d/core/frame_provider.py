"""Frame source abstractions for videos and image folders."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from .models import natural_sort_key

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FrameProvider(ABC):
    """Abstract frame provider shared by video and folder backends."""

    def __init__(self, source_path: Path, source_type: str) -> None:
        self.path = Path(source_path).resolve()
        self.source_type = source_type
        digest = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()
        self.source_id = digest[:16]

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total number of readable frames."""

    @property
    def fps(self) -> float | None:
        """Frames per second when available."""
        return None

    @abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """Return RGB uint8 frame data."""

    def get_timestamp_ms(self, index: int) -> float | None:
        if self.fps and self.fps > 0:
            return (1000.0 * float(index)) / float(self.fps)
        return None

    def get_image_path(self, index: int) -> str:
        return ""

    def default_output_csv_path(self) -> Path:
        if self.path.is_dir():
            return self.path.parent / f"{self.path.name}.camlabel3d.csv"
        return self.path.with_suffix(".camlabel3d.csv")

    def frame_shape(self, index: int = 0) -> tuple[int, int]:
        frame = self.get_frame(index)
        return int(frame.shape[0]), int(frame.shape[1])

    def close(self) -> None:
        """Release any external resources."""


class ImageFolderFrameProvider(FrameProvider):
    """Frame provider for image folders."""

    def __init__(self, folder_path: Path) -> None:
        super().__init__(folder_path, "image_folder")
        if not self.path.is_dir():
            raise ValueError(f"Image folder does not exist: {self.path}")
        self._images = sorted(
            [
                path
                for path in self.path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda item: natural_sort_key(item.name),
        )
        if not self._images:
            raise ValueError(f"No supported images found under {self.path}")

    @property
    def frame_count(self) -> int:
        return len(self._images)

    def get_frame(self, index: int) -> np.ndarray:
        path = self._images[self._clamp_index(index)]
        return np.array(Image.open(path).convert("RGB"))

    def get_image_path(self, index: int) -> str:
        return str(self._images[self._clamp_index(index)])

    def _clamp_index(self, index: int) -> int:
        if not 0 <= int(index) < len(self._images):
            raise IndexError(f"Frame index out of range: {index}")
        return int(index)


class VideoFrameProvider(FrameProvider):
    """Frame provider for video files backed by OpenCV."""

    def __init__(self, video_path: Path) -> None:
        super().__init__(video_path, "video")
        if not self.path.is_file():
            raise ValueError(f"Video file does not exist: {self.path}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for video input. Please install opencv-python."
            ) from exc
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        if self._frame_count <= 0:
            probe_count = 0
            while True:
                ok, _ = self._capture.read()
                if not ok:
                    break
                probe_count += 1
            self._frame_count = probe_count
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self._frame_count <= 0:
            raise RuntimeError(f"No readable frames found in video: {self.path}")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float | None:
        return self._fps

    def get_frame(self, index: int) -> np.ndarray:
        index = self._clamp_index(index)
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {index} from {self.path}")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()

    def _clamp_index(self, index: int) -> int:
        if not 0 <= int(index) < self.frame_count:
            raise IndexError(f"Frame index out of range: {index}")
        return int(index)


def open_media_source(path: str | Path) -> FrameProvider:
    """Open a supported media source from a folder or video file."""
    source = Path(path).resolve()
    if source.is_dir():
        return ImageFolderFrameProvider(source)
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        return VideoFrameProvider(source)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageFolderFrameProvider(source.parent)
    raise ValueError(f"Unsupported media source: {source}")
