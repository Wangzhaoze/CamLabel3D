"""Frame source abstractions for videos and image folders."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .frame_cache import FrameCache
from .models import natural_sort_key

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FrameProvider(ABC):
    """Abstract frame provider shared by video and folder backends."""

    def __init__(self, source_path: Path, source_type: str, cache_bytes: int = 2048 * 1024 * 1024) -> None:
        self.path = Path(source_path).resolve()
        self.source_type = source_type
        digest = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()
        self.source_id = digest[:16]
        self._frame_cache = FrameCache(cache_bytes)
        self._frame_shapes: dict[int, tuple[int, int]] = {}
        self._default_frame_shape: tuple[int, int] | None = None
        self._lifecycle_lock = RLock()
        self._closed = False

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

    def prefetch(self, indices: Iterable[int]) -> None:
        """Schedule best-effort decoding; providers may intentionally no-op."""
        del indices

    def get_preview_frame(self, index: int) -> np.ndarray:
        """Return a frame optimized for interactive preview when available."""

        return self.get_frame(index)

    def default_output_csv_path(self) -> Path:
        if self.path.is_dir():
            return self.path.parent / f"{self.path.name}.camlabel3d.csv"
        return self.path.with_suffix(".camlabel3d.csv")

    def frame_shape(self, index: int = 0) -> tuple[int, int]:
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            cached = self._frame_shapes.get(int(index))
            if cached is not None:
                return cached
            if self._default_frame_shape is not None:
                return self._default_frame_shape
        frame = self.get_frame(index)
        shape = (int(frame.shape[0]), int(frame.shape[1]))
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            self._frame_shapes[int(index)] = shape
        return shape

    def close(self) -> None:
        """Release any external resources."""
        with self._lifecycle_lock:
            self._close_locked()

    def _cached_frame(self, index: int) -> np.ndarray | None:
        """Return a cached frame while rejecting reads after shutdown."""
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            return self._frame_cache.get(index)

    def _store_decoded_frame(self, index: int, frame: np.ndarray) -> np.ndarray:
        """Atomically publish decoded data only while this provider is open."""
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            self._frame_shapes[int(index)] = (int(frame.shape[0]), int(frame.shape[1]))
            return self._frame_cache.put(index, frame)

    def _close_locked(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._frame_cache.clear()
        return True

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError(f"Frame provider is closed: {self.path}")


class ImageFolderFrameProvider(FrameProvider):
    """Frame provider for image folders."""

    def __init__(
        self,
        folder_path: Path,
        *,
        cache_bytes: int = 2048 * 1024 * 1024,
        prefetch_workers: int = 0,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(folder_path, "image_folder", cache_bytes=cache_bytes)
        if not self.path.is_dir():
            raise ValueError(f"Image folder does not exist: {self.path}")
        images: list[Path] = []
        for offset, path in enumerate(self.path.iterdir()):
            if offset % 256 == 0 and should_cancel is not None and should_cancel():
                raise RuntimeError("Image folder discovery was canceled.")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Image folder discovery was canceled.")
        self._images = sorted(images, key=lambda item: natural_sort_key(item.name))
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Image folder discovery was canceled.")
        if not self._images:
            raise ValueError(f"No supported images found under {self.path}")
        worker_count = max(0, int(prefetch_workers))
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=min(4, worker_count), thread_name_prefix="frame-prefetch")
            if worker_count > 0
            else None
        )
        self._prefetch_futures: dict[int, Future[np.ndarray]] = {}

    @property
    def frame_count(self) -> int:
        return len(self._images)

    def get_frame(self, index: int) -> np.ndarray:
        index = self._clamp_index(index)
        cached = self._cached_frame(index)
        if cached is not None:
            return cached
        frame = self._decode_frame(index)
        return self._store_decoded_frame(index, frame)

    def get_image_path(self, index: int) -> str:
        return str(self._images[self._clamp_index(index)])

    def frame_shape(self, index: int = 0) -> tuple[int, int]:
        index = self._clamp_index(index)
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            cached = self._frame_shapes.get(index)
            if cached is not None:
                return cached
        with Image.open(self._images[index]) as image:
            width, height = image.size
        shape = (int(height), int(width))
        with self._lifecycle_lock:
            self._raise_if_closed_locked()
            self._frame_shapes[index] = shape
        return shape

    def prefetch(self, indices: Iterable[int]) -> None:
        for raw_index in indices:
            try:
                index = self._clamp_index(raw_index)
            except IndexError:
                continue
            with self._lifecycle_lock:
                executor = self._prefetch_executor
                if self._closed or executor is None:
                    return
                if self._frame_cache.get(index) is not None:
                    continue
                if index in self._prefetch_futures:
                    continue
                try:
                    future = executor.submit(self.get_frame, index)
                except RuntimeError:
                    # A broken or concurrently shutting-down executor only makes
                    # this best-effort optimization unavailable.
                    return
                self._prefetch_futures[index] = future
                future.add_done_callback(
                    lambda completed, key=index: self._complete_prefetch(key, completed)
                )

    def close(self) -> None:
        with self._lifecycle_lock:
            self._close_locked()
            executor = self._prefetch_executor
            self._prefetch_executor = None
            futures = tuple(self._prefetch_futures.values())
            self._prefetch_futures.clear()
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _complete_prefetch(self, index: int, future: Future[np.ndarray]) -> None:
        # Observe best-effort worker failures so executor exceptions never leak
        # into lifecycle callbacks or affect foreground frame reads.
        if not future.cancelled():
            try:
                future.exception()
            except CancelledError:
                pass
        with self._lifecycle_lock:
            if self._prefetch_futures.get(int(index)) is future:
                self._prefetch_futures.pop(int(index), None)

    def _decode_frame(self, index: int) -> np.ndarray:
        path = self._images[index]
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _clamp_index(self, index: int) -> int:
        if not 0 <= int(index) < len(self._images):
            raise IndexError(f"Frame index out of range: {index}")
        return int(index)


class VideoFrameProvider(FrameProvider):
    """Frame provider for video files backed by OpenCV."""

    def __init__(
        self,
        video_path: Path,
        *,
        cache_bytes: int = 2048 * 1024 * 1024,
        forward_scan_limit: int = 60,
        preload_all_frames: bool = True,
    ) -> None:
        super().__init__(video_path, "video", cache_bytes=cache_bytes)
        if not self.path.is_file():
            raise ValueError(f"Video file does not exist: {self.path}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for video input. Please install opencv-python."
            ) from exc
        self._cv2 = cv2
        self._capture_lock = RLock()
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_width > 0 and frame_height > 0:
            self._default_frame_shape = (frame_height, frame_width)
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
        self._next_read_index = 0
        self._forward_scan_limit = max(0, int(forward_scan_limit))
        self._last_preview_index: int | None = None
        self._preload_executor: ThreadPoolExecutor | None = None
        self._preload_future: Future[None] | None = None
        estimated_video_bytes = self._frame_count * frame_width * frame_height * 3
        if (
            preload_all_frames
            and estimated_video_bytes > 0
            and estimated_video_bytes <= self._frame_cache.stats().max_bytes
        ):
            self._preload_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="video-preload",
            )
            self._preload_future = self._preload_executor.submit(self._preload_all_frames)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float | None:
        return self._fps

    def get_frame(self, index: int) -> np.ndarray:
        index = self._clamp_index(index)
        cached = self._cached_frame(index)
        if cached is not None:
            return cached
        with self._capture_lock:
            cached = self._cached_frame(index)
            if cached is not None:
                return cached
            capture = self._capture
            if capture is None:
                raise RuntimeError(f"Video provider is closed: {self.path}")
            forward_gap = index - self._next_read_index
            if 0 < forward_gap <= self._forward_scan_limit:
                # H.264 frame seeks are commonly 40-100 ms even for a two-frame
                # jump, while advancing the existing decoder with grab() is
                # typically sub-millisecond per frame. Keep playback-like
                # forward scrubbing on the sequential decoder path.
                for _ in range(forward_gap):
                    if not capture.grab():
                        raise RuntimeError(f"Failed to advance video decoder toward frame {index}")
            elif self._next_read_index != index:
                capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            self._next_read_index = index + 1
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {index} from {self.path}")
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        return self._store_decoded_frame(index, frame_rgb)

    def get_preview_frame(self, index: int) -> np.ndarray:
        """Decode interactively, warming a bounded window for reverse scrubs."""

        index = self._clamp_index(index)
        cached = self._cached_frame(index)
        if cached is not None:
            self._last_preview_index = index
            return cached
        with self._capture_lock:
            cached = self._cached_frame(index)
            if cached is not None:
                self._last_preview_index = index
                return cached
            previous_index = self._last_preview_index
            self._last_preview_index = index
            if previous_index is not None and index < previous_index:
                return self._decode_reverse_window_locked(index)
            # RLock permits reuse of the normal adaptive forward path.
            return self.get_frame(index)

    def _decode_reverse_window_locked(self, target_index: int) -> np.ndarray:
        """Seek once, then fill frames immediately preceding a reverse target."""

        capture = self._capture
        if capture is None:
            raise RuntimeError(f"Video provider is closed: {self.path}")
        frame_shape = self._default_frame_shape or (1, 1)
        estimated_frame_bytes = max(1, int(frame_shape[0]) * int(frame_shape[1]) * 3)
        cache_budget = self._frame_cache.stats().max_bytes
        # Use at most one quarter of the decoded-frame cache per window. This
        # remains useful at 4K while never creating an unbounded preload.
        budgeted_frames = max(1, cache_budget // estimated_frame_bytes // 4)
        window_size = min(30, budgeted_frames)
        start_index = max(0, int(target_index) - window_size + 1)
        capture.set(self._cv2.CAP_PROP_POS_FRAMES, start_index)
        target_frame: np.ndarray | None = None
        for frame_index in range(start_index, int(target_index) + 1):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"Failed to decode reverse scrub window {start_index}..{target_index}"
                )
            frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            stored = self._store_decoded_frame(frame_index, frame_rgb)
            if frame_index == target_index:
                target_frame = stored
        self._next_read_index = int(target_index) + 1
        if target_frame is None:
            raise RuntimeError(f"Failed to read frame {target_index} from {self.path}")
        return target_frame

    def wait_until_preloaded(self, timeout_seconds: float | None = None) -> bool:
        """Wait for an eligible full-video preload; primarily useful to tests."""

        future = self._preload_future
        if future is None:
            return False
        try:
            future.result(timeout=timeout_seconds)
        except Exception:
            return False
        return True

    def _preload_all_frames(self) -> None:
        """Sequentially decode an entire fitting video on an independent capture."""

        capture = self._cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            return
        try:
            for frame_index in range(self._frame_count):
                with self._lifecycle_lock:
                    if self._closed:
                        return
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    return
                frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
                try:
                    self._store_decoded_frame(frame_index, frame_rgb)
                except RuntimeError:
                    return
        finally:
            capture.release()

    def close(self) -> None:
        with self._capture_lock:
            capture = getattr(self, "_capture", None)
            self._capture = None
            self._last_preview_index = None
            if capture is not None:
                capture.release()
        super().close()
        preload_future = self._preload_future
        self._preload_future = None
        preload_executor = self._preload_executor
        self._preload_executor = None
        if preload_future is not None:
            preload_future.cancel()
        if preload_executor is not None:
            preload_executor.shutdown(wait=False, cancel_futures=True)

    def _clamp_index(self, index: int) -> int:
        if not 0 <= int(index) < self.frame_count:
            raise IndexError(f"Frame index out of range: {index}")
        return int(index)


def open_media_source(
    path: str | Path,
    *,
    cache_bytes: int = 2048 * 1024 * 1024,
    prefetch_workers: int = 0,
) -> FrameProvider:
    """Open a supported media source from a folder or video file."""
    source = Path(path).resolve()
    if source.is_dir():
        return ImageFolderFrameProvider(source, cache_bytes=cache_bytes, prefetch_workers=prefetch_workers)
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        return VideoFrameProvider(source, cache_bytes=cache_bytes)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageFolderFrameProvider(
            source.parent,
            cache_bytes=cache_bytes,
            prefetch_workers=prefetch_workers,
        )
    raise ValueError(f"Unsupported media source: {source}")
