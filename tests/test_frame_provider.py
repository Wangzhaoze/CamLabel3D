from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image

from camlabel3d.core.frame_provider import ImageFolderFrameProvider


class ImageFolderFrameProviderTests(unittest.TestCase):
    @staticmethod
    def _write_images(folder: Path, count: int = 2) -> None:
        for index in range(count):
            image = Image.fromarray(np.full((4, 5, 3), index + 1, dtype=np.uint8))
            image.save(folder / f"frame{index}.png")

    def test_natural_sort_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            for name, color in [
                ("frame10.png", 10),
                ("frame2.png", 20),
                ("frame1.png", 30),
            ]:
                image = Image.fromarray(np.full((4, 5, 3), color, dtype=np.uint8))
                image.save(folder / name)

            provider = ImageFolderFrameProvider(folder)
            self.assertEqual(provider.frame_count, 3)
            self.assertTrue(provider.get_image_path(0).endswith("frame1.png"))
            self.assertTrue(provider.get_image_path(1).endswith("frame2.png"))
            self.assertTrue(provider.get_image_path(2).endswith("frame10.png"))

            frame = provider.get_frame(1)
            self.assertEqual(frame.shape, (4, 5, 3))
            self.assertEqual(int(frame[0, 0, 0]), 20)

            provider.close()

    def test_close_is_nonblocking_and_running_prefetch_cannot_repopulate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            self._write_images(folder)
            provider = ImageFolderFrameProvider(folder, prefetch_workers=1)
            original_decode = provider._decode_frame
            decode_started = Event()
            allow_decode_to_finish = Event()

            def blocked_decode(index: int) -> np.ndarray:
                decode_started.set()
                if not allow_decode_to_finish.wait(timeout=5.0):
                    raise TimeoutError("test did not release the prefetch decode")
                return original_decode(index)

            provider._decode_frame = blocked_decode  # type: ignore[method-assign]
            provider.prefetch([0, 1])
            self.assertTrue(decode_started.wait(timeout=2.0))
            with provider._lifecycle_lock:
                running_future = provider._prefetch_futures[0]
                pending_future = provider._prefetch_futures[1]

            started_at = time.monotonic()
            provider.close()
            close_elapsed = time.monotonic() - started_at

            self.assertLess(close_elapsed, 0.5)
            self.assertEqual(provider._frame_cache.stats().entries, 0)
            self.assertTrue(pending_future.cancelled())
            provider.prefetch([1])
            self.assertEqual(provider._prefetch_futures, {})

            allow_decode_to_finish.set()
            with self.assertRaisesRegex(RuntimeError, "provider is closed"):
                running_future.result(timeout=2.0)
            self.assertEqual(provider._frame_cache.stats().entries, 0)
            self.assertNotIn(0, provider._frame_shapes)

    def test_close_is_idempotent_and_closed_provider_rejects_new_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            self._write_images(folder)
            provider = ImageFolderFrameProvider(folder, prefetch_workers=1)

            provider.close()
            provider.close()
            provider.prefetch([0, 1])

            self.assertIsNone(provider._prefetch_executor)
            self.assertEqual(provider._prefetch_futures, {})
            with self.assertRaisesRegex(RuntimeError, "provider is closed"):
                provider.get_frame(0)

    def test_folder_discovery_honors_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            self._write_images(folder)

            with self.assertRaisesRegex(RuntimeError, "discovery was canceled"):
                ImageFolderFrameProvider(folder, should_cancel=lambda: True)

    def test_broken_prefetch_executor_is_a_safe_noop(self) -> None:
        class BrokenExecutor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def submit(self, *_args: object, **_kwargs: object) -> object:
                raise RuntimeError("executor is unavailable")

            def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
                self.shutdown_calls += 1
                self.shutdown_args = (wait, cancel_futures)

        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            self._write_images(folder)
            provider = ImageFolderFrameProvider(folder, prefetch_workers=1)
            original_executor = provider._prefetch_executor
            self.assertIsNotNone(original_executor)
            original_executor.shutdown(wait=False, cancel_futures=True)
            broken_executor = BrokenExecutor()
            provider._prefetch_executor = broken_executor  # type: ignore[assignment]

            provider.prefetch([0])
            provider.close()
            provider.close()

            self.assertEqual(provider._prefetch_futures, {})
            self.assertEqual(broken_executor.shutdown_calls, 1)
            self.assertEqual(broken_executor.shutdown_args, (False, True))


if __name__ == "__main__":
    unittest.main()
