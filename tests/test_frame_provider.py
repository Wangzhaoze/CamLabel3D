from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from camlabel3d.core.frame_provider import ImageFolderFrameProvider


class ImageFolderFrameProviderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
