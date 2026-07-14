from __future__ import annotations

import unittest

from camlabel3d.runtime import (
    checkpoint_root,
    default_checkpoint_path,
    default_dataset_config_path,
    project_root,
    wilddet3d_root,
)


class RuntimePathTests(unittest.TestCase):
    def test_default_paths_follow_repo_layout(self) -> None:
        root = project_root()
        self.assertEqual(default_dataset_config_path(), root / "configs" / "dataset_sources.json")
        self.assertEqual(checkpoint_root(), root / "ckpts")
        self.assertEqual(default_checkpoint_path(), root / "ckpts" / "wilddet3d_alldata_all_prompt_v1.0.pt")

    def test_wilddet3d_root_prefers_workers_layout(self) -> None:
        self.assertEqual(wilddet3d_root(), project_root() / "workers" / "WildDet3D")


if __name__ == "__main__":
    unittest.main()
