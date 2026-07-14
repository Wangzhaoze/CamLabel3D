from pathlib import Path

import numpy as np
from PIL import Image

from wilddet3d import build_model, preprocess
from wilddet3d.vis.visualize import draw_3d_boxes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "ckpts" / "wilddet3d_alldata_all_prompt_v1.0.pt"
IMAGE_PATH = SCRIPT_DIR / "1776075514918495264.png"

# Build model
model = build_model(
    checkpoint=str(CHECKPOINT_PATH),
    score_threshold=0.3,
    skip_pretrained=True,
    use_predicted_intrinsics=False,
    # Enable this ONLY if you will pass `depth_gt=...` to `model(...)`
    # (i.e. you preprocessed the image with `depth=`). Monocular callers
    # leave it off.
    # use_depth_input_test=True,
)

# Load and preprocess image
image = np.array(Image.open(IMAGE_PATH)).astype(np.float32)

# With known camera intrinsics
# intrinsics = np.load("intrinsics.npy")  # (3, 3)
intrinsics = np.array([[2304, 0, 1152], [0, 2304, 648], [0, 0, 1]], dtype=np.float32)
data = preprocess(image, intrinsics)

# Without intrinsics (uses default: focal=max(H,W), principal point at center)
# data = preprocess(image)

# With a known depth map (e.g., from LiDAR or stereo), pass it through
# the same preprocess. Depth must be (H, W) float32 in meters at the
# original image resolution; preprocess resizes + center-pads it to
# match the model's input_hw (same transforms eval uses).
#   depth = np.load("depth.npy")  # (H, W) float32, meters
#   data = preprocess(image, intrinsics, depth=depth)
# Omit `depth` (or pass None) to let the model use its monocular
# LingBot-Depth prediction instead. When depth is provided, also pass
# `depth_gt=data["depth_gt"].cuda()` to each model(...) call below.

# Text prompt: detect all instances of given categories
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_texts=["car", "person"],
    # depth_gt=data["depth_gt"].cuda(),  # include only if preprocess was called with depth=...
)

# boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
# # Box prompt (geometric): lift a 2D box to 3D (one-to-one)
# results = model(
#     images=data["images"].cuda(),
#     intrinsics=data["intrinsics"].cuda()[None],
#     input_hw=[data["input_hw"]],
#     original_hw=[data["original_hw"]],
#     padding=[data["padding"]],
#     input_boxes=[[100, 200, 300, 400]],  # pixel xyxy
#     prompt_text="geometric",
#     # depth_gt=data["depth_gt"].cuda(),  # include only if preprocess was called with depth=...
# )

# # Exemplar prompt: use a 2D box as visual exemplar, find all similar objects (one-to-many)
# results = model(
#     images=data["images"].cuda(),
#     intrinsics=data["intrinsics"].cuda()[None],
#     input_hw=[data["input_hw"]],
#     original_hw=[data["original_hw"]],
#     padding=[data["padding"]],
#     input_boxes=[[100, 200, 300, 400]],
#     prompt_text="visual",
#     # depth_gt=data["depth_gt"].cuda(),  # include only if preprocess was called with depth=...
# )

# # Point prompt
# results = model(
#     images=data["images"].cuda(),
#     intrinsics=data["intrinsics"].cuda()[None],
#     input_hw=[data["input_hw"]],
#     original_hw=[data["original_hw"]],
#     padding=[data["padding"]],
#     input_points=[[(150, 250, 1), (200, 300, 0)]],  # (x, y, label): 1=positive, 0=negative
#     prompt_text="geometric",
#     # depth_gt=data["depth_gt"].cuda(),  # include only if preprocess was called with depth=...
# )

# Visualize results
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
draw_3d_boxes(
    image=image.astype(np.uint8),
    boxes3d=boxes3d[0],
    intrinsics=intrinsics,
    scores_2d=scores_2d[0],
    scores_3d=scores_3d[0],
    class_ids=class_ids[0],
    class_names=["car", "person"],
    save_path="output.png",
    # Optional debug overlays (both default off):
    #   predicted 2D boxes (green):
    # boxes_2d=boxes[0],
    # draw_predicted_2d_boxes=True,
    #   user prompt boxes (red) / points (red pos, gray neg):
    # input_boxes=[[100, 200, 300, 400]],
    # input_points=[[(150, 250, 1)]],
    # draw_prompt=True,
)
print(boxes3d, class_ids)
