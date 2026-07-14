pip install huggingface_hub
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${WILDDET3D_CKPT_ROOT:-$SCRIPT_DIR/../../ckpts}"
mkdir -p "$TARGET_DIR"
huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir "$TARGET_DIR"
