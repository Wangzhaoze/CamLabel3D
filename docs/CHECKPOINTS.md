# Checkpoint Download Guide

This repository does not commit model weights. Download the required checkpoints into the repository-level `ckpts/` directory.

## Target Directory

From the repository root:

```powershell
New-Item -ItemType Directory -Force -Path .\ckpts | Out-Null
```

## Required for CamLabel3D

### 1. WildDet3D main checkpoint

File name:

```text
wilddet3d_alldata_all_prompt_v1.0.pt
```

Download with `curl`:

```powershell
curl.exe -L -C - --output ".\ckpts\wilddet3d_alldata_all_prompt_v1.0.pt" "https://huggingface.co/allenai/WildDet3D/resolve/main/wilddet3d_alldata_all_prompt_v1.0.pt?download=true"
```

### 2. LingBot-Depth checkpoint

File name:

```text
lingbot_depth_model.pt
```

Download with `curl`:

```powershell
curl.exe -L -C - --output ".\ckpts\lingbot_depth_model.pt" "https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14/resolve/main/model.pt?download=true"
```

## Optional

### SAM3 checkpoint

CamLabel3D itself runs WildDet3D with `skip_pretrained=True`, so the desktop app does not require a separate SAM3 checkpoint for normal use.

Some standalone upstream WildDet3D workflows may still expect a SAM3 weight file. If you need it, place it in `ckpts/` or in the path expected by the specific upstream script you are using.

## Alternative: Hugging Face CLI

```powershell
huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir .\ckpts
```

## Final Expected Layout

```text
CamLabel3D/
└── ckpts/
    ├── wilddet3d_alldata_all_prompt_v1.0.pt
    └── lingbot_depth_model.pt
```

## Verification

Once both files are in place, you can launch the desktop app from the repository root:

```powershell
python -m camlabel3d
# 或使用便捷入口：python .\ui.py
```
