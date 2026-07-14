# CamLabel3D / WildDet3D Windows Install

这份文档只使用显式 `pip` 命令，不使用任何 `requirements.txt`。

适用目录结构：

```text
D:\CamLabel3D
├── camlabel3d
├── ckpts
├── configs
├── docs
├── workers
│   └── WildDet3D
├── run_camlabel3d_ui.py
└── run_camlabel3d_postprocess.py
```

建议在 PowerShell 中执行。

## 1. 创建 conda 环境

```powershell
conda create -n camlabel3d python=3.11.15 -y
conda activate camlabel3d
python -m pip install --upgrade pip wheel "setuptools<80"
```

## 2. 进入仓库根目录

```powershell
Set-Location D:\CamLabel3D
```

## 3. 安装 PyTorch

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

## 4. 安装 WildDet3D 基础依赖

`triton` 在当前 Windows 环境下使用 `triton-windows`。

```powershell
python -m pip install vis4d==1.0.0
python -m pip install triton-windows==3.1.0.post17
python -m pip install numpy Pillow einops timm transformers huggingface_hub ftfy==6.1.1 regex "iopath>=0.1.10" typing_extensions opencv-python matplotlib pycocotools pyquaternion scipy terminaltables ml_collections tqdm pyarrow trimesh lightning "jsonargparse[signatures]" cloudpickle devtools termcolor h5py safetensors
python -m pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
```

## 5. 安装交互 Demo 依赖

如果你要运行 `WildDet3D\demo\huggingface\app.py`，再执行这一段：

```powershell
python -m pip install "gradio>=5.0.0" open_clip_torch pygltflib
```

如果你要运行新的桌面标注 UI，再执行这一段：

```powershell
python -m pip install PySide6
```

## 6. 安装本地 CUDA 扩展

```powershell
python -m pip install --no-cache-dir --no-build-isolation .\workers\WildDet3D\vis4d_cuda_ops
```

## 7. 下载模型权重

```powershell
New-Item -ItemType Directory -Force -Path .\ckpts | Out-Null

curl.exe -L -C - --output ".\ckpts\wilddet3d_alldata_all_prompt_v1.0.pt" "https://huggingface.co/allenai/WildDet3D/resolve/main/wilddet3d_alldata_all_prompt_v1.0.pt?download=true"

curl.exe -L -C - --output ".\ckpts\lingbot_depth_model.pt" "https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14/resolve/main/model.pt?download=true"
```

## 8. 快速验证

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
Set-Location .\workers\WildDet3D
python -c "import torch; import vis4d_cuda_ops; from wilddet3d import build_model, preprocess; print('wilddet3d import ok')"
Set-Location D:\CamLabel3D
```

## 9. 运行 Demo

仓库内的推理 demo：

```powershell
Set-Location .\workers\WildDet3D
python .\demo.py
```

交互式 Gradio demo：

```powershell
Set-Location .\workers\WildDet3D
python .\demo\huggingface\app.py
```

桌面单帧检测标注 UI：

```powershell
Set-Location D:\CamLabel3D
python .\run_camlabel3d_ui.py
```

默认会启动在：

```text
http://0.0.0.0:7860
```

## 10. 备注

- 数据集配置文件默认位于 `.\configs\dataset_sources.json`。
- `vis4d_cuda_ops` 是本地编译安装，机器上需要可用的 CUDA 开发环境和 C++ 编译工具。
- 当前这套目录结构默认从仓库根目录下的 `.\ckpts` 读取模型权重，从 `.\workers\WildDet3D` 加载上游 WildDet3D 代码。
