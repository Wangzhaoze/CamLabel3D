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
├── pyproject.toml
├── ui.py
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
python -m pip install "pydantic<2" vis4d==1.0.0
python -m pip install triton-windows==3.1.0.post17
python -m pip install ninja
python -m pip install numpy Pillow einops timm transformers huggingface_hub ftfy==6.1.1 regex "iopath>=0.1.10" typing_extensions opencv-python matplotlib pycocotools pyquaternion scipy terminaltables ml_collections tqdm pyarrow trimesh lightning "jsonargparse[signatures]" cloudpickle devtools termcolor h5py safetensors
python -m pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
```

## 5. 安装桌面 UI 依赖

运行 CamLabel3D 桌面 UI：

```powershell
python -m pip install PySide6
```

`vis4d==1.0.0` 要求 `pydantic<2`，而当前 Gradio/FastAPI 要求 Pydantic 2，因此不能在同一个可由 `pip check` 验证通过的环境中同时安装两套依赖。本指南只保证桌面 CamLabel3D 环境；不要在这个环境里继续安装 `gradio>=5`。如需上游 Hugging Face demo，请单独维护其环境并以当时的上游依赖说明为准。

## 6. 安装本地 CUDA 扩展

先确认本机安装了与 PyTorch 兼容的 CUDA Toolkit（包含 `nvcc`）和 MSVC C++ 编译工具。扩展编译架构由 `TORCH_CUDA_ARCH_LIST` 控制；RTX 3060 的计算能力是 `8.6`，对应 NVIDIA 架构标签 `sm_86`：

```powershell
nvcc --version

# RTX 3060: compute capability 8.6 / sm_86
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:VIS4D_CUDA_OPS_BUILD_CUDA = "1"

# 限制 Ninja 编译并发，避免编译时占满 CPU 和内存；可按机器资源调整
$env:MAX_JOBS = "2"

python -m pip install --no-cache-dir --no-build-isolation .\workers\WildDet3D\vis4d_cuda_ops
```

若普通 PowerShell 中找不到 `cl.exe`，或 PyTorch 提示 VS 环境已激活但未设置 `DISTUTILS_USE_SDK`，可通过 VS 2022 的 x64 环境执行同一构建（路径按实际 Edition 调整）：

```powershell
cmd.exe /d /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set "DISTUTILS_USE_SDK=1" && set "MSSdk=1" && python -m pip install --force-reinstall --no-deps --no-cache-dir --no-build-isolation .\workers\WildDet3D\vis4d_cuda_ops'
```

其他显卡可在 PyTorch 能看到 GPU 时查询计算能力，再把输出值设置给 `TORCH_CUDA_ARCH_LIST`：

```powershell
python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))"
```

构建脚本不会再只根据 `torch.cuda.is_available()` 决定是否编译 CUDA：设置了 `TORCH_CUDA_ARCH_LIST`、检测到 CUDA Toolkit，或显式设置 `VIS4D_CUDA_OPS_BUILD_CUDA=1` 时都会选择 CUDA 构建。没有可见 GPU 的构建机必须显式设置架构。若确实只需要 CPU 扩展，可设置 `VIS4D_CUDA_OPS_BUILD_CUDA=0`。

未设置 `MAX_JOBS` 时，构建脚本会保留至少一个 CPU 线程并把默认并发上限限制为 4；显式设置的值始终优先。

## 7. 下载模型权重

```powershell
New-Item -ItemType Directory -Force -Path .\ckpts | Out-Null

curl.exe -L -C - --output ".\ckpts\wilddet3d_alldata_all_prompt_v1.0.pt" "https://huggingface.co/allenai/WildDet3D/resolve/main/wilddet3d_alldata_all_prompt_v1.0.pt?download=true"

curl.exe -L -C - --output ".\ckpts\lingbot_depth_model.pt" "https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14/resolve/main/model.pt?download=true"
```

## 8. 快速验证

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -m pip check
python -m camlabel3d.diagnostics.gpu

Set-Location .\workers\WildDet3D
python -c "import torch; import vis4d_cuda_ops; from wilddet3d import build_model, preprocess; print('wilddet3d import ok')"
Set-Location D:\CamLabel3D
```

`camlabel3d.diagnostics.gpu` 不只检查模块能否导入：它会在 GPU 上实际执行一次 `vis4d_cuda_ops.ms_deform_attn_forward` CUDA kernel，并同步等待结果。成功时应看到 `PASS` 和当前设备的 `sm_XX`；脚本失败时返回非零退出码。机器可读输出可用：

```powershell
python -m camlabel3d.diagnostics.gpu --json
```

如果出现 `no kernel image is available`、`invalid device function`，或诊断提示扩展只包含 CPU 支持，按诊断输出的计算能力重新编译。例如 RTX 3060：

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:VIS4D_CUDA_OPS_BUILD_CUDA = "1"
$env:MAX_JOBS = "2"
python -m pip install --force-reinstall --no-deps --no-cache-dir --no-build-isolation .\workers\WildDet3D\vis4d_cuda_ops
python -m camlabel3d.diagnostics.gpu
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
python -m camlabel3d
```

也可以使用仓库根目录的便捷入口：

```powershell
python .\ui.py
```

这是 PySide6 桌面应用，会直接打开窗口，不会启动 Gradio Web 服务。

后处理 CLI：

```powershell
python -m camlabel3d.postprocess_cli --help
python .\run_camlabel3d_postprocess.py --help
```

## 10. CPU / GPU 资源选择

CamLabel3D 默认在 PyTorch 可用 CUDA 时使用 GPU，否则使用 CPU。应用启动时会读取以下有界资源策略：

| 变量 | 作用 | 备注 |
| --- | --- | --- |
| `CAMLABEL3D_DEVICE` | `auto` / `cpu` / `cuda` / `cuda:N` | 默认 `auto` |
| `CAMLABEL3D_CPU_WORKERS` | 异常分析等应用级并行数 | 默认保留一个逻辑 CPU，且最多 8 |
| `CAMLABEL3D_TORCH_THREADS` | PyTorch 和数值库线程预算 | 默认与 CPU workers 相同 |
| `CAMLABEL3D_TORCH_INTEROP_THREADS` | PyTorch 算子间线程数 | 默认最多 2 |
| `CAMLABEL3D_FRAME_CACHE_MB` | 解码帧 LRU 缓存上限 | 默认 2048；`0` 禁用 |
| `CAMLABEL3D_PRELOAD_VIDEO_FRAMES` | 后台预解码可容纳的完整视频 | 默认 `true`；保持原分辨率 RGB 帧 |
| `CAMLABEL3D_KEEP_MODEL_LOADED` | 推理后保留模型 | 默认 `true`，避免重复加载停顿 |
| `CAMLABEL3D_ENABLE_AMP` | CUDA FP16 autocast | 默认 `false`，启用前验证精度 |
| `CUDA_VISIBLE_DEVICES` | 控制 PyTorch 可见的 GPU | 必须在启动 CamLabel3D 前设置；可用 `-1` 禁用 CUDA |

应用会在导入 NumPy/PyTorch 前，用 `CAMLABEL3D_TORCH_THREADS` 为尚未显式设置的 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS` 设置同一预算，防止线程池叠加过量占用 CPU。

PowerShell 示例：

```powershell
# 强制使用 CPU，并为 UI/系统保留资源
$env:CAMLABEL3D_DEVICE = "cpu"
$env:CAMLABEL3D_CPU_WORKERS = "4"
$env:CAMLABEL3D_TORCH_THREADS = "4"
$env:CAMLABEL3D_FRAME_CACHE_MB = "2048"

python -m camlabel3d
```

完整线程模型与防卡顿约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 11. 备注

- 数据集配置文件默认位于 `.\configs\dataset_sources.json`。
- `vis4d_cuda_ops` 是本地编译安装，机器上需要可用的 CUDA 开发环境和 C++ 编译工具。
- `TORCH_CUDA_ARCH_LIST` 和 `MAX_JOBS` 只影响扩展构建；运行时 GPU 可见性仍由 `CUDA_VISIBLE_DEVICES` 控制。
- 当前这套目录结构默认从仓库根目录下的 `.\ckpts` 读取模型权重，从 `.\workers\WildDet3D` 加载上游 WildDet3D 代码。
