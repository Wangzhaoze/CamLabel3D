# Vis4D Cuda Operations

## Installation

Install a CUDA-enabled PyTorch build and a compatible CUDA Toolkit first. Set
`TORCH_CUDA_ARCH_LIST` explicitly when cross-compiling or when no GPU is visible
to the build process. An RTX 3060 has compute capability `8.6` (`sm_86`).

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
export VIS4D_CUDA_OPS_BUILD_CUDA="1"
export MAX_JOBS="2"

pip install -v . --no-build-isolation --no-cache-dir
```

PowerShell uses `$env:TORCH_CUDA_ARCH_LIST = "8.6"` (and the equivalent
syntax for the other variables). `MAX_JOBS` is optional: the setup script
defaults to at most four concurrent compiler jobs and preserves an explicit
override. Set `VIS4D_CUDA_OPS_BUILD_CUDA=0` only for an intentional CPU-only
build. `FORCE_CUDA=1` remains available as a compatibility alias.

PyTorch's extension builder consumes `TORCH_CUDA_ARCH_LIST`; multiple targets
can be separated by semicolons, and `+PTX` can be used when forward-compatible
PTX is deliberately required. Building only for the deployed GPU minimizes
compile time and binary size.

## Usage
```python
import torch
from vis4d_cuda_ops import ms_deform_attn_forward, ms_deform_attn_backward
...
```

## Add a new Op:
1. Add cuda and cpu ops.
2. Declare its Python interface in `src/vision.cpp`.
