# CamLabel3D 架构与性能约束

本文描述当前桌面应用的分层、线程模型和性能边界。目标不是让所有工作同时运行，而是让耗时工作离开 Qt 主线程，并用有界队列、快照和资源预算避免线程过量、内存失控与界面卡顿。

## 1. 分层结构

```text
camlabel3d/
├── app.py                  # 进程入口；先应用资源策略，再构建 Qt 应用
├── runtime_config.py       # CPU、GPU、缓存、去抖等统一运行时策略
├── application/            # 用例编排与依赖组装，不依赖具体 Qt 控件
│   ├── context.py          # Composition Root，集中创建共享服务
│   ├── indexes.py          # record/outlier 只读查询索引
│   └── source_service.py   # 数据源打开、provider 构造、会话路径准备
├── core/                   # 领域模型和可复用算法
│   ├── detector.py         # WildDet3D 适配器、模型生命周期、预览渲染
│   ├── frame_cache.py      # 有内存上限的线程安全 LRU 帧缓存
│   ├── frame_provider.py   # 图片/视频读取与邻帧预取
│   ├── geometry.py         # 3D 几何、投影和批量 overlay
│   ├── processing.py       # 异常规则、批处理和 CPU 并行执行
│   ├── postprocess.py      # 后处理会话、阶段和有界 undo
│   └── tracking.py         # 跟踪算法及协作式取消
├── io/                     # CSV 等持久化适配器
├── diagnostics/            # 可独立运行的环境/GPU 内核诊断
└── ui/
    ├── main_window.py      # UI 状态与用例协调；不执行重 I/O/计算
    ├── canvas.py           # Qt 绘制和交互
    ├── image_utils.py      # 后台可执行的 PIL -> QImage 转换
    └── workers/            # 推理、预览、持久化、源加载、处理任务
```

依赖方向为 `ui -> application -> core/io`。`ApplicationContext` 是共享对象的唯一组装入口；上游 WildDet3D 被限制在 `DetectorAdapter` 和运行时路径适配层后面。`ui/worker.py` 仅保留为旧导入路径的兼容门面，新代码应直接使用 `ui/workers/`。

## 2. 线程与资源模型

| 执行单元 | 职责 | 约束 |
| --- | --- | --- |
| Qt 主线程 | 控件、选择状态、`QPixmap`、轻量索引切换 | 禁止媒体探测、CSV I/O、模型推理和全量规则扫描 |
| Detection/Tracking worker | GPU 推理或跟踪计算 | 同一时间只运行一个前台计算用例；支持协作式取消 |
| Preview worker | 读取、投影、PIL -> `QImage` | 单个长生命周期线程；latest-wins，只保留最新请求 |
| Source/Function worker | 打开数据源和离线会话转换 | 一次一个；结果以快照返回 UI 线程提交 |
| Outlier worker | 异常规则分析 | 外层离开 UI；内部按 `cpu_workers` 有界并行 |
| CSV writer | 原子保存 CSV | 单写者；同一路径只保留最新待写 revision |
| Image prefetch pool | 解码相邻图片 | 最多 2 个线程；共享有界帧缓存 |

关键约束：

- 后台任务不能读取或写入 Qt 控件；启动任务前在 UI 线程生成普通 Python 快照。
- 后台结果只在 generation/source/frame 仍匹配时提交，过期预览直接丢弃。
- 关闭开始后不再接纳会改变 records 的迟到结果；先提交当前稳定快照，再排空单写者。
- 保存依赖按精确的 `(path, revision)` 等待；失败、超时、被合并和成功不会混淆，最终保存失败时应用保持打开。
- 视频 provider 用锁保护 seek/read；切换数据源时，旧 provider 等预览请求离开后再关闭。
- 图片目录发现支持协作式取消；provider 关闭后，仍在解码的预取任务不能重新写回缓存。
- CSV 使用同目录临时文件、`fsync` 和原子替换；备份与目标文件不会由多个 UI 操作并发写入。
- 模型构建状态在任何导入、显存释放或构建异常后都会唤醒等待者，不会因内部状态标志残留而永久等待。

## 3. 防卡顿策略

- 拖动预览直接提交给 latest-wins 单槽 worker；非拖动刷新默认去抖 35 ms，不积压过期帧。
- 自动保存默认去抖 350 ms；保存时复制稳定快照，不在 UI 线程做磁盘写入。
- 帧缓存默认上限 2048 MiB，按真实 `ndarray.nbytes` 淘汰；能完整容纳的视频由独立 capture 后台顺序预解码，超出上限则使用方向感知窗口。
- 拖动与松手预览均保留原分辨率图像和全部 3D BBX；NumPy 批量投影后由 OpenCV 单次绘制边、朝向和标签，并直接转为 QImage。
- `RecordIndex`/`OutlierIndex` 复用 frame、track、id 查询，避免 UI 刷新反复全表扫描。
- undo 默认只保留 20 个深拷贝快照；调用方已持有快照时转移所有权，避免双重复制。
- PyTorch 使用 `inference_mode`；可选 CUDA AMP。模型默认驻留显存，切换模型变体时先释放旧模型，避免双模型显存峰值。
- CPU 线程预算同时应用到 PyTorch 和常见 BLAS/OpenMP 库，防止多个线程池相乘导致抢占。

这些机制消除了已知的主线程重任务和无界增长点。系统负载、磁盘故障、第三方 CUDA kernel 或超大表格仍可能影响延迟，因此“流畅”应通过目标数据集的实机测试确认，而不是承诺绝对零延迟。

## 4. 运行时配置

所有变量都在进程启动时读取；修改后需重启应用。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CAMLABEL3D_DEVICE` | `auto` | `auto`、`cpu`、`cuda` 或 `cuda:N` |
| `CAMLABEL3D_CPU_WORKERS` | `min(8, CPU-1)` | 应用级 CPU 并行上限 |
| `CAMLABEL3D_TORCH_THREADS` | 同 CPU workers | PyTorch/BLAS 线程预算 |
| `CAMLABEL3D_TORCH_INTEROP_THREADS` | 最多 2 | PyTorch 算子间并行预算 |
| `CAMLABEL3D_FRAME_CACHE_MB` | `2048` | 解码帧 LRU 内存上限；`0` 禁用 |
| `CAMLABEL3D_PRELOAD_VIDEO_FRAMES` | `true` | 缓存可容纳时后台预解码完整原分辨率视频 |
| `CAMLABEL3D_PREVIEW_DEBOUNCE_MS` | `35` | 预览去抖时间 |
| `CAMLABEL3D_AUTOSAVE_DEBOUNCE_MS` | `350` | 自动保存去抖时间 |
| `CAMLABEL3D_KEEP_MODEL_LOADED` | `true` | 推理后保留模型，减少重复加载卡顿 |
| `CAMLABEL3D_ENABLE_AMP` | `false` | CUDA 推理启用 FP16 autocast；需对目标模型验证精度 |

示例：

```powershell
$env:CAMLABEL3D_DEVICE = "cuda:0"
$env:CAMLABEL3D_CPU_WORKERS = "6"
$env:CAMLABEL3D_TORCH_THREADS = "6"
$env:CAMLABEL3D_FRAME_CACHE_MB = "2048"
python -m camlabel3d
```

## 5. GPU 部署边界

`torch.cuda.is_available()` 只说明 PyTorch 能看到 GPU，不说明自定义扩展包含当前显卡的 kernel。部署后必须执行：

```powershell
python -m camlabel3d.diagnostics.gpu
```

该命令会真实调用并同步一次 `vis4d_cuda_ops` CUDA kernel。`DetectorAdapter` 首次选择 CUDA 时也会在隔离子进程中执行同一预检，失败会在加载数 GB 模型前给出明确错误，且不会污染主进程 CUDA context。若出现 `no kernel image is available`，应按诊断输出设置 `TORCH_CUDA_ARCH_LIST` 后重编译扩展；RTX 3060 对应 `8.6 / sm_86`。完整步骤见 [INSTALL_CAMLABEL3D.md](INSTALL_CAMLABEL3D.md)。

## 6. 后续演进边界

当前 `MainWindow` 仍承担较多 UI 状态协调。新增功能应优先放进 application 用例或独立 controller，不要继续把算法和 I/O 写回窗口类。若目标数据达到几十万条记录，可进一步把 `QTableWidget` 替换为基于 `QAbstractTableModel` 的虚拟化表格。若要求立即终止失控的第三方 GPU kernel，则需把推理提升为独立进程；线程级取消只能在 kernel 返回后的检查点生效。
