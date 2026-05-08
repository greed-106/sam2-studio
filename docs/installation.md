# 安装指南

## 环境要求

- Python `>=3.10`
- 本地 SAM2/SAM2.1 Ultralytics `.pt` 权重文件
- 推荐使用 CUDA GPU 进行交互式分割和视频传播
- 已验证 Ultralytics 版本范围：`>=8.4.47,<8.5`

## 使用 uv 安装

```bash
uv venv
uv pip install -e ".[dev]"

# CPU 示例
uv pip install torch torchvision
uv pip install "ultralytics>=8.4.47,<8.5"

uv run sam2-studio
```

如果使用 CUDA，请把上面的 PyTorch 安装命令替换为匹配你显卡驱动和 CUDA 版本的官方 wheel。例如 CUDA 11.8：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install "ultralytics>=8.4.47,<8.5"
uv run sam2-studio
```

`torch`、`torchvision` 和 `ultralytics` 被视为推理后端，不纳入项目核心依赖和 `uv.lock`。`uv run sam2-studio` 仍然可用；`uv run` 会确保项目依赖存在，但不会移除你用 `uv pip install` 额外安装的推理后端。

如果之后执行 `uv sync`，请使用：

```bash
uv sync --inexact
```

或者在 `uv sync` 后重新安装 PyTorch 和 Ultralytics。普通 `uv sync` 可能清理不在项目依赖中的额外包。

## 使用 pip 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install .
sam2-studio
```

## CUDA / CPU / MPS 说明

- CUDA：按 PyTorch 官方说明选择匹配驱动的 `torch` / `torchvision` wheel，再安装 `ultralytics>=8.4.47,<8.5`。
- CPU：可以运行，但 SAM2 推理会明显变慢。
- MPS：取决于 PyTorch 和 Ultralytics 在你的 macOS/PyTorch 组合下的支持情况。

启动 GUI 前，SAM2 Studio 会检查 `torch` 和 `ultralytics` 是否可导入。如果缺失，会在终端给出安装提示并退出。

## 模型权重

项目不包含模型权重。请根据你的使用许可下载 SAM2/SAM2.1 权重，然后在 UI 中点击 `选择模型`，或在命令行 smoke 工具中使用 `--model` 指定。

## 常见问题

- 如果终端打印两次 `Ultralytics ... CUDA ...`，通常是因为当前帧分割和视频传播分别初始化了不同 predictor，不代表同一个 predictor 重复初始化两次。
- 如果传播结束后显存没有立刻降到 0，这是 PyTorch CUDA cache 的正常行为。传播任务结束后会主动删除视频 predictor 并调用 `torch.cuda.empty_cache()`，但已加载的当前帧分割模型会保留以便下一次点击更快响应。
