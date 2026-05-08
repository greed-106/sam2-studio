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
uv run sam2-studio
```

## 使用 pip 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install .
sam2-studio
```

## CUDA / CPU / MPS 说明

- CUDA：如果你需要指定 CUDA 版本，请先按 PyTorch 官方说明安装匹配驱动的 `torch` wheel，再安装 SAM2 Studio。
- CPU：可以运行，但 SAM2 推理会明显变慢。
- MPS：取决于 PyTorch 和 Ultralytics 在你的 macOS/PyTorch 组合下的支持情况。

## 模型权重

项目不包含模型权重。请根据你的使用许可下载 SAM2/SAM2.1 权重，然后在 UI 中点击 `选择模型`，或在命令行 smoke 工具中使用 `--model` 指定。

## 常见问题

- 如果终端打印两次 `Ultralytics ... CUDA ...`，通常是因为当前帧分割和视频传播分别初始化了不同 predictor，不代表同一个 predictor 重复初始化两次。
- 如果传播结束后显存没有立刻降到 0，这是 PyTorch CUDA cache 的正常行为。传播任务结束后会主动删除视频 predictor 并调用 `torch.cuda.empty_cache()`，但已加载的当前帧分割模型会保留以便下一次点击更快响应。
