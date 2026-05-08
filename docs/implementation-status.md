# 实现状态

最后更新：2026-05-07

## 已交付能力

- 根目录 `pyproject.toml`，发行包名为 `sam2-studio`，命令入口为 `sam2-studio`。
- `torch`、`torchvision` 和 `ultralytics` 从核心依赖中剥离，作为用户按本机 CPU/CUDA 环境单独安装的推理后端。
- 新 `src/sam2_studio` 包，使用 PySide6。
- 不复用旧 `MediaPlayer` 的轻量视频播放链路。
- `VideoPlaybackThread` 后台顺序解码视频；`ImageSequencePlaybackThread` 后台按需读取图片序列；二者都支持 seek、generation guard 和 in-flight 帧回压。
- 自定义画布，支持原图坐标点/框 prompt、mask overlay 和等比例显示。
- Ultralytics SAM2 当前帧 adapter，输出原始尺寸 mask。
- 模型加载、当前帧分割、帧序列传播和导出均为异步任务。
- 当前帧导出：二值 PNG、NPZ、safetensors。
- 视频和图片序列传播：正向、反向、先正后反。
- `.sam2studio/` 项目保存/恢复。
- 全部已有 mask 帧的二值 PNG 序列导出。
- 图片序列目录作为输入媒体打开，并按自然排序作为帧序列。
- 多边形转 mask、笔刷增加/擦除 mask。
- Overlay PNG 导出。
- release smoke CLI、安装/发布/许可证/性能文档。
- 开发 smoke CLI：`sam2-studio-segment-smoke` / `python -m sam2_studio.tools.segment_smoke`。
- 基础测试覆盖 prompt、媒体读取、项目 round-trip、导出、backend adapter 和 worker。

## 已验证命令

```bash
uv pip install -e .
uv pip install torch torchvision
uv pip install "ultralytics>=8.4.47,<8.5"
uv run pytest
uv run python -m compileall src tests
uv run sam2-studio --smoke
uv run sam2-studio-release-smoke
uv run python -m sam2_studio.tools.segment_smoke --model ./sam2.1_l.pt --video ./test-video.mp4 --frame 0 --point 960,540 --out /tmp/sam2-studio-mask.png
```

## 已知后续项

- 如果 Ultralytics 暴露稳定的非首帧 video state API，可替换当前临时 clip 传播方案。
- 如需更强 GUI 回归保障，可增加 `pytest-qt` 测试。
- Linux/Windows 桌面打包产物需要在明确第三方许可证姿态后单独准备。
