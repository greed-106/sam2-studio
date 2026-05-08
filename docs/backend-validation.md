# 后端验证记录

最后更新：2026-05-07

## 验证环境

- `torch==2.11.0+cu130`
- CUDA 可用，GPU 为 NVIDIA GeForce RTX 3090
- `ultralytics==8.4.47`
- 模型：`./sam2.1_l.pt`
- 视频：`./test-video.mp4`，250 帧，25 FPS，1920x1080

## 图像 / 当前帧分割

验证命令：

```bash
uv run python -m sam2_studio.tools.segment_smoke --model ./sam2.1_l.pt --video ./test-video.mp4 --frame 0 --point 960,540 --out /tmp/sam2-studio-mask.png
```

观察输出：

```text
wrote /tmp/sam2-studio-mask.png size=(1920, 1080) masks=(1, 1080, 1920)
```

这说明 adapter 返回的是原始帧尺寸 mask，而不是 `1024x1024` 模型输入尺寸 mask。

## 视频传播

直接运行 `VideoPropagationTask` 做 3 帧 smoke，输出 3 张原始尺寸 mask：

```text
[(1080, 1920), (1080, 1920), (1080, 1920)]
```

当前实现说明：从任意帧开始传播时，会创建一个从提示帧开始的临时 MJPG clip，然后用 `SAM2VideoPredictor(stream=True)` 对该 clip 传播。这样可以稳定复用 Ultralytics 高层 API 的首帧 prompt 路径。该方案适合当前版本，但后续如果 Ultralytics 暴露稳定的非首帧 video state API，可以替换为更高效的实现。
