# 性能说明

## 当前保护措施

- 视频播放使用后台解码线程，不在 UI 线程每帧 seek/read。
- slider 的程序化更新会阻断信号，避免反馈式 seek。
- 播放帧投递使用 in-flight guard，避免排队堆积大量 RGB 大帧。
- 模型加载、当前帧分割、帧序列传播和导出都运行在 worker 线程中。
- 分割和传播结果带有 generation、request id、模型路径等校验，旧任务不会覆盖当前状态。
- 全部帧 PNG 导出基于 session 中的稀疏 mask store，不依赖画布截图。

## 终端出现多次 Ultralytics banner

当前帧分割使用 `SAM2Predictor`，视频或图片序列传播使用 `SAM2VideoPredictor`。它们是不同 predictor，因此第一次当前帧分割和第一次传播可能分别打印一次 Ultralytics 设备信息。`先正后反传播` 会顺序执行两个传播方向，每个方向目前也会创建独立的视频 predictor。

## 显存占用说明

- 当前帧分割模型会保留在内存/显存中，以便后续追加 prompt 更快。
- 传播任务结束后会删除视频 predictor，并主动调用 `torch.cuda.empty_cache()`。
- PyTorch 可能仍保留 CUDA cache，因此系统工具看到的显存不一定立刻降到 0，这通常不是泄漏。

## 已知限制

- Ultralytics/PyTorch 在一次长时间 GPU 调用内部不能被 Python 强制中断。取消是协作式的，会在传播开始前、临时 clip 创建期间或 streamed predictor 产出下一帧后生效。
- 非首帧开始传播时会创建临时 MJPG clip，以便使用 Ultralytics 高层 API 的首帧 prompt 路径。这个实现正确但不如未来低层 video state 接口高效。

## 人工性能检查

- 打开 `test-video.mp4`，确认不加载 SAM2 也能开始播放。
- 暂停后反复 seek，确认目标帧能快速显示且不会破坏播放状态。
- 分别在无 mask 和有 overlay 的情况下播放，观察 CPU/GPU 占用。
- 对短片段运行传播，并和直接 `VideoPropagationTask` smoke 的速度对比。
