# 播放性能分析

本文记录旧 `segment-anything-2-ui` 播放链路的卡顿点，以及新播放链路的目标。

## 旧播放热点

- `MediaPlayer.next_frame()` 在 UI 线程同步解码帧。
- 正常播放时调用 `increase_by_step_size()`，实际会频繁 seek。
- `position_slider.setValue()` 通过 `valueChanged` 触发 `move_to_frame()`，形成额外 seek 路径。
- 每帧都会 resize 到 `1024x1024`、转 RGB、新建 `QPixmap`，并可能重新合成 mask overlay。
- `ImageLabel.set_image()` 每帧清空 prompt 状态。
- `paintEvent()` 每次重绘都打印日志。

## 新播放目标

- `VideoPlaybackThread` 后台顺序解码视频并发送 RGB 帧。
- `ImageSequencePlaybackThread` 后台逐张读取图片序列，不一次性加载全部图片。
- UI 线程只绘制最新帧和当前 overlay。
- `PlaybackBar.set_position()` 使用 signal blocking，程序化 slider 更新不会触发 seek。
- SAM2 初始化和 mask 传播不参与纯播放路径。

## Smoke 基线

使用示例视频：

```bash
uv run python -c "import cv2; cap=cv2.VideoCapture('test-video.mp4'); print(cap.isOpened(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)); ok, frame = cap.read(); print(ok, frame.shape if ok else None)"
```
