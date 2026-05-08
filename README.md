# SAM2 Studio

SAM2 Studio 是一个基于 PySide6 和 Ultralytics SAM2/SAM2.1 的交互式掩码生成器。它面向“从图片或视频生成原始尺寸 mask”的工作流，而不是完整数据集标注平台。

## 当前能力

- 打开单张图片、常见视频和图片序列目录。
- 使用正点、负点、框选、多边形和笔刷编辑 mask。
- 对当前帧运行 SAM2 分割，对视频或图片序列执行正向、反向或“先正后反”传播。
- 保存和恢复 `.sam2studio/` 项目，包含媒体引用、对象、prompt、mask 和模型设置。
- 导出原始尺寸 mask：当前帧二值 PNG、NPZ、safetensors、全部帧二值 PNG 序列，以及当前帧叠加预览 PNG。

## 安装

推荐使用 `uv`：

```bash
uv venv
uv pip install -e ".[dev]"
uv run sam2-studio
```

已验证的 Ultralytics 版本范围是 `>=8.4.47,<8.5`。如果需要 CUDA，请先按 PyTorch 官方说明安装匹配显卡驱动的 PyTorch，再安装本项目。

## 运行

```bash
uv run sam2-studio
```

启动后使用 `选择模型` 按钮选择本地 SAM2/SAM2.1 `.pt` 权重文件。模型权重不会随本项目打包。

## 基础验证

```bash
uv run python -m sam2_studio.app.main --smoke
uv run python -m sam2_studio.tools.release_smoke
uv run python -m sam2_studio.tools.segment_smoke --model ./sam2.1_l.pt --video ./test-video.mp4 --frame 0 --point 960,540 --out /tmp/sam2-studio-mask.png
```

## 重要说明

- 终端中出现多次 `Ultralytics ... CUDA ...` banner 通常表示不同 predictor 被初始化。例如当前帧分割使用 `SAM2Predictor`，视频传播使用 `SAM2VideoPredictor`，它们是两条不同路径。
- `导出当前二值掩码` 只导出当前帧，PNG 像素为背景 `0`、前景 `255`。
- 传播后的整段结果请使用 `导出全部二值PNG`。
- 本项目自有代码使用 MIT 许可证。运行或分发包含 Ultralytics 的可执行应用时，必须遵守 Ultralytics AGPL-3.0 或商业授权条款。

## 文档入口

- `docs/features.md`：功能说明和按钮解释。
- `docs/video-segmentation-tutorial.md`：视频掩码生成完整教程。
- `docs/export-guide.md`：导出格式详细说明。
- `docs/installation.md`：安装说明。
- `docs/licenses.md`：许可证和第三方依赖说明。
