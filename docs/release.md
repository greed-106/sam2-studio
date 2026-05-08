# 发布检查清单

## 干净环境安装验证

在项目根目录执行：

```bash
python -m venv /tmp/sam2-studio-release-smoke
source /tmp/sam2-studio-release-smoke/bin/activate
python -m pip install -U pip
python -m pip install build
python -m build
pip install dist/sam2_studio-*.whl
sam2-studio --smoke
sam2-studio-release-smoke
```

这一步只验证基础包和命令入口。完整推理 smoke 需要先单独安装 PyTorch 和 Ultralytics。

## GPU 分割 smoke

```bash
uv pip install torch torchvision
uv pip install "ultralytics>=8.4.47,<8.5"
sam2-studio-segment-smoke --model ./sam2.1_l.pt --video ./test-video.mp4 --frame 0 --point 960,540 --out /tmp/sam2-studio-mask.png
```

期望结果：生成一个非空、原视频尺寸的二值 PNG mask。

## GUI 人工验收

- 启动 `sam2-studio`。
- 选择 SAM2 模型。
- 打开图片，添加点 prompt，导出当前二值掩码和叠加预览图。
- 打开视频，seek、播放、暂停，运行短范围传播并取消一次传播。
- 保存 `.sam2studio` 项目，关闭应用，再重新打开并加载项目。

## 发布前确认

- `LICENSE` 与 `pyproject.toml` 的项目自有代码许可证一致。
- `docs/licenses.md` 已说明 Ultralytics、Qt/PySide6、模型权重和第三方依赖义务。
- 不要把模型权重打入 wheel，除非权重许可证明确允许。
- 安装推理后端后运行 `uv run pytest`、`uv run python -m compileall src tests`、`uv run sam2-studio --smoke` 和模型 smoke。
