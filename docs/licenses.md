# 许可证说明

SAM2 Studio 项目自有代码声明为 MIT 许可证，全文见根目录 `LICENSE`。

请注意：这不代表完整可运行应用整体都是 MIT 约束。SAM2 Studio 的分割功能依赖第三方组件和模型权重，分发前必须分别确认授权。

## Ultralytics

Ultralytics 是 SAM2 推理的必需运行时依赖。本项目代码中直接使用 `SAM2Predictor` 和 `SAM2VideoPredictor`。运行或分发包含 Ultralytics 的 SAM2 Studio 时，必须遵守 Ultralytics 的 AGPL-3.0 条款，或持有有效商业授权。不要把包含 Ultralytics 的可运行应用描述成“纯 MIT 许可”。

## PySide6 / Qt

PySide6 和 Qt 有 LGPL / 商业授权要求。桌面打包分发时需要保留必要许可证文本，并满足 Qt 动态链接和用户替换库等义务，或使用合适的商业授权。

## 其他依赖

PyTorch、OpenCV、NumPy、Pillow、safetensors 等依赖都有各自许可证。发布二进制包前应随包提供第三方 notices 或许可证链接。

## 模型权重

SAM2/SAM2.1 权重不随本项目打包。请仅在权重许可允许的范围内下载、使用和再分发。

## 旧项目目录

仓库中的旧 `segment-anything-2-ui` 目录作为参考材料保留。发布前不要直接复制旧代码，除非已经确认其原始许可证允许你的用途。
