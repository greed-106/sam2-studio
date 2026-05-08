# 现代化重构计划归档

本文原本记录从旧 `segment-anything-2-ui` 迁移到新实现的开发计划。当前用户可见项目名已经更改为 SAM2 Studio，项目定位也收敛为“掩码生成器”，而不是完整数据标注平台。

当前功能、按钮含义和导出格式请以这些文档为准：

- `README.md`
- `docs/features.md`
- `docs/export-guide.md`
- `docs/video-segmentation-tutorial.md`
- `docs/implementation-status.md`

历史计划中曾提到多种标注数据集导出格式；这些功能已经从当前产品范围中移除。当前保留的导出能力为：当前帧二值 PNG/NPZ/safetensors、全部帧二值 PNG 序列和当前帧叠加预览 PNG。
