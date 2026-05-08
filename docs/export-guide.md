# 导出格式说明

## 通用约定

- 所有导出都使用原始媒体尺寸，不使用画布显示尺寸。
- PNG mask 是黑白二值图：背景 `0`，前景 `255`。
- 如果当前帧有多个对象，PNG 会把所有可见对象合并成一个二值前景；对象 id 不会写入 PNG。
- 如需保留对象 id 或每对象独立 mask，请使用 NPZ 或 safetensors。

## 导出当前二值掩码

范围：当前帧。

格式：

- `.png`：单张黑白二值 mask，背景 `0`，前景 `255`。
- `.npz`：保存 `object_ids` 和每对象 bool mask stack。
- `.safetensors`：保存 `label_map` tensor，并在 metadata 中记录 object ids。

适合：快速导出当前帧结果，或将某一帧 mask 交给外部工具。

## 导出全部二值PNG

范围：整个 session 中所有已有 mask 的帧。

输出：

- `frame_000000.png` 等逐帧黑白二值 mask。
- 如果输入是图片序列目录，输出 mask 文件名沿用源图片文件名的 stem，例如 `frame_001.jpg` 导出为 `frame_001.png`。
- `metadata.json`，记录媒体路径、对象信息和帧文件列表。

默认跳过没有 mask 的空帧。传播完成后，如果想导出整段视频的结果，应使用这个功能，而不是 `导出当前二值掩码`。

### metadata.json 字段说明

`metadata.json` 中的 `objects` 用于记录导出时的对象状态：

- `object_id`：SAM2 Studio 内部对象编号，从 `0` 开始。
- `mask_value`：该对象在二值 PNG 中对应的前景值。当前二值 PNG 会把所有对象合并为前景，因此所有对象都是 `255`。
- `name`：对象在 UI 中显示的名称，例如 `object 0`。
- `color`：对象在 UI 叠加预览中的 RGB 颜色，不影响二值 PNG 的像素值。
- `visible`：导出时该对象在 UI 中的可见状态记录。

`metadata.json` 中的 `frames` 用于记录导出的帧：

- `frame_idx`：从 `0` 开始的帧序号。
- `mask_file`：该帧导出的二值 mask 文件名。
- `source_file`：图片序列输入时对应的源图片文件名；普通视频输入时为 `null`。

## 导出当前叠加图

范围：当前帧。

输出：普通 RGB PNG，把 mask 半透明叠加到原图上。

注意：这是可视化预览，不是训练 mask，不含 alpha 通道，也不保留每对象二值 mask。

## 暂不提供的数据集标注导出

SAM2 Studio 当前定位为掩码生成器，暂时不提供数据集标注格式导出。后续如果项目定位扩展为标注工具，再单独设计这些格式。
