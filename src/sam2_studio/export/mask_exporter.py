from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from sam2_studio.session.session import AnnotationSession


def build_label_map(mask_items: list[tuple[int, np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    if not mask_items:
        raise ValueError("No masks are available for export.")
    height, width = mask_items[0][1].shape
    label_map = np.zeros((height, width), dtype=np.uint16)
    for object_id, mask, _color in mask_items:
        if object_id < 0 or object_id > 65534:
            raise ValueError("safetensors label map export requires object ids between 0 and 65534.")
        if mask.shape != (height, width):
            raise ValueError("All masks in a label map export must share the same shape.")
        label_map[mask.astype(bool)] = int(object_id) + 1
    return label_map


def build_binary_mask(mask_items: list[tuple[int, np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    if not mask_items:
        raise ValueError("No masks are available for export.")
    height, width = mask_items[0][1].shape
    binary = np.zeros((height, width), dtype=bool)
    for _object_id, mask, _color in mask_items:
        if mask.shape != (height, width):
            raise ValueError("All masks in a binary mask export must share the same shape.")
        binary |= mask.astype(bool)
    return binary.astype(np.uint8) * 255


def export_masks(mask_items: list[tuple[int, np.ndarray, tuple[int, int, int]]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix == ".png":
        Image.fromarray(build_binary_mask(mask_items), mode="L").save(output)
        return output

    if suffix == ".npz":
        object_ids = np.array([item[0] for item in mask_items], dtype=np.int32)
        masks = np.stack([item[1].astype(bool) for item in mask_items], axis=0)
        np.savez_compressed(output, object_ids=object_ids, masks=masks)
        return output

    if suffix == ".safetensors":
        label_map = torch.from_numpy(build_label_map(mask_items).astype(np.int64))
        metadata = {"object_ids": ",".join(str(item[0]) for item in mask_items)}
        save_file({"label_map": label_map}, str(output), metadata=metadata)
        return output

    raise ValueError("Export path must end with .png, .npz, or .safetensors")


def export_overlay(
    image_rgb: np.ndarray,
    mask_items: list[tuple[int, np.ndarray, tuple[int, int, int]]],
    path: str | Path,
    alpha: float = 0.45,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Overlay export requires an RGB image.")
    height, width = image_rgb.shape[:2]
    blended = image_rgb.astype(np.float32).copy()
    for _object_id, mask, color in mask_items:
        if mask.shape != (height, width):
            raise ValueError("Mask shape does not match image shape for overlay export.")
        color_arr = np.array(color, dtype=np.float32)
        active = mask.astype(bool)
        blended[active] = blended[active] * (1.0 - alpha) + color_arr * alpha
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB").save(output)
    return output


def export_video_masks(session: AnnotationSession, output_dir: str | Path, include_empty_frames: bool = False) -> list[Path]:
    """Export all available session masks as binary PNG sequence with metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if include_empty_frames:
        frame_ids = range(max(session.masks.keys(), default=-1) + 1)
    else:
        frame_ids = sorted(session.masks.keys())

    used_names: set[str] = set()
    frame_records: list[dict] = []
    for frame_idx in frame_ids:
        mask_items = session.mask_items_for_frame(frame_idx, visible_only=False)
        if not mask_items:
            continue
        output_name = _mask_filename_for_frame(session, frame_idx, used_names)
        frame_path = output / output_name
        export_masks(mask_items, frame_path)
        written.append(frame_path)
        frame_records.append(
            {
                "frame_idx": int(frame_idx),
                "mask_file": frame_path.name,
                "source_file": session.frame_names[frame_idx] if 0 <= frame_idx < len(session.frame_names) else None,
            }
        )

    metadata = {
        "format": "sam2-studio-binary-mask-sequence",
        "schema_version": 1,
        "media_path": session.media_path,
        "mask_encoding": {
            "type": "binary_png",
            "background_value": 0,
            "foreground_value": 255,
            "description": "每个 PNG 是黑白二值掩码；所有导出的对象合并为前景。",
        },
        "object_fields": {
            "object_id": "SAM2 Studio 内部对象编号。",
            "mask_value": "该对象在二值 PNG 中对应的前景像素值；当前所有对象合并为 255。",
            "name": "对象显示名称。",
            "color": "UI 叠加显示颜色，RGB 顺序。",
            "visible": "导出时该对象在 UI 中的可见状态记录。",
        },
        "frame_fields": {
            "frame_idx": "从 0 开始的帧序号。",
            "mask_file": "导出的二值掩码文件名。图片序列输入时默认沿用源图片文件名的 stem。",
            "source_file": "图片序列输入时对应的源图片文件名；普通视频输入时为 null。",
        },
        "objects": [
            {
                "object_id": obj.object_id,
                "mask_value": 255,
                "name": obj.name,
                "color": list(obj.color),
                "visible": obj.visible,
            }
            for obj in sorted(session.objects.values(), key=lambda item: item.object_id)
        ],
        "frames": frame_records,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written.append(output / "metadata.json")
    return written


def _mask_filename_for_frame(session: AnnotationSession, frame_idx: int, used_names: set[str]) -> str:
    if 0 <= frame_idx < len(session.frame_names):
        name = f"{Path(session.frame_names[frame_idx]).stem}.png"
    else:
        name = f"frame_{frame_idx:06d}.png"
    if name not in used_names:
        used_names.add(name)
        return name
    stem = Path(name).stem
    deduped = f"{stem}_{frame_idx:06d}.png"
    used_names.add(deduped)
    return deduped
