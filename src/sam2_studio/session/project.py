from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sam2_studio import __version__
from sam2_studio.interaction.prompts import BoxPrompt, PointPrompt, PromptBatch
from sam2_studio.session.session import AnnotationSession, ObjectState


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectModelConfig:
    path: str
    imgsz: int = 1024
    conf: float = 0.25
    device: str = ""
    half: bool = False


@dataclass(frozen=True)
class ProjectData:
    session: AnnotationSession
    model: ProjectModelConfig
    media_metadata: dict[str, Any]


def _prompt_to_dict(frame_idx: int, object_id: int, prompts: PromptBatch) -> dict[str, Any]:
    box = None
    if prompts.box is not None:
        box = {
            "x1": prompts.box.x1,
            "y1": prompts.box.y1,
            "x2": prompts.box.x2,
            "y2": prompts.box.y2,
        }
    return {
        "frame_idx": frame_idx,
        "object_id": object_id,
        "points": [{"x": p.x, "y": p.y, "label": p.label} for p in prompts.points],
        "box": box,
    }


def _prompt_from_dict(data: dict[str, Any]) -> PromptBatch:
    prompts = PromptBatch()
    for point in data.get("points", []):
        prompts.points.append(PointPrompt(float(point["x"]), float(point["y"]), int(point["label"])))
    box = data.get("box")
    if box is not None:
        prompts.box = BoxPrompt(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])).normalized()
    return prompts


def _media_metadata(session: AnnotationSession, frame_count: int | None = None, fps: float | None = None) -> dict[str, Any]:
    if session.media_path is None:
        raise ValueError("Cannot save a project before opening media.")
    path = Path(session.media_path)
    stat = path.stat() if path.exists() else None
    height = width = None
    if session.current_image_rgb is not None:
        height, width = session.current_image_rgb.shape[:2]
    return {
        "path": str(path),
        "kind": session.media_kind,
        "size_bytes": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "frame_names": list(session.frame_names),
    }


def save_project(
    session: AnnotationSession,
    path: str | Path,
    model: ProjectModelConfig,
    frame_count: int | None = None,
    fps: float | None = None,
) -> Path:
    project_dir = Path(path)
    if project_dir.suffix != ".sam2studio":
        project_dir = project_dir.with_suffix(".sam2studio")
    project_dir.mkdir(parents=True, exist_ok=True)

    mask_arrays: dict[str, np.ndarray] = {}
    mask_manifest: list[dict[str, Any]] = []
    for frame_idx, object_masks in sorted(session.masks.items()):
        for object_id, mask in sorted(object_masks.items()):
            key = f"mask_{frame_idx:06d}_{object_id:06d}"
            mask_arrays[key] = mask.astype(bool)
            mask_manifest.append({"frame_idx": frame_idx, "object_id": object_id, "key": key})
    np.savez_compressed(project_dir / "masks.npz", **mask_arrays)

    prompts = []
    for frame_idx, object_prompts in sorted(session.prompts.items()):
        for object_id, prompt_batch in sorted(object_prompts.items()):
            prompts.append(_prompt_to_dict(frame_idx, object_id, prompt_batch))

    document = {
        "format": "sam2-studio-project",
        "schema_version": SCHEMA_VERSION,
        "created_with": {"app": "sam2-studio", "version": __version__},
        "media": _media_metadata(session, frame_count=frame_count, fps=fps),
        "model": {"path": model.path, "imgsz": model.imgsz, "conf": model.conf, "device": model.device, "half": model.half},
        "session": {"current_frame_idx": session.current_frame_idx, "current_object_id": session.current_object_id},
        "objects": [
            {
                "object_id": obj.object_id,
                "name": obj.name,
                "color": list(obj.color),
                "visible": obj.visible,
            }
            for obj in sorted(session.objects.values(), key=lambda item: item.object_id)
        ],
        "prompts": prompts,
        "masks": mask_manifest,
    }
    (project_dir / "project.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    return project_dir


def load_project(path: str | Path) -> ProjectData:
    project_dir = Path(path)
    document = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    if document.get("format") != "sam2-studio-project" or int(document.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Unsupported SAM2 Studio project format.")

    session = AnnotationSession()
    session.objects.clear()
    media = document["media"]
    session.media_path = media["path"]
    session.media_kind = media["kind"]
    session.frame_names = list(media.get("frame_names", []))
    for obj in document.get("objects", []):
        object_id = int(obj["object_id"])
        session.objects[object_id] = ObjectState(
            object_id=object_id,
            name=str(obj["name"]),
            color=tuple(int(c) for c in obj["color"]),
            visible=bool(obj.get("visible", True)),
        )

    for prompt_data in document.get("prompts", []):
        frame_idx = int(prompt_data["frame_idx"])
        object_id = int(prompt_data["object_id"])
        session.prompts.setdefault(frame_idx, {})[object_id] = _prompt_from_dict(prompt_data)

    masks_path = project_dir / "masks.npz"
    if masks_path.exists():
        loaded = np.load(masks_path)
        for mask_data in document.get("masks", []):
            frame_idx = int(mask_data["frame_idx"])
            object_id = int(mask_data["object_id"])
            session.set_mask(frame_idx, object_id, loaded[mask_data["key"]].astype(bool))

    session.current_frame_idx = int(document.get("session", {}).get("current_frame_idx", 0))
    session.current_object_id = int(document.get("session", {}).get("current_object_id", 0))
    model_data = document.get("model", {})
    model = ProjectModelConfig(
        path=str(model_data.get("path", "")),
        imgsz=int(model_data.get("imgsz", 1024)),
        conf=float(model_data.get("conf", 0.25)),
        device=str(model_data.get("device", "")),
        half=bool(model_data.get("half", False)),
    )
    return ProjectData(session=session, model=model, media_metadata=media)
