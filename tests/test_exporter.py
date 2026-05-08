import json

import numpy as np
from safetensors.torch import load_file
from PIL import Image

from sam2_studio.export.mask_exporter import (
    build_label_map,
    export_masks,
    export_overlay,
    export_video_masks,
)
from sam2_studio.session.session import AnnotationSession


def test_build_label_map_preserves_object_ids() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    label_map = build_label_map([(3, mask, (1, 2, 3))])

    assert label_map.shape == (4, 5)
    assert label_map.max() == 4


def test_export_safetensors(tmp_path) -> None:
    mask = np.ones((2, 3), dtype=bool)
    output = tmp_path / "mask.safetensors"

    export_masks([(0, mask, (0, 0, 0))], output)
    loaded = load_file(str(output))

    assert loaded["label_map"].shape == (2, 3)


def test_export_png_writes_binary_mask(tmp_path) -> None:
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    output = tmp_path / "mask.png"

    export_masks([(0, mask, (0, 0, 0))], output)

    with Image.open(output) as image:
        values = set(np.array(image).ravel().tolist())
    assert values == {0, 255}


def test_label_map_rejects_out_of_range_object_id() -> None:
    mask = np.ones((2, 2), dtype=bool)

    try:
        build_label_map([(65535, mask, (0, 0, 0))])
    except ValueError as exc:
        assert "object ids" in str(exc)
    else:
        raise AssertionError("Expected object id range validation")


def test_export_video_masks_writes_metadata(tmp_path) -> None:
    session = AnnotationSession()
    session.reset_media("video.mp4", "video")
    session.set_mask(2, 0, np.ones((3, 4), dtype=bool))

    outputs = export_video_masks(session, tmp_path / "seq")

    assert (tmp_path / "seq" / "frame_000002.png") in outputs
    assert (tmp_path / "seq" / "metadata.json").exists()
    with Image.open(tmp_path / "seq" / "frame_000002.png") as image:
        assert set(np.array(image).ravel().tolist()) == {255}


def test_export_video_masks_uses_source_image_names(tmp_path) -> None:
    session = AnnotationSession()
    session.reset_media("frames", "image_sequence", frame_names=["img_001.jpg", "img_002.jpg"])
    session.set_mask(1, 0, np.ones((3, 4), dtype=bool))

    export_video_masks(session, tmp_path / "seq")

    metadata = json.loads((tmp_path / "seq" / "metadata.json").read_text())
    assert (tmp_path / "seq" / "img_002.png").exists()
    assert metadata["frames"][0]["mask_file"] == "img_002.png"
    assert metadata["frames"][0]["source_file"] == "img_002.jpg"


def test_export_overlay_preserves_image_size(tmp_path) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True

    output = export_overlay(image, [(0, mask, (255, 0, 0))], tmp_path / "overlay.png")

    from PIL import Image

    with Image.open(output) as exported:
        assert exported.size == (4, 3)
