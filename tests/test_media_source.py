from pathlib import Path
import numpy as np

from PIL import Image

from sam2_studio.media.source import ImageSequenceSource, VideoSource, classify_media


def test_classify_media() -> None:
    assert classify_media("example.png") == "image"
    assert classify_media("example.mp4") == "video"
    try:
        classify_media("example.tif")
    except ValueError:
        pass
    else:
        raise AssertionError("TIF should not be supported")


def test_video_source_reads_sample_video() -> None:
    video_path = Path(__file__).resolve().parents[1] / "test-video.mp4"
    source = VideoSource(video_path)
    try:
        packet = source.read_frame(0)
    finally:
        source.close()

    assert packet.index == 0
    assert packet.image_rgb.shape == (1080, 1920, 3)


def test_image_sequence_source_uses_natural_sort(tmp_path) -> None:
    for name, value in [("10.png", 10), ("2.png", 2), ("1.png", 1)]:
        image = np.full((4, 5, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / name)

    assert classify_media(tmp_path) == "image_sequence"
    source = ImageSequenceSource(tmp_path)
    packet = source.read_frame(1)

    assert source.frame_names == ["1.png", "2.png", "10.png"]
    assert source.frame_count == 3
    assert packet.image_rgb.shape == (4, 5, 3)
    assert int(packet.image_rgb[0, 0, 0]) == 2
