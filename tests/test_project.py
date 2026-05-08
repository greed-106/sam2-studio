import numpy as np

from sam2_studio.interaction.prompts import BoxPrompt
from sam2_studio.session.project import ProjectModelConfig, load_project, save_project
from sam2_studio.session.session import AnnotationSession


def test_project_round_trip(tmp_path) -> None:
    media = tmp_path / "image.png"
    media.write_bytes(b"placeholder")
    session = AnnotationSession()
    session.reset_media(str(media), "image_sequence", frame_names=["image.png"])
    session.set_frame(0, np.zeros((4, 5, 3), dtype=np.uint8))
    session.add_object(1)
    prompts = session.prompt_batch(frame_idx=0, object_id=1)
    prompts.add_point(1, 2, 1)
    prompts.add_point(3, 2, 0)
    prompts.set_box(BoxPrompt(4, 3, 1, 0))
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    session.set_mask(0, 1, mask)

    project_dir = save_project(session, tmp_path / "case.sam2studio", ProjectModelConfig(path="sam2.1_l.pt"))
    loaded = load_project(project_dir)

    assert loaded.session.media_path == str(media)
    assert loaded.session.frame_names == ["image.png"]
    assert loaded.session.objects[1].color == session.objects[1].color
    assert loaded.session.prompts[0][1].points[1].label == 0
    assert loaded.session.prompts[0][1].box.x1 == 1
    assert loaded.session.masks[0][1].shape == (4, 5)
    assert loaded.session.masks[0][1].sum() == 4
