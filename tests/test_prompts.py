import numpy as np

from sam2_studio.interaction.prompts import BoxPrompt, PromptBatch
from sam2_studio.session.session import AnnotationSession


def test_prompt_batch_shapes_for_single_object() -> None:
    prompts = PromptBatch()
    prompts.add_point(10, 20, 1)
    prompts.add_point(30, 40, 0)
    prompts.set_box(BoxPrompt(50, 60, 5, 6))

    data = prompts.to_ultralytics()

    assert data["points"].shape == (1, 2, 2)
    assert data["labels"].tolist() == [[1, 0]]
    assert data["bboxes"].tolist() == [[5.0, 6.0, 50.0, 60.0]]


def test_undo_last_prompt_clears_current_mask() -> None:
    session = AnnotationSession()
    prompts = session.prompt_batch()
    prompts.add_point(1, 2, 1)
    session.set_mask(0, 0, np.ones((2, 2), dtype=bool))

    assert session.undo_last_prompt()
    assert 0 not in session.masks
    assert session.prompts[0] == {}


def test_mask_edit_undo_restores_previous_mask() -> None:
    session = AnnotationSession()
    old = np.zeros((2, 2), dtype=bool)
    old[0, 0] = True
    new = np.ones((2, 2), dtype=bool)
    session.set_mask(0, 0, old)
    session.push_mask_undo(0, 0, old)
    session.set_mask(0, 0, new)

    assert session.undo_mask_edit()
    assert session.masks[0][0].sum() == 1
