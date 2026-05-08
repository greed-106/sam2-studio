import torch

from sam2_studio.engine.workers import ExportTask, VideoPropagationTask
from sam2_studio.interaction.prompts import PromptBatch


class _FakeMasks:
    data = torch.ones((1, 2, 3), dtype=torch.bool)

    def __len__(self) -> int:
        return 1


class _FakeBoxes:
    conf = torch.tensor([1.0])
    xyxy = torch.tensor([[0.0, 0.0, 2.0, 1.0]])

    def __len__(self) -> int:
        return 1


class _FakeVideoResult:
    masks = _FakeMasks()
    boxes = _FakeBoxes()


def test_video_propagation_maps_reverse_frames(monkeypatch) -> None:
    import ultralytics.models.sam as sam_module

    class FakeVideoPredictor:
        def __init__(self, overrides):
            self.overrides = overrides

        def __call__(self, **_kwargs):
            yield _FakeVideoResult()
            yield _FakeVideoResult()

    monkeypatch.setattr(sam_module, "SAM2VideoPredictor", FakeVideoPredictor)
    monkeypatch.setattr(VideoPropagationTask, "_write_clip", lambda self, frame_indices, clip_path: None)
    prompts = PromptBatch()
    prompts.add_point(1, 1, 1)
    task = VideoPropagationTask("model.pt", "video.mp4", 1, 3, 25.0, 7, prompts, "reverse", 3, 4)
    frames = []
    task.signals.frame_ready.connect(lambda payload: frames.append((payload[2].frame_idx, payload[2].object_id, payload[2].mask.shape)))

    task.run()

    assert frames == [(1, 7, (2, 3)), (0, 7, (2, 3))]


def test_export_task_emits_label_and_output() -> None:
    task = ExportTask("demo", lambda: "out.txt")
    finished = []
    task.signals.finished.connect(lambda payload: finished.append(payload))

    task.run()

    assert finished == [("demo", "out.txt")]


def test_export_task_emits_traceback_on_failure() -> None:
    def fail():
        raise RuntimeError("boom")

    task = ExportTask("demo", fail)
    failures = []
    task.signals.failed.connect(lambda detail: failures.append(detail))

    task.run()

    assert failures and "boom" in failures[0]
