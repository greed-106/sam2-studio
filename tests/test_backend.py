import numpy as np
import torch

from sam2_studio.engine.ultralytics_sam2 import UltralyticsSam2Backend
from sam2_studio.interaction.prompts import PromptBatch


class _FakeMasks:
    data = torch.tensor([[[True, False], [False, True]]])


class _FakeBoxes:
    conf = torch.tensor([0.9])
    xyxy = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    def __len__(self) -> int:
        return 1


class _FakeResult:
    masks = _FakeMasks()
    boxes = _FakeBoxes()


def test_backend_caches_set_image_and_converts_rgb_to_bgr(monkeypatch) -> None:
    import ultralytics.models.sam as sam_module

    class FakePredictor:
        instances = []

        def __init__(self, overrides):
            self.overrides = overrides
            self.set_images = []
            FakePredictor.instances.append(self)

        def reset_image(self):
            pass

        def set_image(self, image_bgr):
            self.set_images.append(image_bgr.copy())

        def __call__(self, **_kwargs):
            return [_FakeResult()]

    monkeypatch.setattr(sam_module, "SAM2Predictor", FakePredictor)
    backend = UltralyticsSam2Backend()
    backend.load_model("model.pt")
    prompts = PromptBatch()
    prompts.add_point(1, 1, 1)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]

    backend.segment_image(image, "same", 0, prompts)
    backend.segment_image(image, "same", 0, prompts)

    predictor = FakePredictor.instances[0]
    assert len(predictor.set_images) == 1
    assert predictor.set_images[0][0, 0].tolist() == [3, 2, 1]
