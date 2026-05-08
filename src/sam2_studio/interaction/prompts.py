from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointPrompt:
    x: float
    y: float
    label: int


@dataclass(frozen=True)
class BoxPrompt:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "BoxPrompt":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass
class PromptBatch:
    points: list[PointPrompt] = field(default_factory=list)
    box: BoxPrompt | None = None

    def add_point(self, x: float, y: float, label: int) -> None:
        self.points.append(PointPrompt(float(x), float(y), int(label)))

    def set_box(self, box: BoxPrompt) -> None:
        self.box = box.normalized()

    def clear(self) -> None:
        self.points.clear()
        self.box = None

    def clone(self) -> "PromptBatch":
        return PromptBatch(points=list(self.points), box=self.box)

    def is_empty(self) -> bool:
        return not self.points and self.box is None

    def to_ultralytics(self) -> dict[str, np.ndarray | None]:
        if self.points:
            points = np.array([[[p.x, p.y] for p in self.points]], dtype=np.float32)
            labels = np.array([[p.label for p in self.points]], dtype=np.int32)
        else:
            points = None
            labels = None

        if self.box is None:
            bboxes = None
        else:
            box = self.box.normalized()
            bboxes = np.array([[box.x1, box.y1, box.x2, box.y2]], dtype=np.float32)

        return {"points": points, "labels": labels, "bboxes": bboxes}
