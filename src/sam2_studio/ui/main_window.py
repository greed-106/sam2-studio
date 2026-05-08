from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from sam2_studio.app.settings import AppSettings
from sam2_studio.engine.backend import MaskResult
from sam2_studio.engine.ultralytics_sam2 import UltralyticsSam2Backend
from sam2_studio.engine.workers import ExportTask, LoadModelTask, SegmentTask, VideoPropagationTask
from sam2_studio.export.mask_exporter import export_masks, export_overlay, export_video_masks
from sam2_studio.interaction.prompts import BoxPrompt, PromptBatch
from sam2_studio.media.decoder import ImageSequencePlaybackThread, VideoPlaybackThread
from sam2_studio.media.source import FramePacket, ImageSequenceSource, ImageSource, VideoSource, classify_media, open_video_source
from sam2_studio.session.project import ProjectModelConfig, load_project, save_project
from sam2_studio.session.session import AnnotationSession
from sam2_studio.ui.canvas import SamCanvas
from sam2_studio.ui.playback_bar import PlaybackBar


@dataclass(frozen=True)
class PropagationSeed:
    model_path: str
    model_path_norm: str
    video_path: str
    start_frame_idx: int
    frame_count: int
    fps: float
    object_id: int
    prompts: PromptBatch
    generation: int


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or AppSettings()
        self.session = AnnotationSession()
        self.model_imgsz = self.settings.default_imgsz
        self.model_conf = self.settings.default_conf
        self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
        self.thread_pool = QThreadPool(self)
        self.video_source: VideoSource | ImageSequenceSource | None = None
        self.playback_thread: VideoPlaybackThread | ImageSequencePlaybackThread | None = None
        self.model_loaded = False
        self.model_loading = False
        self.loaded_model_path: str | None = None
        self.loading_model_path: str | None = None
        self.pending_segmentation: dict | None = None
        self.generation = 0
        self.model_load_request_id = 0
        self.segmentation_request_id = 0
        self.propagation_request_id = 0
        self.active_propagation_requests: set[int] = set()
        self.propagation_tasks: dict[int, VideoPropagationTask] = {}
        self.propagation_model_paths: dict[int, str] = {}
        self.pending_propagation_specs: list[tuple[str, PropagationSeed]] = []
        self.closed = False
        self.frame_count = 1

        self.setWindowTitle("SAM2 Studio")
        self.resize(1280, 820)
        self.canvas = SamCanvas()
        self.playback_bar = PlaybackBar()
        self.model_path = QLineEdit(str(self.settings.default_model_path))
        self.object_combo = QComboBox()
        self.status = QLabel("Ready")

        self._build_ui()
        self._connect_signals()
        self._refresh_objects()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        open_button = QPushButton("打开媒体")
        open_sequence_button = QPushButton("打开图片目录")
        model_button = QPushButton("加载模型")
        export_button = QPushButton("导出当前二值掩码")
        export_sequence_button = QPushButton("导出全部二值PNG")
        export_overlay_button = QPushButton("导出当前叠加图")
        save_project_button = QPushButton("保存项目")
        load_project_button = QPushButton("加载项目")
        propagate_button = QPushButton("正向传播▶")
        propagate_reverse_button = QPushButton("反向传播◀")
        propagate_both_button = QPushButton("先正后反传播")
        cancel_propagation_button = QPushButton("取消传播")
        positive_button = QPushButton("正点")
        negative_button = QPushButton("负点")
        box_button = QPushButton("框选")
        polygon_button = QPushButton("多边形")
        brush_add_button = QPushButton("笔刷+")
        brush_erase_button = QPushButton("笔刷-")
        undo_button = QPushButton("撤销")
        add_object_button = QPushButton("新增对象")
        choose_model_button = QPushButton("选择模型")

        open_button.clicked.connect(self.open_media)
        open_sequence_button.clicked.connect(self.open_image_sequence)
        model_button.clicked.connect(self.load_model)
        export_button.clicked.connect(self.export_current_masks)
        export_sequence_button.clicked.connect(self.export_mask_sequence)
        export_overlay_button.clicked.connect(self.export_current_overlay)
        save_project_button.clicked.connect(self.save_project_dialog)
        load_project_button.clicked.connect(self.load_project_dialog)
        propagate_button.clicked.connect(lambda: self.propagate_video("forward"))
        propagate_reverse_button.clicked.connect(lambda: self.propagate_video("reverse"))
        propagate_both_button.clicked.connect(lambda: self.propagate_video("both"))
        cancel_propagation_button.clicked.connect(self.cancel_propagation)
        positive_button.clicked.connect(lambda: self.canvas.set_tool("positive"))
        negative_button.clicked.connect(lambda: self.canvas.set_tool("negative"))
        box_button.clicked.connect(lambda: self.canvas.set_tool("box"))
        polygon_button.clicked.connect(lambda: self.canvas.set_tool("polygon"))
        brush_add_button.clicked.connect(lambda: self.canvas.set_tool("brush_add"))
        brush_erase_button.clicked.connect(lambda: self.canvas.set_tool("brush_erase"))
        undo_button.clicked.connect(self.undo_prompt)
        add_object_button.clicked.connect(self.add_object)
        choose_model_button.clicked.connect(self.choose_model)
        self.object_combo.currentIndexChanged.connect(self._on_object_changed)
        self.model_path.textEdited.connect(self._on_model_path_edited)

        for widget in [
            open_button,
            open_sequence_button,
            model_button,
            positive_button,
            negative_button,
            box_button,
            polygon_button,
            brush_add_button,
            brush_erase_button,
            undo_button,
            add_object_button,
            propagate_button,
            propagate_reverse_button,
            propagate_both_button,
            cancel_propagation_button,
            export_button,
            export_sequence_button,
            export_overlay_button,
            save_project_button,
            load_project_button,
        ]:
            toolbar.addWidget(widget)
        toolbar.addWidget(QLabel(" 对象 "))
        toolbar.addWidget(self.object_combo)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(choose_model_button)

        root = QVBoxLayout()
        root.addLayout(model_row)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.playback_bar)
        root.addWidget(self.status)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _connect_signals(self) -> None:
        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.polygon_finished.connect(self._on_polygon_finished)
        self.canvas.brush_stroke.connect(self._on_brush_stroke)
        self.playback_bar.play_toggled.connect(self._on_play_toggled)
        self.playback_bar.seek_requested.connect(self.seek_video)
        self.playback_bar.step_requested.connect(self.step_frame)

    def closeEvent(self, event) -> None:
        self._cancel_propagation_tasks(update_status=False)
        self.thread_pool.clear()
        if not self.thread_pool.waitForDone(3000):
            self.status.setText("Waiting for background tasks to stop before closing...")
            event.ignore()
            return
        self.closed = True
        self._stop_playback_thread()
        if self.video_source is not None:
            self.video_source.close()
        super().closeEvent(event)

    def open_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open media",
            str(self.settings.project_root),
            "Media (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.webm)",
        )
        if path:
            self.load_media(path)

    def open_image_sequence(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "打开图片序列目录", str(self.settings.project_root))
        if directory:
            self.load_media(directory)

    def load_media(self, path: str) -> None:
        self.generation += 1
        self.pending_segmentation = None
        self._cancel_propagation_tasks(update_status=False)
        self._stop_playback_thread()
        if self.video_source is not None:
            self.video_source.close()
            self.video_source = None

        media_kind = classify_media(path)
        if media_kind == "image":
            self.session.reset_media(path, media_kind)
            image_rgb = ImageSource(path).read()
            self.frame_count = 1
            self.session.set_frame(0, image_rgb)
            self.playback_bar.configure(1)
            self._refresh_canvas()
            self.status.setText(f"Loaded image {Path(path).name} ({image_rgb.shape[1]}x{image_rgb.shape[0]})")
            return

        self.video_source = open_video_source(path)
        self.session.reset_media(path, media_kind, frame_names=getattr(self.video_source, "frame_names", []))
        first = self.video_source.read_frame(0)
        self.frame_count = self.video_source.frame_count
        self.session.set_frame(first.index, first.image_rgb)
        self.playback_bar.configure(self.frame_count)
        self._refresh_canvas()
        self._start_playback_thread(path, self.video_source.fps)
        self.status.setText(
            f"Loaded {media_kind} {Path(path).name} ({self.video_source.width}x{self.video_source.height}, "
            f"{self.frame_count} frames @ {self.video_source.fps:.2f} FPS)"
        )

    def load_model(self) -> None:
        model_path = self.model_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Model", "Please choose a SAM2 model path.")
            return
        normalized_model_path = self._normalized_model_path(model_path)
        if self.model_loading:
            if self.loading_model_path == normalized_model_path:
                return
            self.model_load_request_id += 1
            self.model_loading = False
            self.loading_model_path = None
            self.segmentation_request_id += 1
            self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
            self._cancel_propagation_tasks(update_status=False)
            self.pending_segmentation = None
        if self.model_loaded and self.loaded_model_path == normalized_model_path:
            return
        if self.model_loaded and self.loaded_model_path != normalized_model_path:
            self.model_loaded = False
            self.loaded_model_path = None
            self.segmentation_request_id += 1
            self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
            self.model_load_request_id += 1
            self._cancel_propagation_tasks(update_status=False)
            self.pending_segmentation = None
        self.status.setText("Loading SAM2 model...")
        self.model_loading = True
        self.model_load_request_id += 1
        request_id = self.model_load_request_id
        backend = self.backend
        self.loading_model_path = normalized_model_path
        task = LoadModelTask(backend, model_path)
        task.signals.finished.connect(lambda loaded_path, r=request_id, b=backend: self._on_model_loaded(loaded_path, r, b))
        task.signals.failed.connect(lambda detail, r=request_id, b=backend: self._on_model_failed(detail, r, b))
        self.thread_pool.start(task)

    def _on_model_loaded(self, model_path: str, request_id: int, backend: UltralyticsSam2Backend) -> None:
        if self.closed or request_id != self.model_load_request_id or backend is not self.backend:
            return
        self.model_loading = False
        self.model_loaded = True
        self.loaded_model_path = self._normalized_model_path(model_path)
        self.loading_model_path = None
        self.status.setText(f"Loaded model {Path(model_path).name}")
        if self.pending_segmentation is not None:
            pending = self.pending_segmentation
            self.pending_segmentation = None
            if pending["generation"] == self.generation and pending["model_path"] == self.loaded_model_path:
                self._start_segment_task(pending)

    def _on_model_failed(self, detail: str, request_id: int, backend: UltralyticsSam2Backend) -> None:
        if self.closed or request_id != self.model_load_request_id or backend is not self.backend:
            return
        self.loading_model_path = None
        self._on_worker_failed(detail)

    def _start_playback_thread(self, path: str, fps: float) -> None:
        if Path(path).is_dir():
            self.playback_thread = ImageSequencePlaybackThread(path, fps, self)
        else:
            self.playback_thread = VideoPlaybackThread(path, fps, self)
        generation = self.generation
        thread = self.playback_thread
        self.playback_thread.frame_ready.connect(lambda packet: self._on_video_frame(packet, generation, thread))
        self.playback_thread.failed.connect(self._on_playback_failed)
        self.playback_thread.playback_finished.connect(lambda: self.playback_bar.set_playing(False))
        self.playback_thread.start()

    def _stop_playback_thread(self) -> None:
        if self.playback_thread is not None:
            thread = self.playback_thread
            self.playback_thread = None
            thread.stop()
            if not thread.wait(3000):
                thread.terminate()
                thread.wait(1000)
        self.playback_bar.set_playing(False)

    def _on_play_toggled(self, playing: bool) -> None:
        if self.playback_thread is None:
            self.playback_bar.set_playing(False)
            return
        if playing:
            self.playback_thread.play()
        else:
            self.playback_thread.pause()

    def seek_video(self, frame_idx: int) -> None:
        if self.video_source is None:
            return
        was_playing = self.playback_bar.is_playing()
        if self.playback_thread is not None:
            self.playback_thread.request_seek(frame_idx, play_after_seek=was_playing)
        else:
            packet = self.video_source.read_frame(frame_idx)
            self._on_video_frame(packet, self.generation)

    def step_frame(self, delta: int) -> None:
        if self.video_source is None or self.frame_count <= 1:
            return
        target = max(0, min(self.session.current_frame_idx + int(delta), self.frame_count - 1))
        if target == self.session.current_frame_idx:
            return
        if self.playback_thread is not None:
            self.playback_thread.pause()
        self.playback_bar.set_playing(False)
        self.seek_video(target)

    def _on_video_frame(self, packet: FramePacket, generation: int, thread: VideoPlaybackThread | None = None) -> None:
        try:
            if generation != self.generation:
                return
            self.session.set_frame(packet.index, packet.image_rgb)
            self.playback_bar.set_position(packet.index, self.frame_count)
            self._refresh_canvas()
        finally:
            if thread is not None:
                thread.ack_frame()

    def _on_playback_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Playback", message)
        self.playback_bar.set_playing(False)

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        if self.session.current_image_rgb is None:
            return
        if self.playback_thread is not None:
            self.playback_thread.pause()
            self.playback_bar.set_playing(False)
        prompts = self.session.prompt_batch()
        prompts.add_point(x, y, label)
        self._refresh_canvas()
        self._run_segmentation()

    def _on_box_added(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.session.current_image_rgb is None:
            return
        if self.playback_thread is not None:
            self.playback_thread.pause()
            self.playback_bar.set_playing(False)
        prompts = self.session.prompt_batch()
        prompts.set_box(BoxPrompt(x1, y1, x2, y2))
        self._refresh_canvas()
        self._run_segmentation()

    def _on_polygon_finished(self, points: object) -> None:
        if self.session.current_image_rgb is None:
            return
        polygon = list(points)
        if len(polygon) < 3:
            return
        self._begin_manual_mask_edit()
        height, width = self.session.current_image_rgb.shape[:2]
        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        raster_points = np.round(pts).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [raster_points], 1)
        self._replace_current_mask(mask.astype(bool))
        self.status.setText("Polygon converted to current object mask")
        self._refresh_canvas()

    def _on_brush_stroke(self, points: object, radius: float, mode: int) -> None:
        if self.session.current_image_rgb is None:
            return
        stroke_points = list(points)
        if not stroke_points:
            return
        self._begin_manual_mask_edit()
        height, width = self.session.current_image_rgb.shape[:2]
        frame_idx = self.session.current_frame_idx
        object_id = self.session.current_object_id
        existing = self.session.masks.get(frame_idx, {}).get(object_id)
        base = existing.copy() if existing is not None else np.zeros((height, width), dtype=bool)
        stroke = np.zeros((height, width), dtype=np.uint8)
        pixel_points = []
        for x, y in stroke_points:
            px = int(np.clip(round(float(x)), 0, width - 1))
            py = int(np.clip(round(float(y)), 0, height - 1))
            pixel_points.append((px, py))
        thickness = max(1, int(round(radius * 2)))
        if len(pixel_points) == 1:
            cv2.circle(stroke, pixel_points[0], max(1, int(round(radius))), 1, thickness=-1)
        else:
            for start, end in zip(pixel_points, pixel_points[1:]):
                cv2.line(stroke, start, end, 1, thickness=thickness, lineType=cv2.LINE_AA)
        if mode:
            updated = base | stroke.astype(bool)
        else:
            updated = base & ~stroke.astype(bool)
        self._replace_current_mask(updated)
        self.status.setText("Brush updated current object mask")
        self._refresh_canvas()

    def _begin_manual_mask_edit(self) -> None:
        if self.playback_thread is not None:
            self.playback_thread.pause()
            self.playback_bar.set_playing(False)
        self.pending_segmentation = None
        self.segmentation_request_id += 1
        self._cancel_propagation_tasks(update_status=False)

    def _replace_current_mask(self, mask: np.ndarray) -> None:
        frame_idx = self.session.current_frame_idx
        object_id = self.session.current_object_id
        previous = self.session.masks.get(frame_idx, {}).get(object_id)
        self.session.push_mask_undo(frame_idx, object_id, None if previous is None else previous.copy())
        self.session.set_mask(frame_idx, object_id, mask)

    def _run_segmentation(self) -> None:
        model_path = self.model_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Model", "Please choose a SAM2 model path.")
            return
        image_rgb = self.session.current_image_rgb
        if image_rgb is None:
            return
        object_id = self.session.current_object_id
        prompts = self.session.prompt_batch(object_id=object_id).clone()
        image_key = f"{self.session.media_path}:{self.session.current_frame_idx}"
        frame_idx = self.session.current_frame_idx
        if self.model_loaded and self.loaded_model_path != self._normalized_model_path(model_path):
            self.model_loaded = False
            self.loaded_model_path = None
            self.segmentation_request_id += 1
            self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
            self.model_load_request_id += 1
        if not self.model_loaded:
            self.segmentation_request_id += 1
            self.pending_segmentation = {
                "generation": self.generation,
                "request_id": self.segmentation_request_id,
                "model_path": self._normalized_model_path(model_path),
                "image_rgb": image_rgb.copy(),
                "image_key": image_key,
                "frame_idx": frame_idx,
                "object_id": object_id,
                "prompts": prompts,
            }
            self.load_model()
            self.status.setText("Queued segmentation until SAM2 model finishes loading...")
            return
        self.segmentation_request_id += 1
        payload = {
            "generation": self.generation,
            "request_id": self.segmentation_request_id,
            "image_rgb": image_rgb.copy(),
            "image_key": image_key,
            "frame_idx": frame_idx,
            "object_id": object_id,
            "prompts": prompts,
        }
        self._start_segment_task(payload)

    def _start_segment_task(self, payload: dict) -> None:
        self.status.setText(f"Segmenting frame {payload['frame_idx'] + 1}, object {payload['object_id']}...")
        task = SegmentTask(
            self.backend,
            payload["image_rgb"],
            payload["image_key"],
            payload["generation"],
            payload["request_id"],
            payload["frame_idx"],
            payload["object_id"],
            payload["prompts"],
        )
        task.signals.finished.connect(self._on_segmentation_finished)
        task.signals.failed.connect(
            lambda detail, g=payload["generation"], r=payload["request_id"], k=payload["image_key"], f=payload["frame_idx"]: self._on_segmentation_failed(
                g, r, k, f, detail
            )
        )
        self.thread_pool.start(task)

    def _on_segmentation_finished(self, payload: tuple[int, int, str, int, MaskResult]) -> None:
        generation, request_id, image_key, frame_idx, result = payload
        media_key = f"{self.session.media_path}:{frame_idx}"
        if generation != self.generation or request_id != self.segmentation_request_id or image_key != media_key:
            return
        self.session.set_mask(frame_idx, result.object_id, result.masks[0])
        if frame_idx != self.session.current_frame_idx:
            return
        self.status.setText(f"Updated mask for frame {frame_idx + 1}, object {result.object_id}")
        self._refresh_canvas()

    def _on_segmentation_failed(self, generation: int, request_id: int, image_key: str, frame_idx: int, detail: str) -> None:
        media_key = f"{self.session.media_path}:{frame_idx}"
        if (
            generation != self.generation
            or request_id != self.segmentation_request_id
            or image_key != media_key
            or frame_idx != self.session.current_frame_idx
        ):
            return
        self._on_worker_failed(detail)

    def _on_worker_failed(self, detail: str) -> None:
        if self.closed:
            return
        self.model_loading = False
        self.status.setText("Operation failed")
        QMessageBox.critical(self, "SAM2 Studio", detail)

    def propagate_video(self, direction: str) -> None:
        if self.session.media_kind not in {"video", "image_sequence"} or self.session.media_path is None or self.video_source is None:
            QMessageBox.information(self, "Propagation", "请先打开视频或图片序列。")
            return
        if not self.model_path.text().strip():
            QMessageBox.information(self, "Propagation", "Choose a SAM2 model before propagating masks.")
            return
        prompts = self.session.prompt_batch()
        if prompts.is_empty():
            QMessageBox.information(self, "Propagation", "Add a point or box prompt on the current frame first.")
            return
        if self.playback_thread is not None:
            self.playback_thread.pause()
            self.playback_bar.set_playing(False)

        directions = ["forward", "reverse"] if direction == "both" else [direction]
        self._cancel_propagation_tasks(update_status=False)
        seed = self._build_propagation_seed()
        self.pending_propagation_specs = [(item, seed) for item in directions[1:]]
        self._start_propagation(directions[0], seed)

    def _build_propagation_seed(self) -> PropagationSeed:
        model_path = self.model_path.text().strip()
        return PropagationSeed(
            model_path=model_path,
            model_path_norm=self._normalized_model_path(model_path),
            video_path=self.session.media_path or "",
            start_frame_idx=self.session.current_frame_idx,
            frame_count=self.frame_count,
            fps=self.video_source.fps if self.video_source else 25.0,
            object_id=self.session.current_object_id,
            prompts=self.session.prompt_batch().clone(),
            generation=self.generation,
        )

    def _start_propagation(self, direction: str, seed: PropagationSeed) -> None:
        self.propagation_request_id += 1
        request_id = self.propagation_request_id
        self.active_propagation_requests.add(request_id)
        self.propagation_model_paths[request_id] = seed.model_path_norm
        task = VideoPropagationTask(
            model_path=seed.model_path,
            video_path=seed.video_path,
            start_frame_idx=seed.start_frame_idx,
            frame_count=seed.frame_count,
            fps=seed.fps,
            object_id=seed.object_id,
            prompts=seed.prompts,
            direction=direction,
            generation=seed.generation,
            request_id=request_id,
            imgsz=self.model_imgsz,
            conf=self.model_conf,
        )
        task.signals.frame_ready.connect(self._on_propagation_frame)
        task.signals.progress.connect(
            lambda done, total, d=direction, g=self.generation, r=request_id: self._on_propagation_progress(g, r, d, done, total)
        )
        task.signals.finished.connect(self._on_propagation_finished)
        task.signals.failed.connect(lambda detail, g=self.generation, r=request_id: self._on_propagation_failed(g, r, detail))
        self.propagation_tasks[request_id] = task
        self.status.setText(f"Starting {direction} propagation from frame {self.session.current_frame_idx + 1}...")
        self.thread_pool.start(task)

    def cancel_propagation(self) -> None:
        self._cancel_propagation_tasks(update_status=True)

    def _cancel_propagation_tasks(self, update_status: bool) -> None:
        for task in self.propagation_tasks.values():
            task.cancel()
        self.active_propagation_requests.clear()
        self.propagation_tasks.clear()
        self.propagation_model_paths.clear()
        self.pending_propagation_specs.clear()
        if update_status:
            self.status.setText("Propagation cancellation requested")

    def _on_propagation_progress(self, generation: int, request_id: int, direction: str, done: int, total: int) -> None:
        if not self._is_active_propagation(generation, request_id):
            return
        self.status.setText(f"Propagating {direction}: {done}/{total}")

    def _on_propagation_frame(self, payload: object) -> None:
        generation, request_id, result = payload
        if not self._is_active_propagation(generation, request_id):
            return
        self.session.set_mask(result.frame_idx, result.object_id, result.mask)
        if result.frame_idx == self.session.current_frame_idx:
            self._refresh_canvas()

    def _on_propagation_finished(self, payload: object) -> None:
        generation, _request_id, direction = payload
        if not self._is_active_propagation(generation, _request_id):
            return
        self.active_propagation_requests.discard(_request_id)
        self.propagation_tasks.pop(_request_id, None)
        self.propagation_model_paths.pop(_request_id, None)
        self.status.setText(f"Finished {direction} propagation")
        if self.pending_propagation_specs:
            next_direction, seed = self.pending_propagation_specs.pop(0)
            self._start_propagation(next_direction, seed)

    def _on_propagation_failed(self, generation: int, request_id: int, detail: str) -> None:
        if not self._is_active_propagation(generation, request_id):
            return
        self.active_propagation_requests.discard(request_id)
        self.propagation_tasks.pop(request_id, None)
        self.propagation_model_paths.pop(request_id, None)
        if "Propagation cancelled." in detail:
            self.status.setText("Propagation cancelled")
            return
        self._on_worker_failed(detail)

    def add_object(self) -> None:
        self.session.add_object()
        self._refresh_objects()

    def undo_prompt(self) -> None:
        if self.session.undo_mask_edit():
            self.segmentation_request_id += 1
            self.pending_segmentation = None
            self._cancel_propagation_tasks(update_status=False)
            self.status.setText("Undid last mask edit")
            self._refresh_canvas()
        elif self.session.undo_last_prompt():
            self.segmentation_request_id += 1
            self.status.setText("Undid last prompt and cleared the current mask")
            self._refresh_canvas()
        else:
            self.status.setText("No prompt to undo on this frame/object")

    def _refresh_objects(self) -> None:
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        for object_id, state in sorted(self.session.objects.items()):
            self.object_combo.addItem(state.name, object_id)
            if object_id == self.session.current_object_id:
                self.object_combo.setCurrentIndex(self.object_combo.count() - 1)
        self.object_combo.blockSignals(False)

    def _on_object_changed(self, index: int) -> None:
        object_id = self.object_combo.itemData(index)
        if object_id is not None:
            self.session.current_object_id = int(object_id)

    def _refresh_canvas(self) -> None:
        if self.session.current_image_rgb is None:
            return
        self.canvas.set_image(self.session.current_image_rgb)
        self.canvas.set_masks(self.session.current_masks())
        self.canvas.set_prompts(self.session.current_prompts())
        self.playback_bar.set_markers(set(self.session.prompts), set(self.session.masks))

    def export_current_masks(self) -> None:
        mask_items = self.session.current_masks()
        if not mask_items:
            QMessageBox.information(self, "Export", "No masks are available on the current frame.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前帧二值掩码（只包含当前帧可见对象）",
            str(self.settings.project_root / "mask.png"),
            "Mask PNG (*.png);;NumPy archive (*.npz);;SafeTensors (*.safetensors)",
        )
        if not path:
            return
        try:
            snapshot = [(object_id, mask.copy(), color) for object_id, mask, color in mask_items]
            self._start_export("当前帧二值掩码", lambda: export_masks(snapshot, path))
        except Exception as exc:
            QMessageBox.critical(self, "Export", str(exc))

    def export_current_overlay(self) -> None:
        mask_items = self.session.current_masks()
        if not mask_items or self.session.current_image_rgb is None:
            QMessageBox.information(self, "Export", "No image and masks are available for overlay export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前帧叠加预览图",
            str(self.settings.project_root / "overlay.png"),
            "Overlay PNG (*.png)",
        )
        if not path:
            return
        try:
            image_rgb = self.session.current_image_rgb.copy()
            snapshot = [(object_id, mask.copy(), color) for object_id, mask, color in mask_items]
            self._start_export("当前帧叠加图", lambda: export_overlay(image_rgb, snapshot, path))
        except Exception as exc:
            QMessageBox.critical(self, "Export", str(exc))

    def export_mask_sequence(self) -> None:
        if not self.session.masks:
            QMessageBox.information(self, "Export", "No masks are available for sequence export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "导出全部已有掩码帧的二值 PNG 序列", str(self.settings.project_root))
        if not directory:
            return
        try:
            snapshot = copy.deepcopy(self.session)
            self._start_export("全部帧二值 PNG 序列", lambda: export_video_masks(snapshot, directory))
        except Exception as exc:
            QMessageBox.critical(self, "Export", str(exc))

    def _start_export(self, label: str, export_func) -> None:
        self.status.setText(f"正在导出 {label}...")
        task = ExportTask(label, export_func)
        task.signals.finished.connect(self._on_export_finished)
        task.signals.failed.connect(self._on_worker_failed)
        self.thread_pool.start(task)

    def _on_export_finished(self, payload: object) -> None:
        label, output = payload
        if isinstance(output, list):
            self.status.setText(f"已导出 {label}: {len(output)} 个文件")
        else:
            self.status.setText(f"已导出 {label}: {output}")

    def save_project_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 SAM2 Studio 项目",
            str(self.settings.project_root / "annotation.sam2studio"),
            "SAM2 Studio Project (*.sam2studio)",
        )
        if not path:
            return
        try:
            saved = save_project(
                self.session,
                path,
                self._project_model_config(),
                frame_count=self.frame_count,
                fps=self.video_source.fps if self.video_source else None,
            )
            self.status.setText(f"Saved project {saved}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", str(exc))

    def load_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "加载 SAM2 Studio 项目", str(self.settings.project_root))
        if not directory:
            return
        try:
            project = load_project(directory)
            media_path = project.session.media_path
            if media_path is None:
                raise ValueError("Project does not contain a media path.")
            self.load_media(media_path)
            loaded_session = project.session
            target_frame = loaded_session.current_frame_idx
            if self.video_source is not None:
                packet = self.video_source.read_frame(target_frame)
                current_image = packet.image_rgb
                target_frame = packet.index
                if self.playback_thread is not None:
                    self.playback_thread.request_seek(target_frame, play_after_seek=False)
            else:
                current_image = self.session.current_image_rgb
            self.session.objects = loaded_session.objects
            self.session.prompts = loaded_session.prompts
            self.session.masks = loaded_session.masks
            self.session.current_object_id = loaded_session.current_object_id
            self.session.current_frame_idx = target_frame
            self.session.current_image_rgb = current_image
            self.model_path.setText(project.model.path)
            self.model_imgsz = project.model.imgsz
            self.model_conf = project.model.conf
            self.backend = UltralyticsSam2Backend(imgsz=project.model.imgsz, conf=project.model.conf)
            self.model_load_request_id += 1
            self.model_loaded = False
            self.model_loading = False
            self.loaded_model_path = None
            self.loading_model_path = None
            self.pending_segmentation = None
            self._refresh_objects()
            self.playback_bar.set_position(self.session.current_frame_idx, self.frame_count)
            self._refresh_canvas()
            self.status.setText(f"Loaded project {directory}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Project", str(exc))

    def _project_model_config(self) -> ProjectModelConfig:
        return ProjectModelConfig(
            path=self.model_path.text().strip(),
            imgsz=self.model_imgsz,
            conf=self.model_conf,
        )

    def _normalized_model_path(self, model_path: str) -> str:
        return str(Path(model_path).expanduser().resolve())

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose SAM2 model",
            str(self.settings.project_root),
            "Model weights (*.pt *.pth);;All files (*)",
        )
        if path:
            self.model_path.setText(path)
            self._on_model_path_edited(path)

    def _is_active_propagation(self, generation: int, request_id: int) -> bool:
        if generation != self.generation or request_id not in self.active_propagation_requests:
            return False
        current_model_path = self.model_path.text().strip()
        if not current_model_path:
            return False
        return self.propagation_model_paths.get(request_id) == self._normalized_model_path(current_model_path)

    def _on_model_path_edited(self, text: str) -> None:
        if not text.strip():
            return
        normalized = self._normalized_model_path(text.strip())
        if self.model_loaded and self.loaded_model_path != normalized:
            self.model_loaded = False
            self.loaded_model_path = None
            self.segmentation_request_id += 1
            self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
            self.model_load_request_id += 1
            self._cancel_propagation_tasks(update_status=False)
            self.pending_segmentation = None
        if self.model_loading and self.loading_model_path != normalized:
            self.model_loading = False
            self.loading_model_path = None
            self.segmentation_request_id += 1
            self.backend = UltralyticsSam2Backend(imgsz=self.model_imgsz, conf=self.model_conf)
            self.model_load_request_id += 1
            self._cancel_propagation_tasks(update_status=False)
            self.pending_segmentation = None
