from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from sam2_studio.engine.ultralytics_sam2 import UltralyticsSam2Backend
from sam2_studio.export.mask_exporter import export_masks
from sam2_studio.interaction.prompts import PromptBatch
from sam2_studio.media.source import ImageSource, open_video_source


def parse_point(value: str) -> tuple[float, float]:
    x, y = value.split(",", 1)
    return float(x), float(y)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single SAM2 point segmentation smoke test.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--point", type=parse_point, required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.image) == bool(args.video):
        raise SystemExit("Pass exactly one of --image or --video")

    if args.image:
        image_rgb = ImageSource(args.image).read()
        image_key = str(Path(args.image).resolve())
    else:
        source = open_video_source(args.video)
        try:
            packet = source.read_frame(args.frame)
            image_rgb = packet.image_rgb
            image_key = f"{Path(args.video).resolve()}:{packet.index}"
        finally:
            source.close()

    prompts = PromptBatch()
    prompts.add_point(args.point[0], args.point[1], 1)
    backend = UltralyticsSam2Backend()
    backend.load_model(args.model)
    result = backend.segment_image(image_rgb, image_key, object_id=0, prompts=prompts)
    export_masks([(0, result.masks[0], (0, 204, 102))], args.out)
    with Image.open(args.out) as exported:
        print(f"wrote {args.out} size={exported.size} masks={result.masks.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
