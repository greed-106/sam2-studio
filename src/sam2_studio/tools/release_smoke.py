from __future__ import annotations

import argparse
from importlib.metadata import version

from sam2_studio.app.settings import AppSettings
from sam2_studio.session.session import AnnotationSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run release smoke checks that do not require a GPU model.")
    parser.add_argument("--expect-model", action="store_true", help="Fail if the default model file is missing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings()
    model_path = settings.default_model_path
    if args.expect_model and not model_path.exists():
        raise SystemExit(f"Default model is missing: {model_path}")
    session = AnnotationSession()
    print(
        "sam2-studio release smoke ok; "
        f"version={version('sam2-studio')} default_model={model_path} objects={len(session.objects)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
