from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from sam2_studio.app.settings import AppSettings
from sam2_studio.ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM2 Studio mask generation desktop app.")
    parser.add_argument("media", nargs="?", help="Optional image or video to open on startup.")
    parser.add_argument("--smoke", action="store_true", help="Import and configuration smoke check, then exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings()
    if args.smoke:
        print(f"sam2-studio smoke ok; default_model={settings.default_model_path}")
        return 0

    app = QApplication(sys.argv[:1])
    window = MainWindow(settings)
    if args.media:
        window.load_media(args.media)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
