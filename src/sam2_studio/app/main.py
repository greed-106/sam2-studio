from __future__ import annotations

import argparse
import importlib
import sys

from PySide6.QtWidgets import QApplication

from sam2_studio.app.settings import AppSettings


BACKEND_INSTALL_HELP = """SAM2 Studio 缺少推理后端依赖。请先根据你的 CPU/CUDA 环境单独安装 PyTorch 和 Ultralytics，例如：

  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  uv pip install "ultralytics>=8.4.47,<8.5"

安装后仍然可以用 uv run 启动：

  uv run sam2-studio

更多说明见 docs/installation.md。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM2 Studio mask generation desktop app.")
    parser.add_argument("media", nargs="?", help="Optional image or video to open on startup.")
    parser.add_argument("--smoke", action="store_true", help="Import and configuration smoke check, then exit.")
    return parser


def check_inference_backend() -> tuple[bool, str]:
    failures: list[str] = []
    for module_name in ("torch", "ultralytics"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")
    if failures:
        return False, BACKEND_INSTALL_HELP + "\n\n当前错误：\n" + "\n".join(f"- {item}" for item in failures)
    return True, ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings()
    if args.smoke:
        print(f"sam2-studio smoke ok; default_model={settings.default_model_path}")
        return 0

    ok, message = check_inference_backend()
    if not ok:
        print(message, file=sys.stderr)
        return 1

    from sam2_studio.ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow(settings)
    if args.media:
        window.load_media(args.media)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
