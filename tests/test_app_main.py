import importlib

from sam2_studio.app.main import check_inference_backend


def test_check_inference_backend_reports_missing_module(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "torch":
            raise ImportError("missing torch")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import)

    ok, message = check_inference_backend()

    assert not ok
    assert "torch" in message
    assert "uv pip install" in message


def test_check_inference_backend_accepts_installed_modules(monkeypatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda _name: object())

    ok, message = check_inference_backend()

    assert ok
    assert message == ""
