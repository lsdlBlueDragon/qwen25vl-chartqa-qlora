import importlib.util
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("chartqa_app", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_modes_resolve_base_and_adapter(monkeypatch):
    app = load_app_module()
    monkeypatch.setenv("CHARTQA_ADAPTER_PATH", "org/hardmix-adapter")

    base_config = app.inference_config_for_mode(app.BASE_MODE)
    adapter_config = app.inference_config_for_mode(app.ADAPTER_MODE)

    assert base_config.adapter_path is None
    assert adapter_config.adapter_path == "org/hardmix-adapter"
    assert base_config.model_id == adapter_config.model_id


def test_gradio_demo_builds_without_loading_model():
    app = load_app_module()

    demo = app.build_demo()

    assert demo is not None


def test_builtin_examples_use_repository_owned_chart():
    app = load_app_module()

    assert app.EXAMPLE_IMAGE.exists()
    assert app.EXAMPLE_IMAGE.suffix == ".svg"
    assert len(app.EXAMPLES) == 3
    assert all(example[0] == str(app.EXAMPLE_IMAGE) for example in app.EXAMPLES)
