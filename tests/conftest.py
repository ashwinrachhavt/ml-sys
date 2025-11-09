import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if "mlflow" not in sys.modules:

    def _noop(*args, **kwargs):  # pragma: no cover - test stub
        return None

    class _DummyModel:
        def predict(self, features):
            import numpy as _np

            if hasattr(features, "shape"):
                return _np.zeros(len(features))
            return [0] * len(features)

        def predict_proba(self, features):
            import numpy as _np

            length = len(features)
            return _np.tile(_np.array([[0.5, 0.5]]), (length, 1))

    mlflow_module = types.ModuleType("mlflow")
    mlflow_module.set_tracking_uri = _noop
    mlflow_module.set_experiment = _noop
    mlflow_module.start_run = lambda *a, **k: types.SimpleNamespace(info=types.SimpleNamespace(run_id="stub"))
    mlflow_module.end_run = _noop
    mlflow_module.log_params = _noop
    mlflow_module.log_metrics = _noop
    mlflow_module.log_text = _noop
    mlflow_module.log_param = _noop
    mlflow_module.register_model = lambda *a, **k: types.SimpleNamespace(version="1")
    mlflow_module.active_run = lambda: types.SimpleNamespace(info=types.SimpleNamespace(run_id="stub"))

    sklearn_stub = types.ModuleType("mlflow.sklearn")
    sklearn_stub.log_model = _noop
    sklearn_stub.load_model = lambda *a, **k: _DummyModel()
    mlflow_module.sklearn = sklearn_stub

    artifacts_stub = types.ModuleType("mlflow.artifacts")
    artifacts_stub.load_text = lambda *a, **k: ""

    class _StubClient:
        def get_latest_versions(self, *args, **kwargs):
            return []

        def get_run(self, run_id):
            return types.SimpleNamespace(data=types.SimpleNamespace(params={}, metrics={}))

        def create_model_version(self, name, uri, run_id):
            return types.SimpleNamespace(version="1")

        def transition_model_version_stage(self, *args, **kwargs):
            return None

    tracking_stub = types.ModuleType("mlflow.tracking")
    tracking_stub.MlflowClient = _StubClient

    exceptions_stub = types.ModuleType("mlflow.exceptions")

    class StubMlflowError(Exception):
        pass

    exceptions_stub.MlflowException = StubMlflowError

    sys.modules.setdefault("mlflow", mlflow_module)
    sys.modules.setdefault("mlflow.sklearn", sklearn_stub)
    sys.modules.setdefault("mlflow.artifacts", artifacts_stub)
    sys.modules.setdefault("mlflow.tracking", tracking_stub)
    sys.modules.setdefault("mlflow.exceptions", exceptions_stub)


if "imblearn" not in sys.modules:
    over_sampling_module = types.ModuleType("imblearn.over_sampling")

    class _StubSMOTE:
        def __init__(self, *args, **kwargs):
            pass

        def fit_resample(self, data, y):
            return data, y

    over_sampling_module.SMOTE = _StubSMOTE
    imblearn_module = types.ModuleType("imblearn")
    imblearn_module.over_sampling = over_sampling_module

    sys.modules.setdefault("imblearn", imblearn_module)
    sys.modules.setdefault("imblearn.over_sampling", over_sampling_module)
