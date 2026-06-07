import joblib

from gex_core.predict import _load_joblib_cached, clear_model_cache


def test_joblib_cache_reuses_artifact(tmp_path, monkeypatch):
    clear_model_cache()
    path = tmp_path / "model.joblib"
    joblib.dump({"value": 1}, path)

    loads = {"n": 0}
    original_load = joblib.load

    def counting_load(p):
        loads["n"] += 1
        return original_load(p)

    monkeypatch.setattr("gex_core.predict.joblib.load", counting_load)

    first = _load_joblib_cached(path)
    second = _load_joblib_cached(path)

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert loads["n"] == 1


def test_clear_model_cache_forces_reload(tmp_path):
    clear_model_cache()
    path = tmp_path / "model.joblib"
    joblib.dump({"value": 1}, path)
    first = _load_joblib_cached(path)
    clear_model_cache()
    joblib.dump({"value": 2}, path)
    second = _load_joblib_cached(path)
    assert first == {"value": 1}
    assert second == {"value": 2}
