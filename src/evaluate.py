import os
import json
import datetime as _dt
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import joblib
except ImportError:  # nadiren joblib yoksa
    joblib = None


def _to_1d(a):
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()
    return a


def regression_metrics(y_true, y_pred):
    """
    Döndürür:
        {"MAE": float, "RMSE": float, "R2": float}
    squared=False yoksa fallback yapar.
    """
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")

    mae = mean_absolute_error(y_true, y_pred)

    # Eski / gölgelenmiş sklearn olasılığına karşı güvenli RMSE:
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

    r2 = r2_score(y_true, y_pred)

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def _safe_json_dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_experiment(
    output_dir: str,
    model: Any = None,
    metrics: Optional[Dict[str, float]] = None,
    feature_names: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    raw_args: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    exist_ok: bool = True,
) -> Dict[str, str]:
    """
    Deney (model + metrik + metadata) kaydeder.

    output_dir:
        Kaydedilecek klasör (oluşturulur).
    model:
        joblib.dump ile serileştirilecek nesne (opsiyonel).
    metrics:
        Metrik sözlüğü.
    feature_names:
        Özellik isimleri (liste).
    params:
        Model / eğitim parametreleri.
    raw_args:
        Komut satırı argümanları (sys.argv gibi).
    extra:
        Ek sözlük (ör: split boyutları).
    prefix:
        Dosya adlarına (metrics.json -> <prefix>metrics.json) ön ek.
    exist_ok:
        False ise zaten varsa ValueError fırlatır.

    Dönüş: {"model": "...", "metrics": "...", "metadata": "...", "features": "...?"}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    elif not exist_ok:
        raise ValueError(f"Output directory already exists: {output_dir}")

    timestamp = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    paths = {}

    # Metrikler
    if metrics:
        metrics_path = os.path.join(output_dir, f"{prefix}metrics.json")
        _safe_json_dump(metrics_path, {"timestamp": timestamp, "metrics": metrics})
        paths["metrics"] = metrics_path

    # Özellik isimleri
    if feature_names:
        feats_path = os.path.join(output_dir, f"{prefix}feature_names.txt")
        with open(feats_path, "w", encoding="utf-8") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        paths["features"] = feats_path

    # Model
    if model is not None and joblib is not None:
        model_path = os.path.join(output_dir, f"{prefix}model.pkl")
        joblib.dump(model, model_path)
        paths["model"] = model_path

    # Metadata
    meta = {
        "timestamp_utc": timestamp,
        "has_model": model is not None,
        "n_features": len(feature_names) if feature_names else None,
        "params": params,
        "raw_args": list(raw_args) if raw_args else None,
        "extra": extra,
    }
    if model is not None:
        meta["model_class"] = type(model).__name__
        meta["model_module"] = type(model).__module__

    meta_path = os.path.join(output_dir, f"{prefix}metadata.json")
    _safe_json_dump(meta_path, meta)
    paths["metadata"] = meta_path

    return paths