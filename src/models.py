from sklearn.ensemble import RandomForestRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except:
    HAS_LGB = False

def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "random_forest":
        params = dict(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        params.update(kwargs)
        return RandomForestRegressor(**params)
    elif name == "lightgbm":
        if not HAS_LGB:
            raise ImportError("LightGBM yüklü değil. pip install lightgbm")
        params = dict(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        params.update(kwargs)
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")