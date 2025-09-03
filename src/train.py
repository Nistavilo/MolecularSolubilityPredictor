import argparse
import os
from src.data import load_delaney
from src.features import featurize_rows, build_feature_names
from src.models import build_model
from src.evaluate import regression_metrics, save_experiment
from src.utils import set_seed, print_metrics

def parse_args():
    ap = argparse.ArgumentParser(description="ESOL logS model eğitimi")
    ap.add_argument("--model", type=str, default="random_forest", help="random_forest | lightgbm")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=1024)
    ap.add_argument("--merge_valid", action="store_true", help="En iyi yapı biliniyorsa train+valid birleştir.")
    ap.add_argument("--output_model", type=str, default="models/model.pkl")
    ap.add_argument("--experiment_out", type=str, default="experiments/last_experiment.json")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.experiment_out), exist_ok=True)

    train_df, valid_df, test_df = load_delaney()

    X_train, y_train, _, desc_keys = featurize_rows(train_df.smiles.tolist(), train_df.target.values,
                                                   radius=args.radius, n_bits=args.n_bits)
    X_valid, y_valid, _, _ = featurize_rows(valid_df.smiles.tolist(), valid_df.target.values,
                                           radius=args.radius, n_bits=args.n_bits)
    X_test, y_test, test_smiles, _ = featurize_rows(test_df.smiles.tolist(), test_df.target.values,
                                                   radius=args.radius, n_bits=args.n_bits)

    if args.merge_valid:
        import numpy as np
        X_train = np.vstack([X_train, X_valid])
        y_train = np.concatenate([y_train, y_valid])

    model = build_model(args.model)

    model.fit(X_train, y_train)

    # Değerlendirme
    from sklearn.metrics import mean_squared_error
    y_pred_test = model.predict(X_test)
    metrics_test = regression_metrics(y_test, y_pred_test)
    print("=== TEST SONUÇLARI ===")
    print_metrics(metrics_test)

    # Kaydet
    import joblib
    joblib.dump({"model": model, "desc_keys": desc_keys,
                 "radius": args.radius, "n_bits": args.n_bits},
                args.output_model)

    # Deney json
    save_experiment(args.experiment_out,
                    args.model,
                    model.get_params(),
                    metrics_test)

if __name__ == "__main__":
    main()