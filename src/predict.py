import argparse
import joblib
from src.features import featurize_rows

def parse_args():
    ap = argparse.ArgumentParser(description="Yeni SMILES için logS tahmini")
    ap.add_argument("--model_path", type=str, default="models/model.pkl")
    ap.add_argument("--smiles", type=str, required=True, help="Tek bir SMILES")
    return ap.parse_args()

def main():
    args = parse_args()
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    radius = bundle["radius"]
    n_bits = bundle["n_bits"]

    X, _, kept_smiles, desc_keys = featurize_rows([args.smiles], targets=None,
                                                  radius=radius, n_bits=n_bits)
    y_pred = model.predict(X)[0]
    print(f"SMILES: {kept_smiles[0]}  Tahmin logS: {y_pred:.4f}")

if __name__ == "__main__":
    main()