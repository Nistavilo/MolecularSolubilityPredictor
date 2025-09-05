# Molecular Solubility Predictor

A machine learning project for predicting molecular (aqueous) solubility using classical cheminformatics features and gradient boosting / traditional ML models (e.g. LightGBM, Random Forest, etc.).  
The goal is to provide a reproducible pipeline from raw molecular data to a trained, evaluable, and deployable regression model.

---

## Key Features

- Configurable data preprocessing (feature extraction & cleaning)
- Support for multiple regression models (initially LightGBM & scikit-learn estimators)
- Train / validation / test split utilities
- Model persistence with `joblib`
- Basic experiment tracking via saved artifacts in the `experiments/` directory
- Extensible architecture for adding new feature generators or models
- Test suite scaffold in `tests/` (to be expanded)

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── experiments/          # Saved experiment configs, metrics, plots, logs
├── models/               # Serialized trained models (e.g. .joblib files)
├── src/                  # Source code (data prep, feature eng., training, inference)
└── tests/                # Unit / integration tests
```

(Directory internals will evolve; update this section as the codebase grows.)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Nistavilo/MolecularSolubilityPredictor.git
cd MolecularSolubilityPredictor
```

### 2. Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.\.venv\Scripts\activate         # Windows (PowerShell)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dependencies (from `requirements.txt`)
- scikit-learn
- pandas
- numpy
- matplotlib
- datasets (placeholder or Hugging Face `datasets` if intended)
- joblib
- lightgbm

If you plan to compute cheminformatics descriptors (e.g. RDKit), add it:
```bash
conda install -c conda-forge rdkit
```
(or update `requirements.txt` accordingly if wheels are acceptable for your environment.)

---

## Usage

### 1. Data Preparation
Place your raw molecular dataset (e.g. CSV with SMILES + solubility column) in a `data/` directory (you may create it).  
Example expected columns:
```
smiles,solubility
CCO,-1.23
...
```

Add a data loading script under `src/data/` (e.g. `loader.py`) that:
- Reads the file
- Validates schema
- (Optionally) filters invalid SMILES

### 2. Feature Engineering
Typical approaches:
- Molecular descriptors (e.g., RDKit: MolWt, TPSA, LogP)
- Fingerprints (ECFP/Morgan)
- Simple counts (H-bond donors/acceptors, rotatable bonds)

Implement a feature module (e.g., `src/features/descriptors.py`) returning `X (DataFrame/ndarray), y (Series/array)`.

### 3. Training
Create a training script (e.g., `src/train.py`) that:
- Loads features
- Splits data (train/valid/test)
- Trains a model (e.g., LightGBMRegressor)
- Saves:
  - Model → `models/model_name.joblib`
  - Metrics → `experiments/<run_id>/metrics.json`
  - Plots (learning curves, residuals) → `experiments/<run_id>/plots/`

Example (conceptual) snippet:
```python
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

X, y = load_features()  # implement this
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
mse = mean_squared_error(y_val, preds)

dump(model, "models/solubility_lightgbm.joblib")
with open("experiments/run_001/metrics.json", "w") as f:
    json.dump({"val_mse": mse}, f, indent=2)
```

### 4. Inference
Create `src/predict.py` to:
- Load the serialized model
- Accept SMILES (single or batch)
- Generate features with the same pipeline
- Output predicted solubility

---

## Model Evaluation

Recommended regression metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R²
- (Optional) Spearman/Pearson correlation

You may also visualize:
- Predicted vs True scatter
- Residual histogram
- Error distribution across molecular weight bins

---

## Extending the Project

Add new models:
- Implement a wrapper in `src/models/`
- Register in a model factory (e.g., `get_model(name)`)

Add new feature sets:
- Create modules in `src/features/`
- Keep transformations deterministic & documented

Add experiment configs:
- YAML/JSON under `experiments/configs/` referencing:
  - model type
  - feature set
  - hyperparameters
  - random seed

---

## Roadmap (Proposed)

- [ ] Add RDKit descriptor extraction
- [ ] Implement feature caching
- [ ] Add configuration-driven training (YAML)
- [ ] Hyperparameter tuning (Optuna or scikit-learn GridSearchCV)
- [ ] CLI interface (argparse / Typer)
- [ ] Dockerfile for reproducible runs
- [ ] Add unit tests for feature pipeline
- [ ] Model comparison report
- [ ] Deployment demo (FastAPI or Streamlit)

---

## Testing

Populate `tests/` with:
- `test_features.py` (descriptor correctness)
- `test_training.py` (model trains & saves artifacts)
- `test_predict.py` (consistent inference)

Run with:
```bash
pytest -q
```

---

## Contributing

1. Fork & create a feature branch.
2. Follow a consistent code style (PEP8; consider `ruff` or `black`).
3. Ensure tests pass before PR.
4. Update this README if structure changes.

---

## Potential Data Sources

(You must ensure licensing and proper citation.)
- AqSolDB
- ESOL dataset
- Other curated solubility datasets

Include a `data/README.md` describing preprocessing if distributing derived data.

---

## License

Specify your intended license (e.g., MIT, Apache-2.0) and create a `LICENSE` file.  
(Currently not specified.)

---

## Citation

If this project supports academic work, add a BibTeX entry here later.

---

## Disclaimer

This repository is for research and educational purposes. Predictions may not be suitable for real-world pharmaceutical decision-making without validation.

---

## Contact

Author: (GitHub) https://github.com/Nistavilo  
Feel free to open an Issue or Discussion for questions or suggestions.

---

Happy modeling!