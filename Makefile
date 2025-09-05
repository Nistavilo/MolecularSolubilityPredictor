PYTHON ?= python

.PHONY: train predict test lint setup

setup:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Setup complete."

train:
	$(PYTHON) -m src.training.train --config config/config.yaml

predict:
	$(PYTHON) -m src.inference.predict --smiles "CCO"

batch-predict:
	$(PYTHON) -m src.inference.predict --input data/example_smiles.csv --output preds.csv

test:
	pytest -q

lint:
	ruff check . || true

format:
	ruff check . --fix
	black .