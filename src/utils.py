import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Eğer torch vs eklenirse burada ayarlanır

def print_metrics(metrics: dict, prefix=""):
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")