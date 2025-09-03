import pandas as pd
from datasets import load_dataset

def load_delaney():
    ds = load_dataset("zpn/delaney")
    train_df = pd.DataFrame(ds["train"])[["smiles", "target"]]
    valid_df = pd.DataFrame(ds["validation"])[["smiles", "target"]]
    test_df  = pd.DataFrame(ds["test"])[["smiles", "target"]]
    return train_df, valid_df, test_df