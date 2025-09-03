from src.features import featurize_rows

def test_featurize_simple():
    smiles = ["CCO", "CCN", "c1ccccc1"]
    X, y, kept, desc_keys = featurize_rows(smiles, targets=[0,0,0])
    assert X.shape[0] == 3
    # 1024 bit + 6 descriptor varsayımı
    assert X.shape[1] == 1024 + len(desc_keys)