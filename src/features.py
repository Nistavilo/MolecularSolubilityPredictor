from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

# Temel descriptor fonksiyonları
_BASE_DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("LogP", Descriptors.MolLogP),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("TPSA", Descriptors.TPSA),
    ("RingCount", Descriptors.RingCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("FractionCSP3", Descriptors.FractionCSP3),
]

DEFAULT_FP_RADIUS = 2
DEFAULT_FP_BITS = 2048

def build_feature_names(include_fingerprint=True, fp_bits=DEFAULT_FP_BITS):
    names = [name for name, _ in _BASE_DESCRIPTOR_FUNCS]
    if include_fingerprint:
        names.extend([f"FP_{i}" for i in range(fp_bits)])
    return names

def _mol_from_smiles(s):
    m = Chem.MolFromSmiles(s)
    return m

def _calc_base_descriptors(mol):
    vals = []
    for _, fn in _BASE_DESCRIPTOR_FUNCS:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return vals

def _calc_morgan_bits(mol, radius=DEFAULT_FP_RADIUS, n_bits=DEFAULT_FP_BITS):
    arr = np.zeros((n_bits,), dtype=int)
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr.tolist()

def featurize_rows(
    smiles_list,
    y=None,
    radius=None,
    n_bits=None,
    fp_radius=None,
    fp_bits=None,
    include_fingerprint=True,
    replace_invalid_with_nan=True,
    return_invalid_mask=True,
    return_feature_names=True,
    **kwargs
):
    """
    UYUMLULUK WRAPPER'I:
    train.py eski bir imzayla (smiles, y, radius=..., n_bits=...) çağırıyor olsa bile çalışır.
    
    Dönüş (her zaman 4 eleman):
        X (numpy array)
        y_out (numpy array veya None)
        invalid_mask (numpy bool array veya None)  # Hatalı SMILES True
        feature_names (list[str])
    """
    # Parametre önceliklendirme
    if fp_radius is None:
        fp_radius = radius if radius is not None else DEFAULT_FP_RADIUS
    if fp_bits is None:
        if n_bits is not None:
            fp_bits = n_bits
        else:
            fp_bits = DEFAULT_FP_BITS

    feature_names = build_feature_names(include_fingerprint=include_fingerprint, fp_bits=fp_bits)

    rows = []
    invalid_mask = []
    for smi in smiles_list:
        mol = _mol_from_smiles(smi)
        if mol is None:
            invalid_mask.append(True)
            if replace_invalid_with_nan:
                base = [np.nan] * len(_BASE_DESCRIPTOR_FUNCS)
                if include_fingerprint:
                    base.extend([0] * fp_bits)
                rows.append(base)
            else:
                # Eğer invalid satırlar atlanacaksa y'yi de filtrelememiz gerekir
                continue
        else:
            invalid_mask.append(False)
            base = _calc_base_descriptors(mol)
            if include_fingerprint:
                base.extend(_calc_morgan_bits(mol, radius=fp_radius, n_bits=fp_bits))
            rows.append(base)

    X = np.array(rows, dtype=float)

    # y hizalama
    if y is not None:
        y = np.asarray(y)
        if len(y) != len(smiles_list):
            raise ValueError(f"y uzunluğu ({len(y)}) SMILES uzunluğuyla ({len(smiles_list)}) eşleşmiyor.")
        y_out = y
    else:
        y_out = None

    invalid_mask = np.array(invalid_mask, dtype=bool) if return_invalid_mask else None
    if not return_feature_names:
        feature_names = None

    # Train script'i 4 değer bekliyorsa uyum:
    return X, y_out, invalid_mask, feature_names