# utils.py
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def _load_csv_df(path: str) -> pd.DataFrame:
    # Try with header; if first col isn't 'label', load without header.
    try:
        df = pd.read_csv(path)
        if df.columns[0] != "label":
            df = pd.read_csv(path, header=None)
            df.rename(columns={0: "label"}, inplace=True)
    except pd.errors.ParserError:
        df = pd.read_csv(path, header=None)
        df.rename(columns={0: "label"}, inplace=True)
    return df

def load_csv_pair(train_csv="mnist_train.csv", test_csv="mnist_test.csv", bs=128, pin=False, workers=2):
    for f in (train_csv, test_csv):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} next to train.py")

    tr_df = _load_csv_df(train_csv)
    te_df = _load_csv_df(test_csv)

    Xtr = (tr_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0).reshape(-1, 1, 28, 28)
    ytr = tr_df.iloc[:, 0].to_numpy(dtype=np.int64)
    Xte = (te_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0).reshape(-1, 1, 28, 28)
    yte = te_df.iloc[:, 0].to_numpy(dtype=np.int64)

    Xtr = torch.from_numpy(Xtr); ytr = torch.from_numpy(ytr)
    Xte = torch.from_numpy(Xte); yte = torch.from_numpy(yte)

    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                    num_workers=workers, pin_memory=pin)
    te = DataLoader(TensorDataset(Xte, yte), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=pin)
    return tr, te

def wait_key(msg):
    try:
        input(msg)
    except EOFError:
        print("(no stdin; continuing)")
