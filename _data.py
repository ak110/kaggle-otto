"""データの読み書きなど"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

data_dir = pathlib.Path(f"data")
logger = tk.log.get(__name__)


def load_check_data():
    """チェック用データの読み込み"""
    return load_train_data().slice(list(range(0, 61878, 1000)))


def load_train_data():
    """訓練データの読み込み"""
    X_train = pd.read_csv(data_dir / "train.csv")
    y_train = (
        np.char.replace(X_train["target"].values.astype(str), "Class_", "").astype(
            np.uint8
        )
        - 1
    )
    X_train = X_train.drop(columns=["id", "target"])
    return tk.data.Dataset(preprocess(X_train), y_train)


def load_test_data():
    """テストデータの読み込み"""
    X_test = pd.read_csv(data_dir / "test.csv")
    ids_test = X_test["id"]
    X_test = X_test.drop(columns=["id"])
    return tk.data.Dataset(preprocess(X_test), ids=ids_test)


def preprocess(X):
    """前処理"""
    X = np.log1p(X)
    return X


def save_prediction(models_dir, test_set, pred):
    """テストデータの予測結果の保存"""
    df = pd.DataFrame()
    df["id"] = np.arange(1, len(test_set) + 1)
    for i in range(9):
        df[f"Class_{i + 1}"] = pred[:, i]
    df.to_csv(models_dir / "submission.csv", index=False)
