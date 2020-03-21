#!/usr/bin/env python3
"""
xxx
"""
# region imports

# pylint: disable=unused-import

import functools  # noqa: F401
import pathlib  # noqa: F401
import random  # noqa: F401

import albumentations as A  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import sklearn.datasets  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.linear_model  # noqa: F401
import sklearn.metrics  # noqa: F401
import sklearn.model_selection  # noqa: F401
import tensorflow as tf  # noqa: F401
import tensorflow_addons as tfa  # noqa: F401

import _data
import pytoolkit as tk

# endregion

num_classes = 9
nfold = 5
split_seed = 1
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
params = {
    # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "depth": 6,
    "iterations": 9999,
    "early_stopping_rounds": 400,
    "verbose": 100,
    "random_seed": 123,
    "train_dir": str(models_dir / "catboost_info"),
}
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


def create_model():
    return tk.pipeline.CBModel(params=params, nfold=nfold, models_dir=models_dir)


# region data/score


def load_train_data():
    dataset = _data.load_train_data()
    return dataset


def load_test_data():
    dataset = _data.load_test_data()
    return dataset


def score(
    y_true: tk.data.LabelsType, y_pred: tk.models.ModelIOType
) -> tk.evaluations.EvalsType:
    return tk.evaluations.evaluate_classification(y_true, y_pred)


# endregion

# region commands


@app.command(then="validate")
def train():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    model.cv(train_set, folds)


@app.command(then="predict")
def validate():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model().load()
    pred = model.predict_oof(train_set, folds)
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_train.pkl")
        tk.notifications.post_evals(score(train_set.labels, pred))


@app.command()
def predict():
    test_set = load_test_data()
    model = create_model().load()
    pred_list = model.predict_all(test_set)
    pred = np.mean(pred_list, axis=0)
    if tk.hvd.is_master():
        tk.utils.dump(pred_list, models_dir / "pred_test.pkl")
        _data.save_prediction(models_dir, test_set, pred)


# endregion


if __name__ == "__main__":
    app.run(default="train")
