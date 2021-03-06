#!/usr/bin/env python3
"""
logloss: 0.422
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

params = {
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 9,
    "learning_rate": 0.01,
    "bagging_freq": 1,
    "bagging_fraction": 0.75,
    "feature_fraction": 0.25,
    "verbosity": -1,
    "nthread": 4,  # -1
}
num_classes = 9
nfold = 5
split_seed = 1
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


def create_model():
    return tk.pipeline.LGBModel(params=params, nfold=nfold, models_dir=models_dir)


# region data/score


def load_mf(name):
    mf = np.concatenate(
        [
            tk.utils.load(f"models/lv1_cb/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_ert/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_knn_5/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_knn_5c/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_32/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_64/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_128/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_256/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_512/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_1024/pred_{name}.pkl"),
            # tk.utils.load(f"models/lv1_knn_1024c/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_lgb/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_nn/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_nn2/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_nn3/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_rf/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_rgf/pred_{name}.pkl"),
            tk.utils.load(f"models/lv1_xgb/pred_{name}.pkl"),
        ],
        axis=-1,
    )
    return mf


def load_train_data():
    dataset = _data.load_train_data()
    dataset.data = load_mf("train")
    return dataset


def load_test_data():
    mf = load_mf("test")
    dataset = _data.load_train_data()
    for mf_i in mf:
        dataset.data = mf_i
        yield dataset


def score(
    y_true: tk.data.LabelsType, y_pred: tk.models.ModelIOType
) -> tk.evaluations.EvalsType:
    return tk.evaluations.evaluate_classification(y_true, y_pred)


# endregion

# region commands


@app.command()
def train_only():
    train()


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
    test_set_list = list(load_test_data())
    model = create_model().load()
    pred_list = []
    for test_set in test_set_list:
        pred_list.extend(model.predict_all(test_set))
    pred = np.mean(pred_list, axis=0)
    if tk.hvd.is_master():
        tk.utils.dump(pred_list, models_dir / "pred_test.pkl")
        _data.save_prediction(models_dir, test_set_list[0], pred)


# endregion


if __name__ == "__main__":
    app.run(default="train")
