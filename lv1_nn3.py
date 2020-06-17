#!/usr/bin/env python3
"""
acc:     0.818
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
batch_size = 2048
nfold = 5
split_seed = 1
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        score_fn=score,
        nfold=nfold,
        models_dir=models_dir,
        train_data_loader=MyDataLoader(mode="train"),
        refine_data_loader=MyDataLoader(mode="refine"),
        val_data_loader=MyDataLoader(mode="test"),
        epochs=100,
        refine_epochs=5,
        base_models_dir=None,
        callbacks=[tk.callbacks.CosineAnnealing()],
        # parallel_cv=True,
        skip_if_exists=False,  # TODO: 実験用
    )


# region data/score


def load_check_data():
    dataset = _data.load_check_data()
    return dataset


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


@app.command(logfile=False)
def check():  # utility
    create_model().check(load_check_data())


@app.command(logfile=False)
def migrate():  # utility
    create_model().load().save()


@app.command(use_horovod=True, then="validate")
def train():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    model.cv(train_set, folds)


@app.command(use_horovod=True, then="predict")
def validate():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model().load()
    pred = model.predict_oof(train_set, folds)
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_train.pkl")
        tk.notifications.post_evals(score(train_set.labels, pred))


@app.command(use_horovod=True)
def predict():
    test_set = load_test_data()
    model = create_model().load()
    pred_list = model.predict_all(test_set)
    pred = np.mean(pred_list, axis=0)
    if tk.hvd.is_master():
        tk.utils.dump(pred_list, models_dir / "pred_test.pkl")
        _data.save_prediction(models_dir, test_set, pred)


# endregion


def create_network():
    dense = functools.partial(
        tf.keras.layers.Dense,
        use_bias=False,
        kernel_initializer="he_uniform",
        # kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tf.keras.layers.BatchNormalization,
        gamma_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    act = functools.partial(tf.keras.layers.Activation, "relu")

    inputs = x = tf.keras.layers.Input((93,))
    x = dense(1024)(x)
    x = bn()(x)
    x = act()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = dense(1024)(x)
    x = bn()(x)
    x = act()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = dense(9, kernel_initializer="zeros", use_bias=True)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    def loss(y_true, logits):
        return tk.losses.categorical_focal_loss(y_true, logits, from_logits=True)

    tk.models.compile(model, "adam", loss, ["acc"])

    x = tf.keras.layers.Activation("softmax")(x)
    prediction_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model, prediction_model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, mode):
        super().__init__(batch_size=batch_size)
        self.mode = mode

    def get_ds(
        self,
        dataset: tk.data.Dataset,
        shuffle: bool,
        without_label: bool,
        num_replicas_in_sync: int,
    ) -> tf.data.Dataset:
        assert isinstance(dataset.data, pd.DataFrame)
        X = dataset.data.values
        if dataset.labels is None:
            y = np.zeros((len(X), num_classes), dtype=np.float32)  # dummy
        else:
            y = tf.one_hot(dataset.labels, num_classes)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=len(dataset)) if shuffle else ds
        if False and self.mode == "train":
            assert self.data_per_sample == 2
            ds = tk.data.mixup(ds)
        else:
            assert self.data_per_sample == 1
            ds = ds.repeat() if shuffle else ds  # バッチサイズを固定するため先にrepeat
        ds = ds.batch(self.batch_size * num_replicas_in_sync)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


if __name__ == "__main__":
    app.run(default="train")
