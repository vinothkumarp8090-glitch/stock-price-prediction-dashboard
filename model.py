from __future__ import annotations

from typing import Literal

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Attention,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam


TaskType = Literal["regression", "classification"]


def _output_config(task_type: TaskType) -> tuple[str, str, list[str]]:
    if task_type == "classification":
        return "sigmoid", "binary_crossentropy", ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    return "linear", "mse", [tf.keras.metrics.MeanAbsoluteError(name="mae")]


def build_lstm_model(input_shape: tuple[int, int], task_type: TaskType = "regression") -> Model:
    activation, loss, metrics = _output_config(task_type)
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1, activation=activation),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
    return model


def build_bilstm_attention_model(input_shape: tuple[int, int], task_type: TaskType = "regression") -> Model:
    activation, loss, metrics = _output_config(task_type)
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    attn_output = Attention(use_scale=True)([x, x])
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation=activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
    return model


def build_transformer_model(input_shape: tuple[int, int], task_type: TaskType = "regression") -> Model:
    activation, loss, metrics = _output_config(task_type)
    inputs = Input(shape=input_shape)

    x = Dense(64)(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    ff = Dense(128, activation="relu")(x)
    ff = Dropout(0.1)(ff)
    ff = Dense(64)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation=activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
    return model


def get_model_builder(model_name: str):
    builders = {
        "lstm": build_lstm_model,
        "bilstm_attention": build_bilstm_attention_model,
        "transformer": build_transformer_model,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(builders)}")
    return builders[model_name]
