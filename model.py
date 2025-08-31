""" Siamese Network with a Backbone of Resnet50 """
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import resnet

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def call(self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

def siamese_model(target_shape: tuple[int, int], pre_weights: Optional[str] = "imagenet", trainable: bool = False) -> Model:
    base_cnn = resnet.ResNet50(weights=pre_weights, include_top=False, input_shape=target_shape + (3,))
    if not trainable:
        torf = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                torf = True
            layer.trainable = torf
    flatten = layers.GlobalAveragePooling2D()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.Dropout(0.6)(dense1)
    dense2 = layers.Dense(512, activation="relu")(dense1)
    dense2 = layers.Lambda(lambda p: K.l2_normalize(p, axis=1))(dense2)
    embedding = Model(inputs=[base_cnn.input], outputs=dense2, name="Embedding")
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))
    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
