import torch as th
import torch.nn as nn
import tensorflow as tf
import keras
from tensorflow.keras.applications.xception import preprocess_input
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def convert_image_tensor_torch_to_tf(tensor_torch: th.Tensor) -> tf.Tensor:
    tensor_torch_permuted = tensor_torch.permute(0, 2, 3, 1)
    np_array = tensor_torch_permuted.numpy()
    tensor_tf = tf.convert_to_tensor(np_array)
    return tensor_tf

def convert_tensor_tf_to_torch(tensor_tf: tf.Tensor) -> th.Tensor:
    tensor_np = tensor_tf.numpy()
    tensor_torch = th.from_numpy(tensor_np)
    return tensor_torch


class CustomCNN(BaseFeaturesExtractor):

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, base_model):
        if base_model is None:
            print("Defaulting to using Xception model.")
            base_model = tf.keras.applications.Xception(
                include_top=False,
                input_shape=(299,299,3),
                weights="imagenet",
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
            )
        base_model.trainable = False

        inputs = keras.Input(shape=(299, 299, 3))
        with_inputs = base_model(inputs, training=False)
        with_pooling = keras.layers.GlobalAveragePooling2D()(with_inputs)
        outputs = keras.layers.Dense(self.features_dim, activation='relu')(with_pooling)

        self._model = keras.Model(inputs, outputs)
        print(f"Using {base_model.name} as the base model.")

    @property
    def preprocessing_function(self):
        return self._preprocessing_function

    @preprocessing_function.setter
    def preprocessing_function(self, preprocessing_function):
        if preprocessing_function is None:
            preprocessing_function = tf.keras.applications.xception.preprocess_input
        self._preprocessing_function = preprocessing_function
        print(f"Using {preprocessing_function.__qualname__} as preprocessing function.")

    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, base_model = None, preprocessing_function = None):

        super().__init__(observation_space, features_dim)

        self.model = base_model

        self.preprocessing_function = preprocessing_function

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations_tf = convert_image_tensor_torch_to_tf(observations)
        resized_observations = tf.image.resize(observations_tf, [299, 299])
        preprocessed_observations = self.preprocessing_function(resized_observations)
        called = self.model(preprocessed_observations)
        called_torch = convert_tensor_tf_to_torch(called)
        return called_torch


if __name__ == "__main__":
    """Testing with ResNet101. If a model isn't supplied in the policy_kwargs, the Xception
        model from tf.keras.applications is used as the default model."""

    ResNet101 = tf.keras.applications.ResNet101(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(299,299,3),
        pooling=None,
        classes=1000
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128, base_model=ResNet101, 
                                       preprocessing_function=tf.keras.applications.resnet.preprocess_input),
    )

    model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    model.learn(4000)
    model.save("4ktimesteps.zip")
