import torch as th
import torch.nn as nn
import torchvision
from gymnasium import spaces
from gymnasium.wrappers import FrameStack
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from HParamCallback import HParamCallback

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.expand(img.shape[0], 3, *img.shape[2:])

class CustomCNN(BaseFeaturesExtractor):

    @property
    def model(self):
        return self._model

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @model.setter
    def model(self, base_model):
        self._model = base_model
        print(f"Using {base_model._get_name()} as the base model.") # somehow this statement doesn't get run?

    @property
    def preprocessing_function(self):
        return self._preprocessing_function

    @preprocessing_function.setter
    def preprocessing_function(self, preprocessing_function):
        if preprocessing_function is None:
            preprocessing_function = self.weights.transforms()
        self._preprocessing_function = preprocessing_function

    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    :param base_model: PyTorch model
    :param weights: PyTorch weights for the model
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int, base_model = None, weights = None, preprocessing_function = None):

        super().__init__(observation_space, features_dim)
        self.weights = weights
        print("setting model")
        self.model = base_model
        print("model is set")
        if preprocessing_function:
            self.preprocessing_function = preprocessing_function
        else:
            self.preprocessing_function = lambda x : x # should this become a torch Identity?

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.to(device)
        preprocessed_observations = self.preprocessing_function(observations)
        called = self.model(preprocessed_observations)
        return called
