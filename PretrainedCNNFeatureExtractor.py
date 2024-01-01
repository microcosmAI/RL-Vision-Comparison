import torch as th
import torch.nn as nn
import torchvision
from gymnasium import spaces
from gymnasium.wrappers import FrameStack
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from CustomCNN import CustomCNN
from utils import hyperparam_search

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# there needs to be a way to be able to get these and pass them to the
# features extractor kwargs dict without hardcoding
EFFICIENTNET_NUM_FEATURES = 1280 
RESNET50_NUM_FEATURES = 2048
SQUEEZENET1_NUM_FEATURES = 1000
NUM_FEATURES_MAP = {"resnet50": RESNET50_NUM_FEATURES, "efficientnet":EFFICIENTNET_NUM_FEATURES, "squeezenet1": SQUEEZENET1_NUM_FEATURES}

def create_grayscale_preprocessing(weights):
    return nn.Sequential(Grayscale(), weights.transforms())

def efficientnet():
    efficientnet_weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
    efficientnet_model = torchvision.models.efficientnet_b1(weights=efficientnet_weights).to(device)
    efficientnet_model.classifier = nn.Identity()
    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = efficientnet_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]
    print(f"efficient net num features: {num_features}")
    return efficientnet_model, efficientnet_weights, num_features

def resnet50():
    resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet50_model = torchvision.models.resnet50(weights=resnet50_weights).to(device)
    resnet50_model.fc = nn.Identity()
    print(resnet50_model)
    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = resnet50_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]
    print(f"Resnet 50 num features: {num_features}")
    return resnet50_model, resnet50_weights, num_features

def squeezenet1():
    squeezenet1_weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
    squeezenet1_model = torchvision.models.squeezenet1_1(weights=squeezenet1_weights).to(device)
    squeezenet1_model.fc = nn.Identity()
    print(squeezenet1_model)
    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = squeezenet1_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]
    print(f"Squeezenet 1 num features: {num_features}")
    return squeezenet1_model, squeezenet1_weights, num_features

NETWORK_VARS_MAP = {"resnet50":resnet50, "efficientnet":efficientnet, "squeezenet1": squeezenet1}

class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.expand(img.shape[0], 3, *img.shape[2:])

class PretrainedCNNFeatureExtractor(CustomCNN):

    @CustomCNN.weights.setter
    def weights(self, weights):
        self._weights = weights # it doesn't seem possible to call the parent setter
        self.preprocessing_function = weights.transforms() # this may need Grayscale

    @CustomCNN.preprocessing_function.setter
    def preprocessing_function(self, preprocessing_function):
        if preprocessing_function is None:
            preprocessing_function = self.weights.transforms()
        self._preprocessing_function = preprocessing_function

    def __init__(self, observation_space: spaces.Box, features_dim: int, base_model = None, weights = None, preprocessing_function = None):

        super(CustomCNN, self).__init__(observation_space, features_dim)
        if base_model is None or weights is None:
            print("Defaulting to using ResNet50 model because either weights or model was not supplied.")
            base_model, weights, _ = resnet50()
            preprocessing_function = nn.Sequential(Grayscale(), weights.transforms()) # maybe the grayscale needs to be an option
        self.weights = weights
        self.model = base_model
        if preprocessing_function:
            self.preprocessing_function = preprocessing_function
        else:
            self.preprocessing_function = lambda x : x # should this become a torch Identity?
