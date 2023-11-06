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

def hyperparam_search(lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps):
    for lr in lr_values:
        for net_arch in net_arch_values:
            for batch_size in batch_size_values:
                model_name = f"PPO_lr{lr}_netarch{net_arch}_batchsize{batch_size}_timesteps{timesteps}"
                print(f"Training {model_name}...")
                policy_kwargs["net_arch"] = net_arch
                model = PPO("CnnPolicy", fs_vec_env, verbose=1, policy_kwargs=policy_kwargs,
                            learning_rate=lr, batch_size=batch_size, clip_range=0.1, ent_coef=0.01, 
                            gae_lambda=0.9, gamma=0.99, max_grad_norm=0.5, n_epochs=4, 
                            n_steps=128, vf_coef=0.5, device='cuda', tensorboard_log='runs/')
                model.learn(timesteps, tb_log_name=model_name, callback=HParamCallback())
                model.save(f"{model_name}.zip")
                del model

def efficient_net():
    efficient_net_weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
    efficient_net_model = torchvision.models.efficientnet_b1(weights=efficient_net_weights).to(device)
    efficient_net_model.classifier = nn.Identity()
    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = efficient_net_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]

    return efficient_net_model, efficient_net_weights, num_features

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
        if weights is None:
            print("Defaulting to using ResNet50 default weights.")
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        self._weights = weights
        self.preprocessing_function = weights.transforms()

    @model.setter
    def model(self, base_model):
        if base_model is None:
            print("Defaulting to using ResNet50 model.")
            base_model = torchvision.models.resnet50(self.weights)
            base_model.fc = nn.Identity()
        
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
        self.model = base_model
        if preprocessing_function:
            self.preprocessing_function = preprocessing_function

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.to(device)
        preprocessed_observations = self.preprocessing_function(observations)
        called = self.model(preprocessed_observations)
        return called


if __name__ == "__main__":
    """Testing with EfficientNet. If a model isn't supplied in the policy_kwargs, the ResNet50
        model is used as the default model."""

    efficient_net_weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
    efficient_net_model = torchvision.models.efficientnet_b1(weights=efficient_net_weights).to(device)
    efficient_net_model.classifier = nn.Identity()
    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = efficient_net_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]
    print(f"{num_features} output units at the end of the vision model")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=num_features, base_model=efficient_net_model, weights=efficient_net_weights,
                                       preprocessing_function=nn.Sequential(Grayscale(), efficient_net_weights.transforms())),
    )
    LOG_DIR = "./log"
    vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4)
    fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')

    timesteps = 30000
    lr_values = [2.5e-4]
    net_arch_values = [[128, 128]]
    batch_size_values = [128]

    hyperparam_search(lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps)
