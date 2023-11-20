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
from CustomCNN import CustomCNN

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


class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.expand(img.shape[0], 3, *img.shape[2:])

class TrainableCustomCNN(CustomCNN):

    @CustomCNN.weights.setter
    def weights(self, weights):
        if weights is None:
            print("weights is None. Currently unsupported behavior.")
        self._weights = weights

    @CustomCNN.model.setter
    def model(self, base_model):
        if base_model is None:
            print("Defaulting to basic trainable CNN.")
            print(f"obs space: {self._observation_space.shape}")
            n_input_channels = self._observation_space.shape[0]
            base_model = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = base_model(
                    th.as_tensor(self._observation_space.sample()[None]).float()
                ).shape[1]

            full_model = nn.Sequential(base_model, nn.Linear(n_flatten, self.features_dim), nn.ReLU())

        self._model = full_model


if __name__ == "__main__":

    num_features = 512

    policy_kwargs = dict(
        features_extractor_class=TrainableCustomCNN,
        features_extractor_kwargs=dict(features_dim=num_features),
    )
    LOG_DIR = "./log"
    vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4)
    fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')

    timesteps = 30000
    lr_values = [2.5e-4]
    net_arch_values = [[128, 128]]
    batch_size_values = [128]

    hyperparam_search(lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps)
