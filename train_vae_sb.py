import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
from PIL import Image
import torch as th

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 16 * 16, 100)  # For mean
        self.fc_logvar = nn.Linear(128 * 16 * 16, 100)  # For log variance

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def encode(self, x):
        mu, logvar = self.forward(x)
        return reparameterize(mu, logvar)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(100, 128 * 16 * 16)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, 128, 16, 16)  # Reshape to match the encoder's last conv layer output shape
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Use sigmoid for the last layer if images are normalized between [0,1]
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class EncoderPreprocessedObservation(gym.ObservationWrapper):
    def __init__(self, env, encoder):
        super(EncoderPreprocessedObservation, self).__init__(env)
        self.encoder = encoder
        # You need to adjust this shape to match the output of your encoder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(200,), dtype=np.float32)
        
    def observation(self, observation):
        # Assuming observation is numpy array, convert it to PyTorch tensor
        observation = np.array([np.array(Image.fromarray(obs).resize((128, 128))) for obs in  observation])
        observation = torch.tensor(observation).permute(0, 3, 1, 2).float() / 255.0
        # observation = torch.tensor(observation).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            encoded_obs = self.encoder.encode(observation)
        # Flatten the encoded observation to fit the observation space
        return encoded_obs.cpu().numpy().flatten()

autoencoder = VAE()
# Load your pretrained model
autoencoder.load_state_dict(torch.load("models/vae_100.pth"))
autoencoder.eval()
pretrained_encoder = autoencoder.encoder

# Load your environment
env_id = "BoxingNoFrameskip-v4"
env = gym.make(env_id, max_episode_steps=1000)
env = FrameStack(env, num_stack=2)
env = EncoderPreprocessedObservation(env, pretrained_encoder)

# Wrap your environment with the EncoderPreprocessedObservation
# Note: You must load or define your pretrained `encoder` before this step.

# Because we are using a custom observation space, we might need a custom policy or feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Define your custom CNN based on the encoded features' shape
        self.cnn = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

# Use the custom feature extractor in PPO's policy_kwargs
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=96),
)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

# Initialize PPO with the custom environment
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_BoxingNoFrameskip_vae/", device="mps", learning_rate=0.0001, batch_size=256)

# Train the model
model.learn(total_timesteps=4000000, tb_log_name="first_run")

