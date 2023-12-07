import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from CustomCNN import CustomCNN
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import TensorDataset, DataLoader

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

""" these classes were taken from https://avandekleut.github.io/vae/"""
class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        product = 1
        for dim in input_dims:
            product *= dim
        self.linear1 = nn.Linear(product, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = th.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = th.flatten(x, start_dim=1)
        print(f"flattened: {x}")
        x = F.relu(self.linear1(x))
        print(f"relued: {x}")
        mu =  self.linear2(x)
        print(f"mu: {mu}")
        sigma = th.exp(self.linear3(x))
        print(f"sigma: {sigma}")
        z = mu + sigma*self.N.sample(mu.shape)
        print(f"z: {z}")
        self.kl = (sigma**2 + mu**2 - th.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dims):
        self.n_channels, self.height, self.width = output_dims
        self.unrolled_dim = self.n_channels * self.height * self.width
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, self.unrolled_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = th.sigmoid(self.linear2(z))
        return z.reshape((-1, self.n_channels, self.height, self.width))

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, latent_dims)
        self.decoder = Decoder(latent_dims, input_dims)

    def forward(self, x):
        z = self.encoder(x)
        print(f"encoded: {z}")
        #decoded = self.decoder(z)
        #print(f"decoded: {decoded}")
        return self.decoder(z)


class VariationalAutoencoderFeaturesExtractor(CustomCNN):

    DEFAULT_LATENT_DIMS = 128

    @CustomCNN.model.setter
    def model(self, encoder):
        if encoder is None:
            print("Defaulting to basic trainable autoencoder.")
            print(f"obs space: {self._observation_space.shape}")
            self.n_input_channels, self.height, self.width = self._observation_space.shape
            self.variational_autoencoder = VariationalAutoencoder((self.n_input_channels, self.height, self.width), self._features_dim)

            encoder = self.variational_autoencoder.encoder

        self._model = encoder
        for param in self._model.parameters(): # freeze weights when not training the full autoencoder
            param.requires_grad = False

    #@CustomCNN.preprocessing_function.setter
    #def preprocessing_function(self, preprocessing_function):
        #if preprocessing_function is None:
            ## unroll based on dimensions

    @property
    def variational_autoencoder(self):
        return self._variational_autoencoder

    @variational_autoencoder.setter
    def variational_autoencoder(self, variational_autoencoder):
        self._variational_autoencoder = variational_autoencoder


    def __init__(self, observation_space: spaces.Box, features_dim: int, base_model = None, weights = None, preprocessing_function = None):

        super().__init__(observation_space, features_dim)
        self.training_buffer = []

def train(autoencoder, data, epochs=20):
    for param in autoencoder.parameters():
        param.requires_grad = True
    opt = th.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x in data:
            x = x[0]
            x = x.to(device, dtype=th.float32) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()

    for param in autoencoder.parameters():
        param.requires_grad = False
        print(param)
    return autoencoder


class VAETrainingCallback(BaseCallback):
    
    def _on_step(self):
        # is this the right way to access and train the encoder?
        features_extractor = self.model.policy.features_extractor
        if len(features_extractor.training_buffer) >= 32:
            print("Training...")
            tensor_x = th.from_numpy(np.array(features_extractor.training_buffer))
            my_dataset = TensorDataset(tensor_x)
            my_dataloader = DataLoader(my_dataset, batch_size=32)
            features_extractor.variational_autoencoder = train(features_extractor.variational_autoencoder, my_dataloader)
            features_extractor.training_buffer = []
        # is this the right way to get the observations? I tried get_images
        # and that gave me raw data, not the preprocessed images
        if self.num_timesteps % 10 == 0:
            image, _, _, _ = self.training_env.step_wait()
            num_concurrent = len(image)
            for i in range(num_concurrent):
                features_extractor.training_buffer.append(image[i])
