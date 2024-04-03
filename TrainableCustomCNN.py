import torch as th
import torch.nn as nn
import torchvision
from CustomCNN import CustomCNN

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

            full_model = nn.Sequential(base_model, nn.Linear(n_flatten, self._features_dim), nn.ReLU())

        self._model = full_model
