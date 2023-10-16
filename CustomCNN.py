import torch as th
import torch.nn as nn
import torchvision
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

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
        print(f"Using {base_model._get_name()} as the base model.")

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
    def __init__(self, observation_space: spaces.Box, features_dim: int, base_model = None, weights = None):

        super().__init__(observation_space, features_dim)

        self.weights = weights
        self.model = base_model

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.to(device)
        preprocessed_observations = self.preprocessing_function(observations)
        called = self.model(preprocessed_observations)
        return called


if __name__ == "__main__":
    """Testing with ViT_L_16. If a model isn't supplied in the policy_kwargs, the ResNet50
        model is used as the default model."""
    """vit_l_16_weights = torchvision.models.ViT_L_16_Weights.DEFAULT
    vit_l_16_model = torchvision.models.vit_l_16(weights=vit_l_16_weights).to(device)
    vit_l_16_model.heads = nn.Identity()
    print(vit_l_16_model)

    rand_input = th.rand(1,3,224,224).to(device)
    with th.no_grad():
        output = vit_l_16_model(rand_input)
        output_dim = output.shape
    num_features = output_dim[1]
    print(f"{num_features} output units at the end of the vision model")"""

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=2048),
    )

    model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    timesteps = 400000
    model.learn(timesteps)
    model.save(f"{timesteps}_timesteps.zip")
