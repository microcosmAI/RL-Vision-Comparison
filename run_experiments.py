import PretrainedCNNFeatureExtractor, TrainableCustomCNN, VariationalAutoencoder
from utils import hyperparam_search
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from HParamCallback import HParamCallback
from VariationalAutoencoder import VAETrainingCallback

selected_features_extractor_class, num_features, base_model, weights, preprocessing_function = None, None, None, None, None
default_callbacks = [HParamCallback()]

LOG_DIR = "./log"

timesteps = 3_000_000
lr_values = [2.5e-4]
net_arch_values = [[128, 128]]
batch_size_values = [128]

num_features_list = [256, 512, 1024]
environments = ["BreakoutNoFrameskip-v4", "BoxingNoFrameskip-v4"]

# train CNNs
selected_features_extractor_class = TrainableCustomCNN.TrainableCustomCNN
for num_features in num_features_list:
    policy_kwargs = dict(
        features_extractor_class=selected_features_extractor_class,
        features_extractor_kwargs=dict(features_dim=num_features, base_model=base_model, weights=weights,
                                       preprocessing_function=preprocessing_function)
        )
    for env_name in environments:
        vec_env = make_atari_env(env_name, n_envs=4)
        fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')
        hyperparam_search(fs_vec_env, lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps, model_prefix=f"{env_name}_CNN_{num_features}")

# train VAE
selected_features_extractor_class = VariationalAutoencoder.VariationalAutoencoderFeaturesExtractor
for num_features in num_features_list:
    policy_kwargs = dict(
        features_extractor_class=selected_features_extractor_class,
        features_extractor_kwargs=dict(features_dim=num_features, base_model=base_model, weights=weights,
                                       preprocessing_function=preprocessing_function)
        )
    vae_training_callback = VAETrainingCallback()
    vae_callbacks = default_callbacks + [vae_training_callback]

    for env_name in environments:
        vec_env = make_atari_env(env_name, n_envs=4)
        fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')
        hyperparam_search(fs_vec_env, lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps, model_prefix=f"{env_name}_VAE_{num_features}_", custom_callback=vae_callbacks)

# train using pretrained CNN
network = "squeezenet1"
selected_features_extractor_class = PretrainedCNNFeatureExtractor.PretrainedCNNFeatureExtractor
num_features = PretrainedCNNFeatureExtractor.NUM_FEATURES_MAP[network]
base_model, weights, _ = PretrainedCNNFeatureExtractor.NETWORK_VARS_MAP[network]()
preprocessing_function = PretrainedCNNFeatureExtractor.create_grayscale_preprocessing(weights)
policy_kwargs = dict(
    features_extractor_class=selected_features_extractor_class,
    features_extractor_kwargs=dict(features_dim=num_features, base_model=base_model, weights=weights,
                                   preprocessing_function=preprocessing_function)
    )
for env_name in environments:
    vec_env = make_atari_env(env_name, n_envs=4)
    fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')
    hyperparam_search(fs_vec_env, lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps, model_prefix=f"{env_name}_{network}_{num_features}_")
