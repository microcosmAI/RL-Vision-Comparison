import PretrainedCNNFeatureExtractor, TrainableCustomCNN, VariationalAutoencoder
import argparse
from utils import hyperparam_search
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from HParamCallback import HParamCallback
from VariationalAutoencoder import VAETrainingCallback

def create_parser():
    parser = argparse.ArgumentParser(
                    prog='PPO Trainer with Custom Feature Extractor',
                    description='Trains PPO model with a custom feature extractor. \
                    This can be a pretrained CNN or an untrained CNN that gets trained\
                    alongside the RL model.')
    parser.add_argument('-t', '--timesteps', default=500_000, type=int)
    subparser = parser.add_subparsers(dest='feature_extractor')
    subparser.default = 'cnn' 
    pretrained_cnn_subparser = subparser.add_parser('pretrained_cnn')
    pretrained_cnn_subparser.add_argument('network', nargs='?', default='resnet50', choices=['resnet50', 'efficientnet'])
    cnn_subparser = subparser.add_parser('cnn')
    cnn_subparser.add_argument('-nf', '--num_features', nargs='?', default=512, type=int)
    vae_subparser = subparser.add_parser('vae')
    return parser

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    if not hasattr(args, 'num_features'): # not sure why default in subparser doesn't work when cnn isn't supplied
        args.num_features = 512

    selected_features_extractor_class, num_features, base_model, weights, preprocessing_function = None, None, None, None, None
    callbacks = [HParamCallback()]

    if args.feature_extractor == 'cnn':
        selected_features_extractor_class = TrainableCustomCNN.TrainableCustomCNN
        num_features = args.num_features
    elif args.feature_extractor == 'pretrained_cnn':
        selected_features_extractor_class = PretrainedCNNFeatureExtractor.PretrainedCNNFeatureExtractor
        num_features = PretrainedCNNFeatureExtractor.NUM_FEATURES_MAP[args.network]
        base_model, weights, _ = PretrainedCNNFeatureExtractor.NETWORK_VARS_MAP[args.network]()
        preprocessing_function = PretrainedCNNFeatureExtractor.create_grayscale_preprocessing(weights)
    elif args.feature_extractor == 'vae':
        selected_features_extractor_class = VariationalAutoencoder.VariationalAutoencoderFeaturesExtractor
        num_features = args.num_features
        vae_training_callback = VAETrainingCallback()
        callbacks.append(vae_training_callback)
    else:
        print("Unknown feature extractor type.")


    policy_kwargs = dict(
        features_extractor_class=selected_features_extractor_class,
        features_extractor_kwargs=dict(features_dim=num_features, base_model=base_model, weights=weights,
                                       preprocessing_function=preprocessing_function)
    )
    LOG_DIR = "./log"
    vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4)
    fs_vec_env = VecFrameStack(vec_env, 4, channels_order='first')

    timesteps = args.timesteps
    lr_values = [2.5e-4]
    net_arch_values = [[128, 128]]
    batch_size_values = [128]

    hyperparam_search(fs_vec_env, lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps, callbacks)
