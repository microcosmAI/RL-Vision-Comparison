# From Pixels to Actions: Comparing Image Feature Extraction Techniques for Reinforcement Learning

This repository contains the code and experiments for the cognitive science student jounral paper titled "**From Pixels to Actions: Comparing Image Feature Extraction Techniques for Reinforcement Learning**". The study explores the impact of different image feature extraction techniques on reinforcement learning (RL) agents' performance across various visually complex Atari game environments.

## Table of Contents
1. [Abstract](#abstract)
2. [Folder Structure](#folder-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Results](#results)
5. [References](#references)

## Abstract
This study compares three types of feature extractors for reinforcement learning (RL) tasks involving visual data: 
1. Custom two-layer Convolutional Neural Network (CNN)
2. Pre-trained CNNs (such as ResNet and EfficientNet)
3. Variational Autoencoder (VAE).

Our experiments reveal that custom CNNs can outperform more complex pre-trained models, offering a balance of simplicity and efficiency. Pre-trained models show potential in specific configurations, while VAEs present mixed results. The findings emphasize the importance of choosing the right feature extraction method to improve RL agents' learning efficiency in dynamic environments.

## Folder Structure

The repository is organized as follows:

```plaintext
.
├── autoencoder                 # Code related to the Variational Autoencoder (VAE) experiments
├── plots                       # Generated plots and visualizations
├── plots_csvs                  # CSV files of experiment results
├── runs                        # Directory for saving model runs and logs
├── HParamCallback.py           # Hyperparameter callback functions used in experiments
├── plot.ipynb                  # Notebook for generating plots from results
├── PretrainedCNNFeatureExtractor.py  # Script for using pre-trained CNNs as feature extractors
├── run_experiments.py          # Main script to run different RL experiments
├── train_models_stable_baselines.py  # Script to train RL models using Stable Baselines3
├── TrainableCustomCNN.py       # Custom CNN architecture for feature extraction
├── utils.py                    # Utility functions for data preprocessing and model management
├── VariationalAutoencoder.py   # Variational Autoencoder (VAE) implementation
```

## Results

Results from our experiments can be found in the `plots/` folder. Example plots include the comparison of RL agent performance when using different feature extraction methods across various Atari environments.

Some of the key findings include:
- Custom CNNs consistently outperform more complex pre-trained models in terms of final performance across several environments.
- Pre-trained models, such as ResNet50, can struggle when the visual inputs are out-of-distribution compared to their original ImageNet training data.
- VAEs provide a compact representation of the visual input but can sometimes omit critical details, such as small objects important for game dynamics.

## References

- **Stable Baselines3**: [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Gymnasium**: [https://gymnasium.farama.org](https://gymnasium.farama.org)

For more details about the experiments and methodologies used in this project, please refer to the paper: *From Pixels to Actions: Comparing Image Feature Extraction Techniques for Reinforcement Learning*.