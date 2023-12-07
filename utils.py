from HParamCallback import HParamCallback
from stable_baselines3 import PPO

def hyperparam_search(env, lr_values, batch_size_values, net_arch_values, policy_kwargs, timesteps, custom_callback=HParamCallback()):
    for lr in lr_values:
        for net_arch in net_arch_values:
            for batch_size in batch_size_values:
                model_name = f"PPO_lr{lr}_netarch{net_arch}_batchsize{batch_size}_timesteps{timesteps}"
                print(f"Training {model_name}...")
                policy_kwargs["net_arch"] = net_arch
                model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                            learning_rate=lr, batch_size=batch_size, clip_range=0.1, ent_coef=0.01, 
                            gae_lambda=0.9, gamma=0.99, max_grad_norm=0.5, n_epochs=4, 
                            n_steps=128, vf_coef=0.5, device='cuda', tensorboard_log='runs/')
                model.learn(timesteps, tb_log_name=model_name, callback=custom_callback)
                model.save(f"{model_name}.zip")
                del model

