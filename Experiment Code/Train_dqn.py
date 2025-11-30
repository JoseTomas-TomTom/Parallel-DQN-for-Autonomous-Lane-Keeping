# train_dqn.py
import os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from lane_keeping_env import LaneKeepingEnv


def make_env(seed_offset=0):
    def _init():
        env = LaneKeepingEnv()
        env = Monitor(env)
        env.reset(seed=seed_offset)
        return env
    return _init


def lr_schedule(progress_remaining: float) -> float:
    # same as notebook: linear decay from 3e-4 → 0
    return 3e-4 * progress_remaining


def main():
    N_ENVS = 8

    # Training env (parallel)
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Eval env (dummy vec)
    eval_env = DummyVecEnv([make_env(10_000)])
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    eval_env.obs_rms = train_env.obs_rms

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_dqn/",
        log_path="./logs_dqn/",
        eval_freq=10_000 // N_ENVS,
        deterministic=True,
        render=False,
    )

    ckpt_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path="./logs_dqn/",
        name_prefix="dqn_lane",
    )

    policy_kwargs = dict(net_arch=[256, 256])

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=lr_schedule,
        gamma=0.99,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=8_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_dqn/",
    )

    TIMESTEPS = 400_000
    model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, ckpt_callback])

    # Save base model + VecNormalize stats
    model.save("dqn_lane_keep")
    train_env.save("vecnormalize_train.pkl")

    # Proper eval (same as notebook)
    eval_env = DummyVecEnv([make_env(42)])
    eval_env = VecNormalize.load("vecnormalize_train.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    best_path = "./logs_dqn/best_model.zip"
    model_path = best_path if os.path.exists(best_path) else "dqn_lane_keep"

    model = DQN.load(model_path, env=eval_env)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f"Eval mean reward: {mean_r:.3f} ± {std_r:.3f}")


if __name__ == "__main__":
    main()
