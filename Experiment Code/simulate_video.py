# simulate_video.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML  # safe to keep if you run in notebook
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from lane_keeping_env import LaneKeepingEnv


def make_env(seed=42):
    def _init():
        env = LaneKeepingEnv()
        env.reset(seed=seed)
        return env
    return _init


def compute_metrics(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    center_rmse = float(np.sqrt(np.mean(np.square(ys))))
    in_lane = float(np.mean(np.abs(ys) <= 1.8))
    return {"center_rmse": center_rmse, "pct_in_lane": in_lane}


def main():
    # Rebuild VecNormalize + env
    eval_env = DummyVecEnv([make_env(42)])
    eval_env = VecNormalize.load("vecnormalize_train.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Load best model (or fallback)
    best_path = "./logs_dqn/best_model.zip"
    model_path = best_path if os.path.exists(best_path) else "dqn_lane_keep"
    model = DQN.load(model_path, env=eval_env)
    print(f"Loaded model from: {model_path}")

    # Rollout
    obs = eval_env.reset()
    done = np.array([False])
    xs, ys, deltas, psis = [], [], [], []

    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = eval_env.step(action)
        info = infos[0]
        xs.append(info["x"])
        ys.append(info["y"])
        deltas.append(info["delta"])
        psis.append(info["psi"])

    if len(xs) == 0:
        xs, ys, deltas, psis = [0.0], [0.0], [0.0], [0.0]

    metrics = compute_metrics(xs, ys)
    print("Metrics:", metrics)

    # Animation
    fig, (ax_traj, ax_steer) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [2.5, 1]},
        constrained_layout=True,
    )

    x_min, x_max = 0.0, max(xs) if len(xs) > 0 else 10.0
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(-2.2, 2.2)
    ax_traj.axhline(0.0, ls="--", label="Lane center")
    ax_traj.axhline(+1.8, ls=":", color="r", label="Lane boundary")
    ax_traj.axhline(-1.8, ls=":", color="r")
    traj_path, = ax_traj.plot([], [], lw=2, label="Path")
    car, = ax_traj.plot([], [], marker=(3, 0, 0), markersize=14,
                        linestyle="None", label="Car")
    ax_traj.legend(loc="upper right")

    def set_car_marker(line, psi_rad):
        line.set_marker((3, 0, np.degrees(psi_rad)))

    ax_steer.set_xlim(0, len(deltas))
    delta_lim = max(0.25, float(np.max(np.abs(deltas))))
    ax_steer.set_ylim(-delta_lim * 1.1, delta_lim * 1.1)
    steer_line, = ax_steer.plot([], [], lw=2)
    ax_steer.set_xlabel("Step")
    ax_steer.set_ylabel("δ [rad]")

    def init_anim():
        traj_path.set_data([], [])
        car.set_data([], [])
        set_car_marker(car, 0.0)
        steer_line.set_data([], [])
        return traj_path, car, steer_line

    def update(i):
        traj_path.set_data(xs[:i+1], ys[:i+1])
        car.set_data([xs[i]], [ys[i]])
        set_car_marker(car, psis[i])
        steer_line.set_data(range(i+1), deltas[:i+1])
        return traj_path, car, steer_line

    frames = len(xs)
    interval_ms = 40
    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        init_func=init_anim, interval=interval_ms, blit=True
    )

    out_name = "lane_keeping_split_demo.mp4"
    ani.save(out_name, writer="ffmpeg", fps=int(1000 / interval_ms))
    plt.close(fig)
    print(f"✅ Saved: {out_name}")

    # If running in Jupyter, you can display:
    try:
        display(HTML(f'<video controls width="720" src="{out_name}"></video>'))
    except Exception:
        pass


if __name__ == "__main__":
    main()
