# src/run_and_animate.py
import numpy as np
from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all
from planner import select_best_trajectory
from visualize import animate_scene
import os

DT = 0.1
HORIZON = 3.0
N_SAMPLES = 80    # reduce for demo; increase for more accurate risk
FPS = 10

def run_and_animate(save_gif=False, gif_path="results/run_anim.gif"):
    env = HighwayEnv()
    ego_state, others_state = env.reset()

    # generate candidates and estimate risk
    ego_trajs = generate_trajectories(ego_state, dt=DT, horizon=HORIZON)
    risks = estimate_risk_for_all(ego_trajs, others_initial_state=others_state,
                                 n_samples=N_SAMPLES, dt=DT, horizon=HORIZON)

    # select best
    best_idx, scores = select_best_trajectory(ego_trajs, risks)
    print("Scores:", ["{:.4f}".format(s) for s in scores])
    print("Chosen action index:", best_idx)

    # For animation we will animate the chosen ego trajectory deterministically
    chosen_ego_traj = ego_trajs[best_idx]

    # Build deterministic other vehicle rollouts for visualization (constant speed)
    T = chosen_ego_traj.shape[0]
    others_trajs = []
    for ox0, oy0, ov0 in others_state:
        traj = []
        x = ox0
        for _ in range(T):
            x += ov0 * DT
            traj.append([x, oy0, ov0])
        others_trajs.append(np.array(traj))

    # ensure results directory
    os.makedirs("results", exist_ok=True)
    save_path = gif_path if save_gif else None

    anim = animate_scene(ego_state, others_state, chosen_ego_traj, others_trajs,
                         step_dt=DT, save_path=save_path, fps=FPS)
    return anim

if __name__ == "__main__":
    run_and_animate(save_gif=False)
