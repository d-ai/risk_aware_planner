# src/dashboard.py

import numpy as np
import matplotlib.pyplot as plt

from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all
from planner import select_best_trajectory, compute_score
from visualize import animate_scene  # animation comes from here

DT = 0.1
HORIZON = 3.0
N_SAMPLES = 50        # keep small for fast tests


def run_planning_pipeline():
    """Runs the full AV pipeline (env → trajectories → risk → plan)."""
    env = HighwayEnv()
    ego_state, others_state = env.reset()

    ego_trajs = generate_trajectories(ego_state, dt=DT, horizon=HORIZON)

    risks = estimate_risk_for_all(
        ego_trajs,
        others_initial_state=others_state,
        n_samples=N_SAMPLES,
        dt=DT,
        horizon=HORIZON,
    )

    scores = [compute_score(r, ego_trajs[i]) for i, r in enumerate(risks)]
    best_idx, _ = select_best_trajectory(ego_trajs, risks)

    return ego_state, others_state, ego_trajs, risks, scores, best_idx


def plot_dashboard(ego_state, others_state, ego_trajs, risks, scores, best_idx):
    """Creates a 4-panel dashboard of static plots."""

    best_traj = ego_trajs[best_idx]
    time = np.arange(best_traj.shape[0]) * DT

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # ---------------------------------------------
    # Panel 1 — Speed profile of chosen trajectory
    # ---------------------------------------------
    axs[0].plot(time, best_traj[:, 2], color='tab:red', linewidth=2)
    axs[0].set_title("Ego Speed Profile (Chosen Trajectory)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].grid(True)

    # ---------------------------------------------
    # Panel 2 — Risk summary
    # ---------------------------------------------
    cp = risks[best_idx]["collision_prob"]
    avgd = risks[best_idx]["avg_min_distance"]
    worst = risks[best_idx]["worst_min_distance"]

    axs[1].bar(["Collision Prob", "Avg Min Dist", "Worst Dist"],
               [cp, avgd, worst],
               color=["tab:red", "tab:blue", "tab:green"])

    axs[1].set_title("Risk Indicators (Chosen Trajectory)")
    axs[1].set_ylabel("Value")

    # ---------------------------------------------
    # Panel 3 — Planner scores
    # ---------------------------------------------
    bar_colors = ["tab:gray"] * len(scores)
    bar_colors[best_idx] = "tab:red"

    axs[2].bar([f"T{i+1}" for i in range(len(scores))],
               scores,
               color=bar_colors)

    axs[2].set_title("Planner Scores Comparison")
    axs[2].set_ylabel("Score")
    axs[2].set_xlabel("Trajectory ID")

    # ---------------------------------------------
    # Panel 4 — Static top-down snapshot
    # ---------------------------------------------
    axs[3].set_title("Top-Down Snapshot (Start State)")
    axs[3].set_xlabel("x (m)")
    axs[3].set_ylabel("y (lane)")

    # Draw lanes
    lane_w = 3.5
    n_lanes = 3
    for i in range(n_lanes + 1):
        axs[3].axhline(i * lane_w, color="gray", linewidth=0.7, alpha=0.5)

    # Ego start
    ex0, ey0, _ = ego_state
    axs[3].scatter([ex0], [ey0], color="tab:red", s=60, label="Ego")

    # Other vehicles
    for (ox, oy, _) in others_state:
        axs[3].scatter([ox], [oy], color="tab:gray", s=40)

    axs[3].legend()

    plt.tight_layout()
    plt.show()


def run_dashboard():
    # ------------------------------------------------
    # Compute all planning logic first
    # ------------------------------------------------
    ego_state, others_state, ego_trajs, risks, scores, best_idx = run_planning_pipeline()

    print("Scores:", scores)
    print("Chosen trajectory index:", best_idx)

    # ------------------------------------------------
    # Open animation window (separate)
    # ------------------------------------------------
    animate_scene(
        ego_state,
        others_state,
        ego_trajs[best_idx],
        step_dt=DT
    )

    # ------------------------------------------------
    # Open the dashboard window
    # ------------------------------------------------
    plot_dashboard(ego_state, others_state, ego_trajs, risks, scores, best_idx)


if __name__ == "__main__":
    run_dashboard()
