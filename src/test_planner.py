import numpy as np
import random
from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all
from planner import select_best_trajectory, compute_score
from logger import make_run_id, append_summary_row, save_detail_json

DT = 0.1
HORIZON = 3.0
N_SAMPLES = 50  # reduce for quick runs


print(">>> Running planner test")

# -----------------------------------
# 1. Initialize environment
# -----------------------------------
env = HighwayEnv()
ego_state, others_state = env.reset()
print("Ego state:", ego_state)
print("Other vehicles:", len(others_state))

# -----------------------------------
# 2. Generate trajectories
# -----------------------------------
ego_trajs = generate_trajectories(
    ego_state, dt=DT, horizon=HORIZON
)
print("Generated", len(ego_trajs), "candidate trajectories")

# -----------------------------------
# 3. Monte Carlo risk estimation
# -----------------------------------
risks = estimate_risk_for_all(
    ego_trajs,
    others_initial_state=others_state,
    n_samples=N_SAMPLES,
    dt=DT,
    horizon=HORIZON
)

print("\n--- Risk Summary ---")
for i, r in enumerate(risks):
    print(
        f"Trajectory {i+1}: "
        f"P={r['collision_prob']:.3f}, "
        f"AvgDist={r['avg_min_distance']:.2f}, "
        f"WorstDist={r['worst_min_distance']:.2f}"
    )

# -----------------------------------
# 4. Compute planner scores
# -----------------------------------
scores = [
    compute_score(r, ego_trajs[i])
    for i, r in enumerate(risks)
]

print("\n--- Scores ---")
for i, sc in enumerate(scores):
    print(f"Trajectory {i+1}: Score={sc:.4f}")

best_idx, _ = select_best_trajectory(ego_trajs, risks)
print(f"\n>>> Best trajectory selected = {best_idx+1}")

# -----------------------------------
# 5. LOGGING SYSTEM
# -----------------------------------
run_id = make_run_id()
seed = random.randint(0, 10**9)
traj_types = ["keep", "brake", "lane_left", "lane_right"]

# Save detailed JSON info
detail = {
    "run_id": run_id,
    "seed": seed,
    "ego_state": ego_state.tolist(),
    "others_state": [list(o) for o in others_state],
    "risks": risks,
    "scores": scores,
    "best_idx": int(best_idx)
}
save_detail_json(run_id, detail)

# Append one summary row per trajectory
for i, r in enumerate(risks):
    row = {
        "run_id": run_id,
        "timestamp": run_id,
        "seed": seed,
        "ego_x0": float(ego_state[0]),
        "ego_y0": float(ego_state[1]),
        "ego_v0": float(ego_state[2]),
        "n_other": len(others_state),
        "horizon_s": HORIZON,
        "dt_s": DT,
        "traj_id": i,
        "traj_type": traj_types[i] if i < len(traj_types) else f"traj_{i}",
        "collision_prob": float(r["collision_prob"]),
        "avg_min_distance": float(r["avg_min_distance"]),
        "worst_min_distance": float(r["worst_min_distance"]),
        "score": float(scores[i]),
        "chosen": 1 if i == best_idx else 0,
        "notes": ""
    }
    append_summary_row(row)

print(f"\n✓ Log saved to logs/summary.csv")
print(f"✓ Detailed JSON saved to logs/detail_{run_id}.jsonl")
