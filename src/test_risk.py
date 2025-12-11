import numpy as np
from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all

DT = 0.1
HORIZON = 3.0
N_SAMPLES = 100

print(">>> Running Monte Carlo risk estimation test")

# 1. Create environment and get initial state
env = HighwayEnv()
ego_state, others_state = env.reset()

print("Initial ego state:", ego_state)
print("Number of other vehicles:", len(others_state))

# 2. Generate candidate ego trajectories
ego_trajs = generate_trajectories(ego_state, dt=DT, horizon=HORIZON)
print("Number of ego trajectories:", len(ego_trajs))

# 3. Estimate risk for each trajectory
risks = estimate_risk_for_all(
    ego_trajs,
    others_initial_state=others_state,
    n_samples=N_SAMPLES,
    dt=DT,
    horizon=HORIZON
)

# 4. Print results
for i, info in enumerate(risks):
    print(f"\nTrajectory {i+1}:")
    print(f"  Collision probability: {info['collision_prob']:.3f}")
    print(f"  Avg min distance: {info['avg_min_distance']:.2f} m")
    print(f"  Worst min distance: {info['worst_min_distance']:.2f} m")
