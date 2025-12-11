import numpy as np
from trajectories import generate_trajectories

print(">> test_trajectories.py is running")

ego_state = np.array([0.0, 3.5, 20.0])  # x, y, vel

trajs = generate_trajectories(ego_state)

print("Number of trajectories:", len(trajs))

for i, traj in enumerate(trajs):
    print(f"\nTrajectory {i+1} (first 5 points):")
    print(traj[:5])
