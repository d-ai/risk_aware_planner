import numpy as np
from trajectories import generate_trajectories
from collision import check_collision

print(">>> Running forced collision test")

# Ego starts at x=0, center lane, 20 m/s
ego_state = np.array([0.0, 3.5, 20.0])

# Generate ego trajectories
ego_trajs = generate_trajectories(ego_state)

# Make ONE other car directly ahead in SAME lane, slower or stopped
T = ego_trajs[0].shape[0]
others_trajs = []

# Static car at x = 30 m, same lane (y=3.5), v = 0
other_traj = []
for t in range(T):
    other_traj.append([20.0, 3.5, 0.0])
others_trajs.append(np.array(other_traj))

# Now check collisions
for i, ego_traj in enumerate(ego_trajs):
    collided, t, min_dist = check_collision(ego_traj, others_trajs)
    print(f"\nTrajectory {i+1}:")
    print("  Collided:", collided)
    print("  Collision timestep:", t)
    print("  Minimum distance:", round(min_dist, 2))
