import numpy as np
from env import HighwayEnv
from trajectories import generate_trajectories
from collision import check_collision

# Create environment
env = HighwayEnv()
ego_state, others_state = env.reset()

# Create mock trajectories for others (they keep constant speed)
others_trajs = []
for ox, oy, ov in others_state:
    traj = []
    x = ox
    for _ in range(30):  # 3 seconds at dt=0.1
        x += ov * 0.1
        traj.append([x, oy, ov])
    others_trajs.append(np.array(traj))

# Generate ego trajectories
ego_trajs = generate_trajectories(ego_state)

# Check collisions for each trajectory
for i, ego_traj in enumerate(ego_trajs):
    collided, t, min_dist = check_collision(ego_traj, others_trajs)
    print(f"\nTrajectory {i+1}:")
    print("  Collided:", collided)
    print("  Collision timestep:", t)
    print("  Minimum distance:", round(min_dist, 2))
