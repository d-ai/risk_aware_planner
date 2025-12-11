from env import HighwayEnv

env = HighwayEnv()
state = env.reset()

print("Initial state:")
print(state)

for i in range(5):
    ego_state, others_state = env.step(ego_ax=0.0)
    print(f"Step {i+1} ego:", ego_state)
