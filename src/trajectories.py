import numpy as np

def generate_trajectories(ego_state, lane_width=3.5, dt=0.1, horizon=3.0):
    
    """Returns a list of candidate trajectories for the ego vehicle.
    Each trajectory is a sequence of [x, y, v] states over time."""
    x0, y0, v0 = ego_state
    T = int(horizon / dt)

    return [
        keep_lane(x0, y0, v0, dt, T),
        brake(x0, y0, v0, dt, T),
        lane_change(x0, y0, v0, dt, T, direction="left", lane_width=lane_width),
        lane_change(x0, y0, v0, dt, T, direction="right", lane_width=lane_width)
    ]


def keep_lane(x0, y0, v0, dt, T):
    traj = []
    x, y, v = x0, y0, v0

    for _ in range(T):
        x += v * dt
        traj.append([x, y, v])

    return np.array(traj)


def brake(x0, y0, v0, dt, T, decel=-2.0):
    traj = []
    x, y, v = x0, y0, v0

    for _ in range(T):
        v = max(0.0, v + decel * dt)
        x += v * dt
        traj.append([x, y, v])

    return np.array(traj)


def lane_change(x0, y0, v0, dt, T, direction, lane_width):
    traj = []
    x, y, v = x0, y0, v0

    if direction == "left":
        target_y = y + lane_width
    else:
        target_y = y - lane_width

    for t in range(T):
        alpha = t / (T - 1)
        y_t = y + (target_y - y) * np.sin(alpha * np.pi / 2)
        x += v * dt
        traj.append([x, y_t, v])

    return np.array(traj)
