import numpy as np
from collision import check_collision


def simulate_others_rollout(others_initial_state, dt=0.1, horizon=3.0, rng=None):
    """
    Simulate one possible future for all other vehicles with random accelerations.
    others_initial_state: (N, 3) array of [x, y, v] for each vehicle at t=0
    Returns:
        others_trajs: list of (T, 3) arrays for each other vehicle
    """
    if rng is None:
        rng = np.random.default_rng()

    T = int(horizon / dt)
    others_trajs = []

    for ox0, oy0, ov0 in others_initial_state:
        x = ox0
        y = oy0
        v = ov0
        traj = []

        for _ in range(T):
            # Sample random acceleration: bias toward maintaining speed
            # values in m/s^2
            ax = rng.choice([-2.0, 0.0, 1.0], p=[0.2, 0.6, 0.2])

            # Update velocity and position
            v = max(0.0, v + ax * dt)
            x += v * dt

            traj.append([x, y, v])

        others_trajs.append(np.array(traj))

    return others_trajs


def estimate_risk_for_trajectory(ego_traj, others_initial_state,
                                 n_samples=100, dt=0.1, horizon=3.0,
                                 rng=None):
    """
    Estimate collision risk for a single ego trajectory using Monte Carlo simulation.

    Returns:
        risk_info: dict with keys:
            'collision_prob': float
            'avg_min_distance': float
            'worst_min_distance': float
    """
    if rng is None:
        rng = np.random.default_rng()

    collisions = 0
    min_dists = []

    for k in range(n_samples):
        # Sample one future for all other vehicles
        others_trajs = simulate_others_rollout(
            others_initial_state,
            dt=dt,
            horizon=horizon,
            rng=rng
        )

        collided, t_coll, min_dist = check_collision(ego_traj, others_trajs)
        if collided:
            collisions += 1

        min_dists.append(min_dist)

    collision_prob = collisions / n_samples
    avg_min_distance = float(np.mean(min_dists)) if min_dists else float("inf")
    worst_min_distance = float(np.min(min_dists)) if min_dists else float("inf")

    return {
        "collision_prob": collision_prob,
        "avg_min_distance": avg_min_distance,
        "worst_min_distance": worst_min_distance,
    }


def estimate_risk_for_all(ego_trajs, others_initial_state,
                          n_samples=100, dt=0.1, horizon=3.0,
                          rng=None):
    """
    Compute risk estimates for a list of ego trajectories.
    Returns:
        list of risk_info dicts (one per trajectory)
    """
    if rng is None:
        rng = np.random.default_rng()

    risks = []
    for ego_traj in ego_trajs:
        info = estimate_risk_for_trajectory(
            ego_traj,
            others_initial_state,
            n_samples=n_samples,
            dt=dt,
            horizon=horizon,
            rng=rng
        )
        risks.append(info)
    return risks
