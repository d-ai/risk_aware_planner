import numpy as np

def trajectory_comfort_cost(ego_traj):
    """
    Simple comfort cost: sum of absolute acceleration (approximate).
    Lower is better.
    """
    vs = ego_traj[:, 2]
    accs = np.diff(vs)  # v[t+1] - v[t]
    return float(np.sum(np.abs(accs)))

def compute_score(risk_info, ego_traj, weights=None):
    """
    Compute a scalar score for a trajectory. Lower score = better.
    risk_info: dict with keys 'collision_prob', 'avg_min_distance'
    ego_traj: (T,3)
    weights: dict with keys 'p', 'd', 'c' (probability, distance, comfort)
    """
    if weights is None:
        weights = {'p': 1.0, 'd': 0.5, 'c': 0.1}

    p = risk_info.get('collision_prob', 0.0)
    avg_d = risk_info.get('avg_min_distance', 1e6)

    # We want lower score for lower p, larger avg_d, lower comfort cost
    # Normalize distance by a scale (e.g. 10 m)
    dist_term = 1.0 / max(avg_d, 1e-3)  # higher when avg_d is small
    comfort = trajectory_comfort_cost(ego_traj)

    score = weights['p'] * p + weights['d'] * dist_term + weights['c'] * (comfort / 10.0)
    return score

def select_best_trajectory(ego_trajs, risks, weights=None):
    """
    ego_trajs: list of trajectories (np arrays)
    risks: list of risk_info dicts, same order
    returns: index_of_best, scores (list)
    """
    scores = []
    for traj, risk in zip(ego_trajs, risks):
        sc = compute_score(risk, traj, weights)
        scores.append(sc)
    best_idx = int(np.argmin(scores))
    return best_idx, scores
