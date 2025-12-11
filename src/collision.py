import numpy as np

EGO_WIDTH = 2.0
EGO_LENGTH = 4.5

OTHER_WIDTH = 2.0
OTHER_LENGTH = 4.5


def check_collision(ego_traj, others_trajs):
    """
    ego_traj: (T, 3) array for ego [x, y, v]
    others_trajs: list of (T, 3) arrays for other cars

    Returns:
        collided: bool
        collision_t: timestep of the first collision or None
        min_distance: float (minimum center-to-center distance)
    """
    T = ego_traj.shape[0]

    min_distance = float("inf")
    collision_t = None

    for t in range(T):
        ex, ey, _ = ego_traj[t]

        for other in others_trajs:
            ox, oy, _ = other[t]

            # Compute center distance
            dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
            min_distance = min(min_distance, dist)

            # AABB overlap check
            if aabb_overlap(ex, ey, ox, oy):
                return True, t, min_distance

    return False, None, min_distance


def aabb_overlap(ex, ey, ox, oy):
    """Check bounding-box overlap between ego and another vehicle."""
    half_w_e, half_l_e = EGO_WIDTH / 2, EGO_LENGTH / 2
    half_w_o, half_l_o = OTHER_WIDTH / 2, OTHER_LENGTH / 2

    # Check for overlap in x direction
    x_overlap = abs(ex - ox) < (half_l_e + half_l_o)

    # Check for overlap in y direction
    y_overlap = abs(ey - oy) < (half_w_e + half_w_o)

    return x_overlap and y_overlap
