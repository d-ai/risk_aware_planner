import numpy as np

class Vehicle:
    def __init__(self, x, y, vx, width=2.0, length=4.5):
        self.x = x
        self.y = y
        self.vx = vx
        self.width = width
        self.length = length

    def step(self, dt, ax=0.0):
        """Move the vehicle based on acceleration ax."""
        self.vx += ax * dt
        self.x += self.vx * dt


class HighwayEnv:
    def __init__(self, n_lanes=3, lane_width=3.5):
        self.n_lanes = n_lanes
        self.lane_width = lane_width
        self.ego = None
        self.others = []

    def reset(self):
        """Initialize ego vehicle and some random traffic."""
        # Ego starts in center lane
        ego_lane = self.n_lanes // 2
        ego_y = ego_lane * self.lane_width
        self.ego = Vehicle(x=0.0, y=ego_y, vx=20.0)

        # Random other vehicles
        self.others = []
        for lane in range(self.n_lanes):
            for i in range(2):  # 2 vehicles per lane
                x = np.random.uniform(20, 80)
                vx = np.random.uniform(15, 25)
                y = lane * self.lane_width
                self.others.append(Vehicle(x=x, y=y, vx=vx))

        return self.get_state()

    def get_state(self):
        """Return the state of ego + others as arrays."""
        ego_state = np.array([self.ego.x, self.ego.y, self.ego.vx])
        others_state = np.array([[v.x, v.y, v.vx] for v in self.others])
        return ego_state, others_state

    def step(self, ego_ax, dt=0.1):
        """Step ego + others forward."""
        self.ego.step(dt, ego_ax)
        for v in self.others:
            v.step(dt, ax=0.0)  # other cars keep constant speed
        return self.get_state()
