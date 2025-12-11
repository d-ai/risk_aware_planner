# src/visualize.py

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # Force a stable backend on Windows
import matplotlib.pyplot as plt
from matplotlib import animation
import os


def animate_scene(ego_state, others_state, ego_traj, others_trajs=None,
                  step_dt=0.1, save_path=None, fps=10):
    """
    Robust animation of ego + other vehicles.
    - No blitting (stable on Windows)
    - No axis clearing during frames
    - Full blocking show() so window stays open
    """

    T = ego_traj.shape[0]
    lane_w = 3.5
    n_lanes = int(max([o[1] for o in others_state]) / lane_w) + 1 if len(others_state) else 3

    # ---------------------------------------------------
    # Build deterministic other trajectories if none given
    # ---------------------------------------------------
    if others_trajs is None:
        others_trajs = []
        for ox0, oy0, ov0 in others_state:
            traj = []
            x = ox0
            for _ in range(T):
                x += ov0 * step_dt
                traj.append([x, oy0, ov0])
            others_trajs.append(np.array(traj))

    # ---------------------------------------------------
    # Determine x-range for plotting
    # ---------------------------------------------------
    max_x = ego_traj[:, 0].max()
    if others_trajs:
        max_x = max(max_x, max([ot[:, 0].max() for ot in others_trajs]))
    xlim = (0, max_x + 10)

    # ---------------------------------------------------
    # Setup figure
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(xlim)
    ax.set_ylim(-1, n_lanes * lane_w + 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Autonomous Vehicle Trajectory Animation")

    # Draw lane lines
    for i in range(n_lanes + 1):
        ax.axhline(i * lane_w, color='gray', linewidth=0.8, alpha=0.6)

    # Draw faint trajectory outline
    ax.plot(ego_traj[:, 0], ego_traj[:, 1], color='red', alpha=0.25, linewidth=2)

    # ---------------------------------------------------
    # Create ego + other rectangular patches
    # ---------------------------------------------------
    ego_rect = plt.Rectangle((0, 0), 4.5, 2.0, color='tab:red', alpha=0.9)
    ax.add_patch(ego_rect)

    other_rects = []
    for _ in others_trajs:
        r = plt.Rectangle((0, 0), 4.5, 2.0, color='tab:gray', alpha=0.9)
        ax.add_patch(r)
        other_rects.append(r)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # ---------------------------------------------------
    # init() function — frame 0 setup
    # ---------------------------------------------------
    def init():
        ex0, ey0, _ = ego_traj[0]
        ego_rect.set_xy((ex0 - 4.5 / 2, ey0 - 2.0 / 2))

        for ot, r in zip(others_trajs, other_rects):
            ox0, oy0, _ = ot[0]
            r.set_xy((ox0 - 4.5 / 2, oy0 - 2.0 / 2))

        time_text.set_text("t = 0.00 s")
        return [ego_rect] + other_rects + [time_text]

    # ---------------------------------------------------
    # Frame update
    # ---------------------------------------------------
    def animate_frame(t_idx):
        ex, ey, _ = ego_traj[t_idx]
        ego_rect.set_xy((ex - 4.5 / 2, ey - 2.0 / 2))

        for ot, r in zip(others_trajs, other_rects):
            ox, oy, _ = ot[t_idx]
            r.set_xy((ox - 4.5 / 2, oy - 2.0 / 2))

        time_text.set_text(f"t = {t_idx * step_dt:.2f} s")
        return [ego_rect] + other_rects + [time_text]

    # ---------------------------------------------------
    # Build animation — no blit (stable)
    # ---------------------------------------------------
    anim = animation.FuncAnimation(
        fig,
        animate_frame,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=False
    )

    # ---------------------------------------------------
    # Show animation (blocking)
    # ---------------------------------------------------
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # Optional: Save animation (GIF or MP4)
    # ---------------------------------------------------
    if save_path is not None:
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='me'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Saved animation to {save_path}")
        except Exception:
            print("FFmpeg not available, falling back to imageio...")
            try:
                import imageio
                frames = []
                for i in range(T):
                    animate_frame(i)
                    fname = f"_temp_frame_{i:04d}.png"
                    fig.savefig(fname)
                    frames.append(imageio.imread(fname))
                    os.remove(fname)
                imageio.mimsave(save_path, frames, fps=fps)
                print(f"GIF saved to {save_path}")
            except Exception as e:
                print("Could not save animation. Install ffmpeg or imageio.")
                print("Error:", e)

    return anim
