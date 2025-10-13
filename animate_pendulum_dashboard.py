import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

def animate_pendulum_dashboard(
    t, dt,
    states_truth,     # (N,2): [theta, theta_dot]
    states_est,       # (N,2)
    measurements,     # (N,)  position measurements (rad)
    P_hist,           # (N,2,2)
    control_hist,     # (N,)
    L,                # pendulum length
):
    # Pendulum coordinates
    x_pend = L * np.cos(states_truth[:, 0])   # sin for x
    y_pend = L * np.sin(states_truth[:, 0])  # -cos for y (standard orientation)

    # Sliding window length for time histories
    window = 3  # seconds

    # === Figure layout ===
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])

    # Left: Pendulum plot
    ax_pend = fig.add_subplot(gs[:, 0])
    ax_pend.set_xlim(-L*1.2, L*1.2)
    ax_pend.set_ylim(-L*1.2, L*1.2)
    ax_pend.set_aspect('equal', 'box')
    ax_pend.grid(True)
    ax_pend.set_title("Pendulum Animation")

    rod_line, = ax_pend.plot([], [], 'o-', lw=2, color='C0')
    trace_line, = ax_pend.plot([], [], '-', lw=2, color='r', alpha=0.4)

    # Keep trace index separate
    trace_x, trace_y = [], []

    # UI text
    UI_text = ax_pend.text(
        0.02, 0.95, '', transform=ax_pend.transAxes,
        ha='left', va='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # === Right: Time histories ===
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_ctrl = fig.add_subplot(gs[2, 1])

    for ax in [ax_pos, ax_vel, ax_ctrl]:
        ax.grid(True)

    # Position
    pos_meas_scatter = ax_pos.scatter([], [], s=5, c='blue', alpha=0.3, label='Meas')
    pos_est_line, = ax_pos.plot([], [], 'r', label='Est')
    pos_truth_line, = ax_pos.plot([], [], 'gray', label='Truth')
    pos_sigma_up, = ax_pos.plot([], [], 'C1--', lw=1, label='±3 sigma')
    pos_sigma_dn, = ax_pos.plot([], [], 'C1--', lw=1)
    ax_pos.set_ylabel("θ [rad]")
    ax_pos.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )
    ax_pos.set_title("State Estimates")

    # Velocity
    vel_est_line, = ax_vel.plot([], [], 'r', label='Est')
    vel_truth_line, = ax_vel.plot([], [], 'gray', label='Truth')
    vel_sigma_up, = ax_vel.plot([], [], 'C1--', lw=1, label='±3 sigma')
    vel_sigma_dn, = ax_vel.plot([], [], 'C1--', lw=1)
    ax_vel.set_ylabel("θ̇ [rad/s]")
    ax_vel.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    # Control input
    ctrl_line, = ax_ctrl.plot([], [], 'b', label='Control Input')
    ax_ctrl.set_ylabel("τ [Nm]")
    ax_ctrl.set_xlabel("Time [s]")
    ax_ctrl.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    # === Initialization ===
    def init():
        rod_line.set_data([], [])
        trace_line.set_data([], [])
        trace_x.clear()
        trace_y.clear()

        pos_meas_scatter.set_offsets(np.empty((0, 2)))
        for line in [pos_est_line, pos_truth_line, pos_sigma_up, pos_sigma_dn,
                    vel_est_line, vel_truth_line, vel_sigma_up, vel_sigma_dn,
                    ctrl_line]:
            line.set_data([], [])

        UI_text.set_text('')
        return (rod_line, trace_line,
                pos_est_line, pos_truth_line, pos_sigma_up, pos_sigma_dn,
                vel_est_line, vel_truth_line, vel_sigma_up, vel_sigma_dn,
                ctrl_line, pos_meas_scatter, UI_text)

    # === Update each frame ===
    def update(frame):
        t_now = t[frame]

        # Pendulum position
        x = x_pend[frame]
        y = y_pend[frame]
        rod_line.set_data([0, x], [0, y])

        trace_x.append(x)
        trace_y.append(y)
        trace_line.set_data(trace_x, trace_y)

        # Update text
        UI_text.set_text(
            f"t = {t_now:.2f} s\n"
            f"θ = {states_truth[frame,0]*180/np.pi:.1f}°\n"
            f"τ = {control_hist[frame]:.2f} Nm"
        )

        # Grow → Scroll mode for time histories
        if t_now < window:
            t_min = 0
        else:
            t_min = t_now - window
        mask = (t >= t_min) & (t <= t_now)
        tt = t[mask]

        # Position subplot
        pos_meas_scatter.set_offsets(np.column_stack([t[mask], measurements[mask]]))
        pos_est_line.set_data(tt, states_est[mask, 0])
        pos_truth_line.set_data(tt, states_truth[mask, 0])
        pos_sigma_up.set_data(tt, states_est[mask, 0] + 3*np.sqrt(P_hist[mask, 0, 0]))
        pos_sigma_dn.set_data(tt, states_est[mask, 0] - 3*np.sqrt(P_hist[mask, 0, 0]))
        ax_pos.set_xlim(t_min, max(t_now, 1e-6))
        ax_pos.relim()
        ax_pos.autoscale_view(scalex=False, scaley=True)

        # Velocity subplot
        vel_est_line.set_data(tt, states_est[mask, 1])
        vel_truth_line.set_data(tt, states_truth[mask, 1])
        vel_sigma_up.set_data(tt, states_est[mask, 1] + 3*np.sqrt(P_hist[mask, 1, 1]))
        vel_sigma_dn.set_data(tt, states_est[mask, 1] - 3*np.sqrt(P_hist[mask, 1, 1]))
        ax_vel.set_xlim(t_min, max(t_now, 1e-6))
        ax_vel.relim()
        ax_vel.autoscale_view(scalex=False, scaley=True)

        # Control subplot
        ctrl_line.set_data(tt, control_hist[mask])
        ax_ctrl.set_xlim(t_min, max(t_now, 1e-6))
        ax_ctrl.relim()
        ax_ctrl.autoscale_view(scalex=False, scaley=True)

        return (rod_line, trace_line,
                pos_est_line, pos_truth_line, pos_sigma_up, pos_sigma_dn,
                vel_est_line, vel_truth_line, vel_sigma_up, vel_sigma_dn,
                ctrl_line, pos_meas_scatter, UI_text)

    # === Animation ===
    fps = int(1/dt) // 60  # adjust playback speed
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(t), fps),
        init_func=init,
        blit=False,
        interval=dt*1000*fps,
        repeat=True
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()