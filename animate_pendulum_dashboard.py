import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

R2D = 180 / np.pi

def animate_pendulum_dashboard(
    t, dt,
    states_truth,     # (N,2): [theta, theta_dot]
    states_est,       # (N,2)
    measurements,     # (N,)  position measurements (rad)
    P_hist,           # (N,2,2)
    control_hist,     # (N,)
    L,                # pendulum length
    window=3.0,       # seconds visible
    trace_len=0.75    # seconds of pendulum tail to keep
):
    N = len(t)
    skip = max(1, int(1/dt) // 60)       # playback throttle
    frames = range(0, N, skip)
    max_pts = int(window / dt)
    max_tail = int(trace_len / dt)

    # ---- Orientation you asked for: x= L*cos(theta), y = L*sin(theta)
    x_pend = L * np.cos(states_truth[:, 0])
    y_pend = L * np.sin(states_truth[:, 0])

    # Precompute ±3 sigma envelopes (fast per-frame)
    pos_3s = 3.0 * np.sqrt(P_hist[:, 0, 0])
    vel_3s = 3.0 * np.sqrt(P_hist[:, 1, 1])

    # Reasonable fixed y-limits (avoids autoscale cost each frame)
    pos_min = np.min([states_truth[:,0]-pos_3s, states_truth[:,0]+pos_3s, measurements])    * 1.2 * R2D
    pos_max = np.max([states_truth[:,0]-pos_3s, states_truth[:,0]+pos_3s, measurements])    * 1.2 * R2D
    vel_min = np.min([states_truth[:,1]-vel_3s, states_truth[:,1]+vel_3s, states_est[:,1]]) * 1.2 * R2D
    vel_max = np.max([states_truth[:,1]-vel_3s, states_truth[:,1]+vel_3s, states_est[:,1]]) * 1.2 * R2D
    tau_min = np.min(control_hist) * 1.2
    tau_max = np.max(control_hist) * 1.2

    # ---- Figure layout
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])

    # Left: pendulum
    ax_pend = fig.add_subplot(gs[:, 0])
    ax_pend.set_xlim(-L*1.2, L*1.2)
    ax_pend.set_ylim(-L*1.2, L*1.2)
    ax_pend.set_aspect('equal', 'box')
    ax_pend.grid(True)
    ax_pend.set_title("Pendulum")

    rod_line, = ax_pend.plot([], [], 'o-', lw=2, color='C0')
    trace_line, = ax_pend.plot([], [], '-', lw=2, color='r', alpha=0.4)
    trace_x, trace_y = [], []

    UI_text = ax_pend.text(
        0.02, 0.95, '', transform=ax_pend.transAxes,
        ha='left', va='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Right: time histories (absolute time on x-axis)
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_ctrl = fig.add_subplot(gs[2, 1])

    for ax in (ax_pos, ax_vel, ax_ctrl):
        ax.grid(True)

    # Position
    meas_line,     = ax_pos.plot([], [], '.', ms=3, alpha=0.5, c='C0', label='Meas')
    pos_est_line,  = ax_pos.plot([], [], 'r',  label='Est')
    pos_truth_line,= ax_pos.plot([], [], 'gray', label='Truth')
    pos_sig_up,    = ax_pos.plot([], [], 'C1--', lw=1, label='±3 sigma')
    pos_sig_dn,    = ax_pos.plot([], [], 'C1--', lw=1)
    ax_pos.set_ylabel("θ [ deg]")
    ax_pos.set_ylim(pos_min, pos_max)
    ax_pos.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax_pos.set_title("State Estimates")

    # Velocity
    vel_est_line,  = ax_vel.plot([], [], 'r',   label='Est')
    vel_truth_line,= ax_vel.plot([], [], 'gray',label='Truth')
    vel_sig_up,    = ax_vel.plot([], [], 'C1--', lw=1, label='±3 sigma')
    vel_sig_dn,    = ax_vel.plot([], [], 'C1--', lw=1)
    ax_vel.set_ylabel("Vel [ deg/s]")
    ax_vel.set_ylim(vel_min, vel_max)
    ax_vel.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Control
    ctrl_line,     = ax_ctrl.plot([], [], 'b', label='Ctrl Input')
    ax_ctrl.set_ylabel("τ [Nm]")
    ax_ctrl.set_xlabel("Time [s]")
    ax_ctrl.set_ylim(tau_min, tau_max)
    ax_ctrl.set_title("Control Input")

    # Init x-lims (will update each frame)
    ax_pos.set_xlim(0, min(window, t[-1]))
    ax_vel.set_xlim(0, min(window, t[-1]))
    ax_ctrl.set_xlim(0, min(window, t[-1]))

    def init():
        rod_line.set_data([], [])
        trace_line.set_data([], [])
        trace_x.clear()
        trace_y.clear()
        for line in (meas_line, pos_est_line, pos_truth_line,
                     pos_sig_up, pos_sig_dn,
                     vel_est_line, vel_truth_line,
                     vel_sig_up, vel_sig_dn,
                     ctrl_line):
            line.set_data([], [])
        UI_text.set_text('')
        return (rod_line, trace_line, meas_line, pos_est_line, pos_truth_line,
                pos_sig_up, pos_sig_dn, vel_est_line, vel_truth_line,
                vel_sig_up, vel_sig_dn, ctrl_line, UI_text)

    def update(frame_i):
        i = frame_i
        t_now = t[i]

        # --- Pendulum (with short tail only)
        x = x_pend[i]; y = y_pend[i]
        rod_line.set_data([0, x], [0, y])
        trace_x.append(x); trace_y.append(y)
        if len(trace_x) > max_tail:
            del trace_x[0]; del trace_y[0]
        trace_line.set_data(trace_x, trace_y)

        UI_text.set_text(
            f"t = {t_now:.2f} s\n"
            f"θ = {states_truth[i,0]*180/np.pi:.1f}°\n"
            f"τ = {control_hist[i]:.2f} Nm"
        )

        # --- Windowing: GROW -> SCROLL (absolute time on x)
        if t_now <= window:
            i0 = 0
            ax_pos.set_xlim(0, max(t_now, dt))
            ax_vel.set_xlim(0, max(t_now, dt))
            ax_ctrl.set_xlim(0, max(t_now, dt))
        else:
            i0 = i - max_pts + 1
            left = t[i0]; right = t_now
            ax_pos.set_xlim(left, right)
            ax_vel.set_xlim(left, right)
            ax_ctrl.set_xlim(left, right)

        sl = slice(i0, i+1)

        # Position
        meas_line.set_data(t[sl],       R2D*(measurements[sl]))
        pos_est_line.set_data(t[sl],    R2D*(states_est[sl, 0]))
        pos_truth_line.set_data(t[sl],  R2D*(states_truth[sl, 0]))
        pos_sig_up.set_data(t[sl],      R2D*(states_est[sl, 0] + pos_3s[sl]))
        pos_sig_dn.set_data(t[sl],      R2D*(states_est[sl, 0] - pos_3s[sl]))

        # Velocity
        vel_est_line.set_data(t[sl],    R2D*(states_est[sl, 1]))
        vel_truth_line.set_data(t[sl],  R2D*(states_truth[sl, 1]))
        vel_sig_up.set_data(t[sl],      R2D*(states_est[sl, 1] + vel_3s[sl]))
        vel_sig_dn.set_data(t[sl],      R2D*(states_est[sl, 1] - vel_3s[sl]))

        # Control
        ctrl_line.set_data(t[sl], control_hist[sl])

        # With moving x-lims, blit must be False to redraw ticks
        return (rod_line, trace_line, meas_line, pos_est_line, pos_truth_line,
                pos_sig_up, pos_sig_dn, vel_est_line, vel_truth_line,
                vel_sig_up, vel_sig_dn, ctrl_line, UI_text)

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames,
        init_func=init,
        blit=False,              # needed so x-axis ticks update with moving limits
        interval=max(15, int(1000*dt*skip)),
        repeat=True
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
