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

    x_pend = L * np.cos(states_truth[:, 0])
    y_pend = L * np.sin(states_truth[:, 0])

    # Precompute ±3 sigma envelopes 
    pos_3s = 3.0 * np.sqrt(P_hist[:, 0, 0])
    vel_3s = 3.0 * np.sqrt(P_hist[:, 1, 1])

    # Reasonable fixed y-limits 
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
    ax_vel.set_xlabel("Time [s]")

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

def make_pendulum_static_plots(t, t_final,
    states_truth,     # (N,2): [theta, theta_dot]
    states_est,       # (N,2)
    measurements,     # (N,)  position measurements (rad)
    P_hist,           # (N,2,2)
    u_hist,           # (N,)
    yc,               # (N,) 
    e_hist,           # (N,)
    edot_hist,        # (N,)
    eint_hist,        # (N,)
):
    # Plot displacement and velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Displacement
    ax1.plot(t, measurements, linestyle = 'none', marker = '.', label="Meas", alpha=0.3, markersize=2)
    ax1.plot(t, states_est[:, 0], label = "Est")
    ax1.plot(t, states_truth[:, 0], label = "Truth")
    ax1.axhline(y=yc, label = "cmd", color = "gray")
    ax1.set_ylabel("Displacement [rad]")
    ax1.set_title("Pendulum Simulation")
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, t_final)

    # Velocity
    ax2.plot(t, states_est[:, 1], label = "Est")
    ax2.plot(t, states_truth[:, 1], label = "Truth")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity [rad/s]")
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, t_final)

    plt.tight_layout()
    plt.show(block=False)

    # Plot error states
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    # Position error
    ax1.plot(t, e_hist)
    ax1.set_ylabel("e [rad]")
    ax1.set_title("Controller States")
    ax1.grid(True)
    ax1.set_xlim(0, t_final)

    # Velocity error
    ax2.plot(t, edot_hist)
    ax2.set_ylabel("ė [rad/s]")
    ax2.grid(True)
    ax2.set_xlim(0, t_final)

    # Integral error
    ax3.plot(t, eint_hist)
    ax3.set_ylabel("∫e dt")
    ax3.grid(True)
    ax3.set_xlim(0, t_final)

    # Control Input
    ax4.plot(t, u_hist)
    ax4.set_ylabel("control input [Nm]")
    ax4.set_xlabel("Time [s]")
    ax4.grid(True)
    ax4.set_xlim(0, t_final)

    plt.tight_layout()
    plt.show(block=False)

    # Plot Kalman Filter Covariance
    fig_cov, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Position variance
    ax[0].plot(t, P_hist[:, 0, 0])
    ax[0].set_ylabel("Var(position)")
    ax[0].set_title("Kalman Filter Covariance")
    ax[0].grid(True)
    ax[0].set_xlim(0, t_final)

    # Velocity variance
    ax[1].plot(t, P_hist[:, 1, 1])
    ax[1].set_ylabel("Var(velocity)")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True)
    ax[1].set_xlim(0, t_final)

    plt.tight_layout()
    plt.show(block=False)

    # show state estimates with shaded bounds
    fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    pos_sigma = np.sqrt(P_hist[:, 0, 0])
    vel_sigma = np.sqrt(P_hist[:, 1, 1])

    # Plot displacement and velocity
    # Displacement
    ax4.fill_between(
        t,
        states_est[:, 0] - 3*pos_sigma,
        states_est[:, 0] + 3*pos_sigma,
        color='C1', alpha=0.2, label='±3 sigma'
    )
    ax4.scatter(t, measurements, label="Sensed", s=2, alpha=0.25)
    ax4.plot(t, states_est[:, 0], label = "Est", color = "Red")
    ax4.plot(t, states_truth[:, 0], label = "Truth", color = "Gray")
    ax4.set_ylabel("Displacement [rad]")
    ax4.set_title("EKF Estimates")
    ax4.grid(True)
    ax4.legend()
    ax4.set_xlim(0, t_final)

    # Velocity
    ax5.fill_between(
        t,
        states_est[:, 1] - 3*vel_sigma,
        states_est[:, 1] + 3*vel_sigma,
        color='C1', alpha=0.2, label='±3 sigma'
    )
    # ax5.scatter(t, states_sensed[:, 1], label = "Sensed", s=2)
    ax5.plot(t, states_est[:, 1], label = "Est", color = "Red")
    ax5.plot(t, states_truth[:, 1], label = "Truth", color = "Gray")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Velocity [rad/s]")
    ax5.grid(True)
    ax5.legend()
    ax5.set_xlim(0, t_final)

    plt.tight_layout()
    plt.show(block=False)
