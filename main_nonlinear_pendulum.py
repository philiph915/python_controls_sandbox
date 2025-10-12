import numpy as np
import matplotlib.pyplot as plt

# I*theta_ddot = tau - m*g*L*cos(theta) - c*theta_dot
def get_state_derivative(state, u, sys_props):
    m = sys_props["m"] 
    g = sys_props["g"]
    L = sys_props["L"]
    c = sys_props["c"]
    x, xdot = state
    dx1 = xdot
    dx2 = u / (m*L**2) - g / L * np.cos(x) - c * dx1 / (m*L**2) # theta_ddot = tau/m*L^2 - g/L*cos(theta) - c*theta_dot / m*L^2 ==> assumes point mass
    return np.array([dx1, dx2])

def euler_integrate(x, dx, dt):
    return x + dx * dt

def rk4_integrate(x, u, dt, f, sys_props):
    """
    Integrate one step using classical Runge–Kutta (RK4).
    x: current state
    u: control input
    dt: timestep
    f: function f(x, u, sys_props) returning xdot

    example usage: state_truth = rk4_integrate(state_truth, u, dt, get_state_derivative, sys_props)
    
    """
    k1 = f(x, u, sys_props)
    k2 = f(x + 0.5 * dt * k1, u, sys_props)
    k3 = f(x + 0.5 * dt * k2, u, sys_props)
    k4 = f(x + dt * k3, u, sys_props)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def add_noise(y, sigmas):
    y = np.array(y)
    return y + np.random.normal(0, sigmas)

def PID_law(y,yc,ydot,ycdot,int_e,dt,gains,u_sat):

    # PID law gains
    kp = gains["kp"]
    kd = gains["kd"]
    ki = gains["ki"]

    # calculate errors
    e = yc-y
    edot = ycdot - ydot     # this isn't truly a PID, because we aren't differentiating positional error; this is more akin to a PIR law from aerospace
    eint = int_e + e*dt

    # evaluate control law
    u = kp*e + ki*eint + kd*edot

    # anti-windup logic
    if np.abs(u) > u_sat:
        eint = int_e
        u = np.clip(u,-u_sat,u_sat)

    # return states calculated in controller block
    control_output = {"u": u, "e": e, "edot": edot, "eint": eint}

    return control_output

def numerical_jacobian(x, u, f, sys_props, delta=1e-5):

    # Compute Jacobian of f(x,u) wrt x using central finite differences
    n = len(x)
    F = np.zeros((n, n))
    for j in range(n):
        perturb = np.zeros(n)
        perturb[j] = delta

        f_plus  = f(x + perturb, u, sys_props)
        f_minus = f(x - perturb, u, sys_props)

        F[:, j] = (f_plus - f_minus) / (2 * delta)
    return F

def extended_kalman_filter(measurement,state_est_prev,control_input,sys_props,dt,P_prev,Q,R):
    
    # Predict (use RK4 for the state)
    state_est_pre = rk4_integrate(state_est_prev, control_input, dt, get_state_derivative, sys_props)

    # Discrete transition Jacobian (from continuous via Euler linearization)
    F_c = numerical_jacobian(state_est_prev, control_input, get_state_derivative, sys_props)
    F_d = np.eye(len(state_est_prev)) + F_c * dt
    P_est_pre = F_d @ P_prev @ F_d.T + Q
    
    # corrector equations
    H = np.array([[1.0, 0.0]])                                                  # only measuring position
    
    y_tilde = measurement - H @ state_est_pre                                   # measurement residual
    S = H @ P_est_pre @ np.transpose(H) + R                                     # covariance residual
    K = P_est_pre @ np.transpose(H) @ np.linalg.inv(S)                          # Kalman gain
    
    state_est_post = state_est_pre + K @ y_tilde                                # updated state estimate
    P_est_post = (np.eye(len(state_est_post)) - K @ H ) @ P_est_pre             # updated covariance estimate

    EKF_out = {
        "F": F_d,
        "H": H,
        "state_est_post": state_est_post,
        "K": K,
        "P_est_post": P_est_post
    }

    return EKF_out


# ======================== Begin Simulation ============================== #

# Simulation settings
dt = 0.001
t_final = 5
num_steps = int(t_final / dt)
t = np.linspace(0, t_final, num_steps)

theta_0_deg = 30        # starting position [deg]
cmd_pos_deg = 45        # desired position [deg]
t_ctrl_enable = 2       # time at which to turn on the controller [s]

# system properties
L = 0.4                              # pendulum length [m]
m = 1                                # mass [kg]
g = 9.81                             # gravity [mps2]
c = 0.02                             # viscous damping coefficient [N*m*s/rad]

Q_sigmas = np.array([0.001, 0.001])  # simulation process noise (velocity and acceleration noise)
R_sigmas = np.array([0.25])          # simulation sensor noise [rad] (only sensing position)

sys_props = {"L": L, "m": m, "g": g, "c": c, "Q": Q_sigmas, "R": R_sigmas}

# controller settings
kp = 100
ki = 40
kd = 10
u_sat = 25  # max control torque in Nm
gains = {"kp": kp, "ki": ki, "kd": kd}
yc = cmd_pos_deg * np.pi / 180   # desired position in radians
ycdot = 0               # desired velocity in radians per second

# initial states
x0 = np.array([ theta_0_deg*np.pi / 180, 0 ])
int_e = 0
state_truth = x0.copy()
u = 0



# Initial State and Covariance estimates for EKF
state_est = np.array([1e-1, 1e-3])
P_est = np.diag([4, 10])

# State histories
outputs_truth    = [state_truth[0]]
measurements     = [state_truth[0] + np.random.normal(0, R_sigmas[0])]
states_truth     = [state_truth.copy()]
states_est       = [state_est.copy()]
e_hist           = [(yc - x0[0])]
edot_hist        = [(ycdot - x0[1])]
eint_hist        = [int_e]
u_hist           = [u]
P_hist           = [P_est.copy()]  # Kalman Filter Covariance Estimate (this is our confidence in our state estimates)

# Kalman filter covariance terms
Q = np.diag(Q_sigmas**2)                            # process covariance
R = np.array([[(R_sigmas[0]*10)**2]])                    # sensor covariance (variance = sigma^2)

# Flag for enabling the Kalman Filter
useEKF = True

# run sim
for ti in t[1:]:
    # add sensor noise
    measurement = (add_noise(state_truth[0], R_sigmas)).item()  # float(state_truth[0] + np.random.normal(0, R_sigmas[0]))

    # run Extended Kalman Filter
    if useEKF:
        EKF_out = extended_kalman_filter(measurement,state_est,u,sys_props,dt,P_est,Q,R)
        P_est = EKF_out["P_est_post"]
        state_est = EKF_out["state_est_post"]
    else:
        state_est = state_truth.copy()

    # evaluate control law (use state estimates for feedback)
    control = PID_law(state_est[0], yc, state_est[1], ycdot, int_e, dt, gains, u_sat)
    if ti > t_ctrl_enable:
        u = control["u"]
        int_e = control["eint"] 
    else:
        u = 0
        int_e = 0

    # evaluate dynamics
    state_truth = rk4_integrate(state_truth, u, dt, get_state_derivative, sys_props)
    state_truth = add_noise(state_truth,sys_props["Q"]) # add proces noise
    
    # store state histories
    states_truth.append(state_truth.copy())
    measurements.append(measurement)
    outputs_truth.append(state_truth[0])
    states_est.append(state_est.copy())
    P_hist.append(P_est.copy())
    e_hist.append(control["e"])
    edot_hist.append(control["edot"])
    eint_hist.append(int_e)
    u_hist.append(u)

# post-process
states_truth  = np.array(states_truth)
measurements  = np.array(measurements)
states_est    = np.array(states_est)
P_hist    = np.array(P_hist)
e_hist    = np.array(e_hist)
edot_hist = np.array(edot_hist)
eint_hist = np.array(eint_hist)
u_hist    = np.array(u_hist)


# ================ Show Plots ================== # 

# Plot displacement and velocity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Displacement
ax1.plot(t, measurements, label="Meas", alpha=0.6)
ax1.plot(t, states_est[:, 0], label = "Est")
ax1.plot(t, states_truth[:, 0], label = "Truth")
ax1.axhline(y=yc, label = "cmd", color = "gray")
ax1.set_ylabel("Displacement [rad]")
ax1.set_title("Nonlinear Pendulum")
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
ax3.set_xlabel("Time [s]")
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


import matplotlib.animation as animation

# Pendulum position from truth data
x_pend = L * np.cos(states_truth[:, 0])
y_pend = L * np.sin(states_truth[:, 0])

fig_anim, ax_anim = plt.subplots(figsize=(5, 5))
ax_anim.set_xlim(-L*1.2, L*1.2)
ax_anim.set_ylim(-L*1.2, L*1.2)
ax_anim.set_aspect('equal', 'box')
ax_anim.grid(True)
ax_anim.set_title("Pendulum Animation")

# Create line for pendulum rod and point for mass
rod_line, = ax_anim.plot([], [], 'o-', lw=2, color='C0')
trace_line, = ax_anim.plot([], [], 'r-', lw=2, alpha=0.25)  # optional trace of the bob

# Text for displaying simulation time
time_text = ax_anim.text(
    0.02, 0.95, '', transform=ax_anim.transAxes,
    ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

# Text for displaying pendulum angle
angle_text = ax_anim.text(
    0.02, 0.85, '', transform=ax_anim.transAxes,
    ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

trace_x, trace_y = [], []

def init():
    rod_line.set_data([], [])
    trace_line.set_data([], [])
    trace_x.clear()
    trace_y.clear()
    time_text.set_text('')
    angle_text.set_text('')
    return rod_line, trace_line, time_text, angle_text

def update(frame):
    x = x_pend[frame]
    y = y_pend[frame]

    # Rod from origin to mass
    rod_line.set_data([0, x], [0, y])

    # # Remove trace history
    # if frame > 1000:
    #     trace_x.pop(0)
    #     trace_y.pop(0)

    # Add to trace
    trace_x.append(x)
    trace_y.append(y)
    trace_line.set_data(trace_x, trace_y)

    # Update time text
    time_text.set_text(f't = {t[frame]:.2f} s')
    angle_text.set_text(f'theta = {outputs_truth[frame]*180/np.pi:.2f} deg')

    return rod_line, trace_line, time_text, angle_text

# Animate
fps = int(1/dt) // 60  # skip frames to control playback speed
ani = animation.FuncAnimation(
    fig_anim,
    update,
    frames=range(0, len(t), fps),
    init_func=init,
    blit=True,
    interval=dt*1000*fps,  # in ms
    repeat=True
)

plt.show()



# Keep the figures open
input("Press Enter to close...")