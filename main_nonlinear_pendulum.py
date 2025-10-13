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
        eint = int_e                    # stop integrating
        u = np.clip(u,-u_sat,u_sat)     # clamp control input

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
    F_d = np.eye(len(state_est_prev)) + F_c * dt                                # Euler linearization
    P_est_pre = F_d @ P_prev @ F_d.T + Q                                        # A priori covariance estimate
    
    # corrector equations
    H = np.array([[1.0, 0.0]])                                                  # only measuring position
    
    y_tilde = measurement - H @ state_est_pre                                   # measurement residual
    S = H @ P_est_pre @ np.transpose(H) + R                                     # covariance residual
    K = P_est_pre @ np.transpose(H) @ np.linalg.inv(S)                          # Kalman gain
    
    state_est_post = state_est_pre + K @ y_tilde                                # updated state estimate
    P_est_post = (np.eye(len(state_est_post)) - K @ H ) @ P_est_pre             # A posteriori covariance estimate

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

Q_sigmas = np.array([0.001, 0.001])  # simulation process noise 
R_sigmas = np.array([0.1])          # simulation sensor noise [rad] (only sensing position)

sys_props = {"L": L, "m": m, "g": g, "c": c, "Q": Q_sigmas, "R": R_sigmas}

# controller settings
kp = 100
ki = 40
kd = 10
u_sat = 10                              # max control torque in Nm
gains = {"kp": kp, "ki": ki, "kd": kd}
yc = cmd_pos_deg * np.pi / 180          # desired position in radians
ycdot = 0                               # desired velocity in radians per second

# initial states
x0 = np.array([ theta_0_deg*np.pi / 180, 0 ])
int_e = 0
state_truth = x0.copy()
u = 0

# Initial State and Covariance estimates for EKF
state_est = np.array([1e-1, 1e-3])
P_est = np.diag([.1, 10])

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
R = np.array([1**2])                                # sensor covariance (variance = sigma^2)

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
P_hist        = np.array(P_hist)
e_hist        = np.array(e_hist)
edot_hist     = np.array(edot_hist)
eint_hist     = np.array(eint_hist)
u_hist        = np.array(u_hist)


# ================ Show Plots ================== # 

# Make static plots
from pendulum_plots import make_pendulum_static_plots
make_pendulum_static_plots(
    t,t_final, 
    states_truth, states_est, measurements, 
    P_hist, u_hist, yc, 
    e_hist, edot_hist, eint_hist
    )

# # Show pendulum "DAQ" figure
# from pendulum_plots import animate_pendulum_dashboard
# ani = animate_pendulum_dashboard(
#     t, dt, states_truth, states_est, measurements, P_hist, u_hist, L, saveFig = False
# )

# Show pendulum "DAQ" figure (includes dynamic y-axis scaling)
from pendulum_plots import animate_pendulum_dashboard_dynamic
ani = animate_pendulum_dashboard_dynamic(
    t, dt, states_truth, states_est, measurements, P_hist, u_hist, L, saveFig = False, window=1
)

# NOTE: ADD BACK IN DYNAMIC Y-SCALING FOR README, REDUCE WINDOW SIZE TO SOMETHING LIKE 1 SECOND