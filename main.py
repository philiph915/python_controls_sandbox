import control
import numpy as np
import matplotlib.pyplot as plt

# m*xddot + c*xdot + k*x = u
def get_state_derivative(state, u, sys_props):
    k = sys_props["k"]
    c = sys_props["c"]
    m = sys_props["m"] 
    x, xdot = state
    dx1 = xdot
    dx2 = (-k*x - c*xdot + u) / m
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
    return y + np.random.normal(loc=0, scale=sigmas)

def PID_law(y,yc,ydot,ycdot,int_e,dt,gains):

    # PID law gains
    kp = gains["kp"]
    kd = gains["kd"]
    ki = gains["ki"]

    # calculate errors
    e = yc-y
    edot = ycdot - ydot
    eint = int_e + e*dt

    # evaluate control law
    u = kp*e + ki*eint + kd*edot

    # return states calculated in controller block
    control_output = {"u": u, "e": e, "edot": edot, "eint": eint}

    return control_output

def numerical_jacobian(x, u, sys_props, delta=1e-5):

    # Compute Jacobian of f(x,u) wrt x using central finite differences
    n = len(x)
    F = np.zeros((n, n))
    for j in range(n):
        perturb = np.zeros(n)
        perturb[j] = delta

        f_plus  = get_state_derivative(x + perturb, u, sys_props)
        f_minus = get_state_derivative(x - perturb, u, sys_props)

        F[:, j] = (f_plus - f_minus) / (2 * delta)
    return F

def extended_kalman_filter(state_sensed,state_est_prev,control_input,sys_props,dt,P_prev,Q,R):
    
    # predictor equations
    state_est_dot = get_state_derivative(state_est_prev,control_input,sys_props)    # evaluate onboard model using previous state estimate
    state_est_pre = euler_integrate(state_est_prev,state_est_dot,dt)                # precicted state estimate
    F = numerical_jacobian(state_est_prev,control_input,sys_props)                  # state transition matrix (continuous)
    F_discrete = np.eye(len(state_est_prev)) + F * dt                               # state transition matrix (discrete, assumes 1st order euler integration)
    P_est_pre = F_discrete @ P_prev @ np.transpose(F_discrete) + Q                  # predicted covariance estimate
    
    # corrector equations
    H = np.eye(len(state_est_pre))                                                  # there is no measurement model, so H is simply identity
    
    # HACK: only measure position, not velocity (forces EKF to estimate velocity)
    if True:
        state_sensed = state_sensed[0]
        H = np.array([[1,0]])
        R = np.array([R[0,0]]) 
    
    y_tilde = state_sensed - H @ state_est_pre                                      # measurement residual
    S = H @ P_est_pre @ np.transpose(H) + R                                         # covariance residual
    K = P_est_pre @ np.transpose(H) @ np.linalg.inv(S)                              # Kalman gain
    state_est_post = state_est_pre + K @ y_tilde                                    # updated state estimate
    P_est_post = (np.eye(len(state_est_post)) - K @ H ) @ P_est_pre                 # updated covariance estimate

    EKF_out = {
        "F": F,
        "H": H,
        "state_est_post": state_est_post,
        "K": K,
        "P_est_post": P_est_post
    }

    return EKF_out


# ======================== Begin Simulation ============================== #

# system properties
c = 0.4                              # viscous damping 
m = 1                                # mass
k = 0.5                              # stiffness
c = -c;                              # make damping negative to make the system unstable
output_sigmas = np.array([0.2, 0.6]) # sensor noise
sys_props = {"c" : c, "m" : m, "k" : k, "output_sigmas" : output_sigmas}

# controller settings
kp = 5
ki = 1
kd = 3
gains = {"kp": kp, "ki": ki, "kd": kd}
yc = 1
ycdot = 0

# initial states
x0 = np.array([0,0])
int_e = 0
state_truth = x0.copy()
u = 0

# Initial State and Covariance estimates for EKF
state_est = np.array([1e-1, 1e-3])
P_est = np.diag([1, 4])

# Simulation settings
dt = 0.01
t_final = 10
num_steps = int(t_final / dt)
t = np.linspace(0, t_final, num_steps)

# State histories
states_truth    = [x0.copy()]
states_sensed   = states_truth.copy()
states_est      = [state_est.copy()]
e_hist          = [(yc - x0[0])]
edot_hist       = [(ycdot - x0[1])]
eint_hist       = [int_e]
P_hist          = [P_est.copy()]  # Kalman Filter Covariance Estimate (this is our confidence in our state estimates)

Q = np.diag([1e-2, 1e-2])                        # process covariance
R = np.diag([0.5**2, 1.0**2])                    # sensor covariance (variance = sigma^2)

# Flag for enabling the Kalman Filter
useEKF = True

# run sim
for ti in t[1:]:
    # add sensor noise
    state_sensed = add_noise(state_truth,output_sigmas);

    # run Extended Kalman Filter
    state_est_prev = state_est
    if useEKF:
        EKF_out = extended_kalman_filter(state_sensed,state_est_prev,u,sys_props,dt,P_est,Q,R)
        P_est = EKF_out["P_est_post"]
        state_est = EKF_out["state_est_post"]
    else:
        state_est = state_sensed

    # evaluate control law (use state estimates for feedback)
    control_output = PID_law(state_est[0], yc, state_est[1], ycdot, int_e, dt, gains)
    u = control_output["u"] 
    int_e = control_output["eint"]

    # evaluate dynamics
    # u = 0  # no input force
    dx = get_state_derivative(state_truth, u, sys_props)
    
    # integrate 
    state_truth = euler_integrate(state_truth, dx, dt)
    
    # store state histories
    states_truth.append(state_truth.copy())
    states_sensed.append(state_sensed.copy())
    states_est.append(state_est.copy())
    P_hist.append(P_est.copy())
    e_hist.append(control_output["e"])
    edot_hist.append(control_output["edot"])
    eint_hist.append(int_e)

# post-process
states_truth  = np.array(states_truth)
states_sensed = np.array(states_sensed)
states_est    = np.array(states_est)
P_hist    = np.array(P_hist)
e_hist    = np.array(e_hist)
edot_hist = np.array(edot_hist)
eint_hist = np.array(eint_hist)


# ================ Show Plots ================== # 

# Plot displacement and velocity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Displacement
ax1.plot(t, states_sensed[:, 0], label="Sensed")
ax1.plot(t, states_est[:, 0], label = "Est")
ax1.plot(t, states_truth[:, 0], label = "Truth")
ax1.set_ylabel("Displacement [m]")
ax1.set_title("Damped Spring-Mass System")
ax1.grid(True)
ax1.legend()
ax1.set_xlim(0, t_final)

# Velocity
ax2.plot(t, states_sensed[:, 1], label = "Sensed")
ax2.plot(t, states_est[:, 1], label = "Est")
ax2.plot(t, states_truth[:, 1], label = "Truth")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Velocity [m/s]")
ax2.grid(True)
ax2.legend()
ax2.set_xlim(0, t_final)

plt.tight_layout()
plt.show(block=False)

# Plot error states
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Position error
ax1.plot(t, e_hist)
ax1.set_ylabel("e [m]")
ax1.set_title("PID Error States")
ax1.grid(True)
ax1.set_xlim(0, t_final)

# Velocity error
ax2.plot(t, edot_hist)
ax2.set_ylabel("ė [m/s]")
ax2.grid(True)
ax2.set_xlim(0, t_final)

# Integral error
ax3.plot(t, eint_hist)
ax3.set_ylabel("∫e dt")
ax3.set_xlabel("Time [s]")
ax3.grid(True)
ax3.set_xlim(0, t_final)

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
ax4.scatter(t, states_sensed[:, 0], label="Sensed", s=2)
ax4.plot(t, states_est[:, 0], label = "Est", color = "Red")
ax4.plot(t, states_truth[:, 0], label = "Truth", color = "Gray")
ax4.set_ylabel("Displacement [m]")
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
ax5.scatter(t, states_sensed[:, 1], label = "Sensed", s=2)
ax5.plot(t, states_est[:, 1], label = "Est", color = "Red")
ax5.plot(t, states_truth[:, 1], label = "Truth", color = "Gray")
ax5.set_xlabel("Time [s]")
ax5.set_ylabel("Velocity [m/s]")
ax5.grid(True)
ax5.legend()
ax5.set_xlim(0, t_final)



plt.tight_layout()
plt.show(block=False)

# Keep the figures open
input("Press Enter to close...")