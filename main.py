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
    control_output = {"u": u, "eint": eint, "e": e, "edot": edot, "eint": eint}

    return control_output


# ======================== Begin Simulation ============================== #

# system properties
c = 0.4                              # viscous damping 
m = 1                                # mass
k = 0.5                              # stiffness
c = -c;                              # make damping negative to make the system unstable
output_sigmas = np.array([0.1, 0.6]) # sensor noise
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

# Simulation settings
dt = 0.01
t_final = 20
num_steps = int(t_final / dt)
t = np.linspace(0, t_final, num_steps)

# State histories
states_truth    = [x0.copy()]
states_sensed   = states_truth.copy()
e_hist          = [(yc - x0[0])]
edot_hist       = [(ycdot - x0[1])]
eint_hist       = [int_e]

# run sim
for ti in t[1:]:
    # add sensor noise
    state_sensed = add_noise(state_truth,output_sigmas);

    # evaluate control law (use noisy signal for feedback)
    control_output = PID_law(state_sensed[0], yc, state_sensed[1], ycdot, int_e, dt, gains)
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
    e_hist.append(control_output["e"])
    edot_hist.append(control_output["edot"])
    eint_hist.append(int_e)

# post-process
states_truth = np.array(states_truth)
states_sensed = np.array(states_sensed)
e_hist    = np.array(e_hist)
edot_hist = np.array(edot_hist)
eint_hist = np.array(eint_hist)

# Plot displacement and velocity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Displacement
ax1.plot(t, states_sensed[:, 0], label="Sensed", linestyle='--')
ax1.plot(t, states_truth[:, 0], label = "Truth")
ax1.set_ylabel("Displacement [m]")
ax1.set_title("Damped Spring-Mass System")
ax1.grid(True)
ax1.legend()

# Velocity
ax2.plot(t, states_sensed[:, 1], label = "Sensed",linestyle='--')
ax2.plot(t, states_truth[:, 1], label = "Truth")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Velocity [m/s]")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show(block=False)

# Plot error states
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Position error
ax1.plot(t, e_hist)
ax1.set_ylabel("e [m]")
ax1.set_title("PID Error States")
ax1.grid(True)

# Velocity error
ax2.plot(t, edot_hist)
ax2.set_ylabel("ė [m/s]")
ax2.grid(True)

# Integral error
ax3.plot(t, eint_hist)
ax3.set_ylabel("∫e dt")
ax3.set_xlabel("Time [s]")
ax3.grid(True)

plt.tight_layout()
plt.show(block=False)

# Keep the figures open
input("Press Enter to close...")