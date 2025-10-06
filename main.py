import control
import numpy as np
import matplotlib.pyplot as plt

# m*xddot + c*xdot + k*x = u
def get_state_derivative(state, u, k, m, c):
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
    e = y-yc
    edot = ydot - ycdot
    eint = int_e + e*dt

    # evaluate control law
    u = -kp*e -ki*eint -kd*edot
    return u, eint



# system properties
c = 0.4                         # viscous damping 
m = 1                           # mass
k = 0.5                         # stiffness
output_sigmas = np.array([0,0]) # sensor noise

# make damping negative to make the system unstable
c = -c; 

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

# Simulation settings
dt = 0.01
t_final = 20
num_steps = int(t_final / dt)
t = np.linspace(0, t_final, num_steps)

state = x0.copy()
states = [state.copy()]

for ti in t[1:]:
    # evaluate control law
    u, int_e = PID_law(state[0], yc, state[1], ycdot, int_e, dt, gains)

    # u = 0  # no input force
    dx = get_state_derivative(state, u, k, m, c)
    state = euler_integrate(state, dx, dt)
    states.append(state.copy())

states = np.array(states)

# Plot displacement and velocity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Displacement
ax1.plot(t, states[:, 0])
ax1.set_ylabel("Displacement [m]")
ax1.set_title("Damped Spring-Mass System")
ax1.grid(True)

# Velocity
ax2.plot(t, states[:, 1])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Velocity [m/s]")
ax2.grid(True)

plt.tight_layout()
plt.show()


# plot position only
# plt.plot(t, states[:, 0])
# plt.title("Damped Spring-Mass System")
# plt.xlabel("Time [s]")
# plt.ylabel("Displacement [m]")
# plt.grid(True)
# plt.show()