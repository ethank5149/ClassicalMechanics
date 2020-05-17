import matplotlib
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


# Constants
g = 9.81
tf = 100
t0 = 0.0

# Simulation Parameters
m = 1.0
l = 6.0

# Visual Parameters
fps = 24
quality = 300  # DPI of animation

# Initial Conditions
#  V = [v, dv]
# dV = [dv, ddv]

V0 = np.array([0.9*np.pi, 0])


def dVF(t, V):
    v = V[0]
    dv = V[1]
    return np.array([dv, -(g/l)*np.sin(v)])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))

v = soln.y[0, :]
dv = soln.y[1, :]

x = l*np.sin(v)
y = -l*np.cos(v)
dx = l*dv*np.cos(v)
dy = l*dv*np.sin(v)

# # Phase Plots
###############################################################################
# plt.figure()
# plt.suptitle('Phase Plots', fontsize=48)

# plt.subplot(131)
# plt.plot(v, m*dv, color='red', label=r'$p_{\theta} \mathrm{vs.} \theta$')
# plt.xlabel(r'$\theta$', fontsize=16)
# plt.ylabel(r'$p_{\theta}$', fontsize=16)
# plt.title(r'$p_{\theta} \mathrm{vs.} \theta$', fontsize=32)

# plt.subplot(132)
# plt.plot(x, m*dx, color='blue', label=r'$p_x \mathrm{vs.} x$')
# plt.xlabel(r'$x$', fontsize=16)
# plt.ylabel(r'$p_x$', fontsize=16)
# plt.title(r'$p_x \mathrm{vs.} x$', fontsize=32)

# plt.subplot(133)
# plt.plot(y, m*dy, color='darkgreen', label=r'$p_y \mathrm{vs.} y$')
# plt.xlabel(r'$y$', fontsize=16)
# plt.ylabel(r'$p_y$', fontsize=16)
# plt.title(r'$p_y \mathrm{vs.} y$', fontsize=32)
# plt.show()
###############################################################################


# # Poincare Plots
###############################################################################
# plt.figure()
# plt.suptitle('Poincare Plots', fontsize=48)

# plt.subplot(131)
# plt.plot(v[:-1], v[1:], color='red')
# plt.xlabel(r'$\theta_i$', fontsize=16)
# plt.ylabel(r'$\theta_{i+1}$', fontsize=16)
# plt.title(r'$\theta_{i+1} \mathrm{vs.} \theta_i$', fontsize=32)

# plt.subplot(132)
# plt.plot(x[:-1], x[1:], color='gold')
# plt.xlabel(r'$x_i$', fontsize=16)
# plt.ylabel(r'$x_{i+1}$', fontsize=16)
# plt.title(r'$x_{i+1} \mathrm{vs.} x_i$', fontsize=32)

# plt.subplot(133)
# plt.plot(y[:-1], y[1:], color='blue')
# plt.xlabel(r'$y_i$', fontsize=16)
# plt.ylabel(r'$y_{i+1}$', fontsize=16)
# plt.title(r'$y_{i+1} \mathrm{vs.} y_i$', fontsize=32)

# plt.show()
###############################################################################

fig, ax = plt.subplots()
fig.suptitle("Simple Pendulum", fontsize=12)
# xdata, ydata = [], []
ln1, = plt.plot([], [], 'ro', markersize=10)
ln2, = plt.plot([], [], 'k')


def init():
    ax.set_xlim(-1.1*l, 1.1*l)
    ax.set_ylim(-1.1*l, 1.1*l)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ln1, ln2,


def update(frame):
    #     xdata.append(x[frame])
    #     ydata.append(y[frame])
    ln1.set_data(x[frame], y[frame])
    ln2.set_data([0, x[frame]], [0, y[frame]])

    textstr = '\n'.join((
        r'Parameters',
        r'$m=%s$' % m,
        r'$l=%s$' % l,
        r'fps=%s' % fps))

    ax.text(-0.075, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ln1, ln2,


ani = FuncAnimation(fig, update, frames=np.arange(0, len(soln.t)),
                    init_func=init, interval=1000/fps)  # , blit=True)
plt.show()
# matplotlib.use("Agg")
# ani.save(filename='sim.mp4', fps=fps, dpi=quality)
