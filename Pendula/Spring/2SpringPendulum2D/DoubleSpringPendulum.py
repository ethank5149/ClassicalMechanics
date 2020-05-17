from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# Constants
g = 9.81
tf = 100
t0 = 0.0

# Simulation Parameters
m1 = 3.0
m2 = 1.0
l01 = 3.0
l02 = 3.0
k1 = 20
k2 = 20

# Visual Parameters
fps = 24
quality = 300  # DPI of animation

#    Initial Conditions
#    V = [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
#    dV = [dx1, dy1, dx2, dy2, ddx1, ddy1, ddx2, ddy2]
V0 = np.array([3, 0, 6, 0, 0, 0, 0, 0])


def dVF(t, V):
    x1 = V[0]
    y1 = V[1]
    x2 = V[2]
    y2 = V[3]
    dx1 = V[4]
    dy1 = V[5]
    dx2 = V[6]
    dy2 = V[7]


    ddx1 = (k1*l01*x1)/(m1*np.sqrt(x1**2+y1**2))-(k1*x1)/(m1)+(k2*l02*x1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2 -
                                                                                      2*y1*y2+y2**2))-(k2*l02*x2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*x1)/(m1)+(k2*x2)/(m1)
    ddx2 = -(k2*l02*x1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*l02*x2) / \
    (m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*x1)/(m2)-(k2*x2)/(m2)
    ddy1 = -g+(k1*l01*y1)/(m1*np.sqrt(x1**2+y1**2))-(k1*y1)/(m1)+(k2*l02*y1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1 **
                                                                                     2-2*y1*y2+y2**2))-(k2*l02*y2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*y1)/(m1)+(k2*y2)/(m1)
    ddy2 = -g-(k2*l02*y1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*l02*y2) / \
    (m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*y1)/(m2)-(k2*y2)/(m2)

    return np.array([dx1, dy1, dx2, dy2, ddx1, ddy1, ddx2, ddy2])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))

#    Remember
#    V = [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
x1 = soln.y[0, :]
y1 = soln.y[1, :]
x2 = soln.y[2, :]
y2 = soln.y[3, :]
dx1 = soln.y[4, :]
dy1 = soln.y[5, :]
dx2 = soln.y[6, :]
dy2 = soln.y[7, :]

fig = plt.figure()
ax = plt.subplot()
fig.suptitle("Double Spring Pendulum", fontsize=12)
ln3, = ax.plot([0, x1[0]], [0, y1[0]], 'k')
ln4, = ax.plot([x1[0], x2[0]], [y1[0], y2[0]], 'k')
ln1, = ax.plot((x1[0],), (y1[0],), 'ro', markersize=10)
ln2, = ax.plot((x2[0],), (y2[0],), 'bo', markersize=10)

def init():
    ax.set_xlim(
        [1.1*min([min(x1), min(x2)])-1, 1.1*max([max(x1), max(x2)])+1])
    ax.set_ylim(
        [1.1*min([min(y1), min(y2)])-1, 1.1*max([max(y1), max(y2)])+1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ln1, ln2, ln3, ln4,


def update(frame):
    ln3.set_data([0, x1[frame]], [0, y1[frame]])
    ln4.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    ln1.set_data(x1[frame], y1[frame])
    ln2.set_data(x2[frame], y2[frame])
    return ln1, ln2, ln3, ln4,


ani = FuncAnimation(fig, update, frames=np.arange(0, len(soln.t)),
                    init_func=init, interval=1000/fps, repeat=True)  # , blit=True)

# plt.show()
matplotlib.use("Agg")
ani.save(filename='sim.mp4', fps=fps, dpi=quality)
