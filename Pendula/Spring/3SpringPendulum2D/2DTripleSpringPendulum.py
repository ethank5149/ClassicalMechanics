from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# Constants
g = 9.81
tf = 100
t0 = 0.0
scale = 3

# Simulation Parameters
m1 = 1.0
m2 = 1.0
m3 = 1.0
l01 = 3.0
l02 = 3.0
l03 = 3.0
k1 = 20
k2 = 20
k3 = 20

# Visual Parameters
fps = 24
quality = 300  # DPI of animation

#    Initial Conditions
#    V = [x1, y1, x2, y2, x3, y3, dx1, dy1, dx2, dy2, dx3, dy3]
#    dV = [dx1, dy1, dx2, dy2, dx3, dy3, ddx1, ddy1, ddx2, ddy2, ddx3, ddy3]
V0 = np.array([3, 0, 6, 0, 9, 0, 0, 0, 0, 0, 0, 0])


def dVF(t, V):
    x1 = V[0]
    y1 = V[1]
    x2 = V[2]
    y2 = V[3]
    x3 = V[4]
    y3 = V[5]
    dx1 = V[6]
    dy1 = V[7]
    dx2 = V[8]
    dy2 = V[9]
    dx3 = V[10]
    dy3 = V[11]

    ddx1 = (k1*l01*x1)/(m1*np.sqrt(x1**2+y1**2))-(k1*x1)/(m1)+(k2*l02*x1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*l02*x2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*x1)/(m1)+(k2*x2)/(m1)
    ddx2 = -(k2*l02*x1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*l02*x2)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*x1)/(m2)-(k2*x2)/(m2)+(k3*l03*x2)/(m2*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))-(k3*l03*x3)/(m2*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))-(k3*x2)/(m2)+(k3*x3)/(m2)
    ddx3 = -(k3*l03*x2)/(m3*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))+(k3*l03*x3)/(m3*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))+(k3*x2)/(m3)-(k3*x3)/(m3)
    ddy1 = -g+(k1*l01*y1)/(m1*np.sqrt(x1**2+y1**2))-(k1*y1)/(m1)+(k2*l02*y1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*l02*y2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))-(k2*y1)/(m1)+(k2*y2)/(m1)
    ddy2 = -g-(k2*l02*y1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*l02*y2)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2))+(k2*y1)/(m2)-(k2*y2)/(m2)+(k3*l03*y2)/(m2*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))-(k3*l03*y3)/(m2*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))-(k3*y2)/(m2)+(k3*y3)/(m2)
    ddy3 = -g-(k3*l03*y2)/(m3*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))+(k3*l03*y3)/(m3*np.sqrt(x2**2-2*x2*x3+x3**2+y2**2-2*y2*y3+y3**2))+(k3*y2)/(m3)-(k3*y3)/(m3)

    return np.array([dx1, dy1, dx2, dy2, dx3, dy3, ddx1, ddy1, ddx2, ddy2, ddx3, ddy3])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))

#    Remember
#    V = [x1, y1, x2, y2, x3, y3, dx1, dy1, dx2, dy2, dx3, dy3]
x1 = soln.y[0, :]
y1 = soln.y[1, :]
x2 = soln.y[2, :]
y2 = soln.y[3, :]
x3 = soln.y[4, :]
y3 = soln.y[5, :]
dx1 = soln.y[6, :]
dy1 = soln.y[7, :]
dx2 = soln.y[8, :]
dy2 = soln.y[9, :]
dx3 = soln.y[10, :]
dy3 = soln.y[11, :]

fig = plt.figure()
ax = plt.subplot()
fig.suptitle("Triple Spring Pendulum", fontsize=12)
ln4, = ax.plot([0, x1[0]], [0, y1[0]], 'k')
ln5, = ax.plot([x1[0], x2[0]], [y1[0], y2[0]], 'k')
ln6, = ax.plot([x2[0], x3[0]], [y2[0], y3[0]], 'k')
ln1, = ax.plot((x1[0],), (y1[0],), 'ro', markersize=10)
ln2, = ax.plot((x2[0],), (y2[0],), 'bo', markersize=10)
ln3, = ax.plot((x3[0],), (y3[0],), 'go', markersize=10)


def init():
    ax.set_xlim([-scale*(l01+l02+l03), scale*(l01+l02+l03)])
    ax.set_ylim([-scale*(l01+l02+l03), scale*(l01+l02+l03)])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ln1, ln2, ln3, ln4, ln5, ln6,


def update(frame):
    ln4.set_data([0, x1[frame]], [0, y1[frame]])
    ln5.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    ln6.set_data([x2[frame], x3[frame]], [y2[frame], y3[frame]])
    ln1.set_data(x1[frame], y1[frame])
    ln2.set_data(x2[frame], y2[frame])
    ln3.set_data(x3[frame], y3[frame])
    return ln1, ln2, ln3, ln4, ln5, ln6,


ani = FuncAnimation(fig, update, frames=np.arange(0, len(soln.t)),
                    init_func=init, interval=1000/fps, repeat=True)  # , blit=True)

# plt.show()
matplotlib.use("Agg")
ani.save(filename='sim.mp4', fps=fps, dpi=quality)
