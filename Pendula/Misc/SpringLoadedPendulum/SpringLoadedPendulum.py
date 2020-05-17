from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime


# Constants
g = 9.81
tf = 100
t0 = 0.0

# Simulation Parameters
m = 1.0
k1 = 2
k2 = 2
l = 2.0
l01 = 3
l02 = 5
w = l  # Total width is 2w

# Visual Parameters
fps = 24
quality = 300  # DPI of animation

# Initial Conditions
#  V = [v, dv]
# dV = [dv, ddv]
V0 = np.array([-(1)*np.pi, 0])


def dVF(t, V):
    v = V[0]
    dv = V[1]

    ddv = (-g*m*np.sin(v)+(k1*l*l01*np.sin(v))/(np.sqrt(-2*l**2*np.cos(v)+2*l**2+2*l*w*np.sin(v)+w**2))-k1*l*np.sin(v)+(k1*l01*w*np.cos(v))/(np.sqrt(-2*l**2*np.cos(v)+2*l**2+2*l*w*np.sin(v)+w**2))-k1*w *
           np.cos(v)+(k2*l*l02*np.sin(v))/(np.sqrt(-2*l**2*np.cos(v)+2*l**2-2*l*w*np.sin(v)+w**2))-k2*l*np.sin(v)-(k2*l02*w*np.cos(v))/(np.sqrt(-2*l**2*np.cos(v)+2*l**2-2*l*w*np.sin(v)+w**2))+k2*w*np.cos(v))/(l*m)

    return np.array([dv, ddv])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))


v = soln.y[0, :]
dv = soln.y[1, :]
x = l*np.sin(v)
y = -l*np.cos(v)
dx = l*dv*np.cos(v)
dy = l*dv*np.sin(v)

print(f'Starting Simulation At: {datetime.datetime.now()}')
print('\nParameters')
print(f'm = {m}')
print(f'k_1 = {k1}')
print(f'k_2 = {k2}')
print(f'l = {l}')
print(fr'l_{0,1} = {l01}')
print(fr'l_{0,2} = {l02}')
print(f'fps = {fps}')
print(f'DPI = {quality}')
print(f'\nInitial Conditions')
print(f'theta = {V0[0]}')
print(f'dtheta = {V0[1]}')

darkred = (0.545, 0, 0)
darkgreen = (0.004, 0.196, 0.125)

fig, ax = plt.subplots()
fig.suptitle("Spring Loaded Pendulum", fontsize=12)
plt.axvline(x=-w, color='k', linewidth=4)
plt.axvline(x=w, color='k', linewidth=4)
ln2, = plt.plot([], [], 'k')
ln3, = plt.plot([], [], 'k--')
ln4, = plt.plot([], [], 'k--')
ln1, = plt.plot([], [], 'ro', markersize=10+2*m)


def init():
    ax.set_xlim(-1.5*w, 1.5*w)
    ax.set_ylim(-1.5*w, 1.5*w)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ln1, ln2, ln3, ln4,


def update(frame):
    plt.axvline(x=-w, color='k', linewidth=4)
    plt.axvline(x=w, color='k', linewidth=4)
    ln2.set_data([0, x[frame]], [0, y[frame]])
    ln3.set_data([-w, x[frame]], [-l, y[frame]])
    ln4.set_data([w, x[frame]], [-l, y[frame]])
    ln1.set_data(x[frame], y[frame])

    l1 = (x[frame]+w)**2+(y[frame]+l)**2
    l2 = (x[frame]-w)**2+(y[frame]+l)**2

    if l1 > l01:
        ln3.set_color(darkred)
    else:
        ln3.set_color(darkgreen)
    if l2 > l02:
        ln4.set_color(darkred)
    else:
        ln4.set_color(darkgreen)

    return ln1, ln2, ln3, ln4,


ani = FuncAnimation(fig, update, frames=np.arange(
    0, len(soln.t)), init_func=init, interval=1000/fps)  # , blit=True)
# plt.show()
matplotlib.use("Agg")
ani.save(filename='sim.mp4', fps=fps, dpi=quality)
