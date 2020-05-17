from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime


# Constants
g = 9.81
tf = 10
t0 = 0.0

# Simulation Parameters
M = 40
m = 1
r0 = 1
L = 4
d = 4

# Visual Parameters
fps = 60
quality = 300  # DPI of animation

# Initial Conditions
#  V = [r, v, dr, dv]
# dV = [dr, dv, ddr, ddv]
V0 = np.array([r0, (0)*np.pi, 0, 0])


def dVF(t, V):
    r = V[0]
    v = V[1]
    dr = V[2]
    dv = V[3]

    ddv = -(g*np.sin(v)+2*dv*dr)/r
    ddr = (M*g+m*g*np.cos(v)+m*r*dv**2)/(m+M)

    return np.array([dr, dv, ddr, ddv])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))


r = soln.y[0, :]
v = soln.y[1, :]
dr = soln.y[2, :]
dv = soln.y[3, :]

x = d + r*np.sin(v)
y = -r*np.cos(v)
dx = r*dv*np.cos(v)
dy = r*dv*np.sin(v)
# L= l+r -> l=L-r
Y = -(L-r)
dY = dr

print(datetime.datetime.now())
print('\nParameters')
print(f'm = {m}')
print(f'M = {M}')
print(f'L = {L}')
print(f'd = {d}')
print(f'fps = {fps}')
print(f'DPI = {quality}')

print(f'\nInitial Conditions')
print(f'r = {V0[0]}')
print(f'theta = {V0[1]}')
print(f'dr = {V0[2]}')
print(f'dtheta = {V0[3]}')


fig, ax = plt.subplots()
ax.set_xlim(-L, L+d)
ax.set_ylim(-L, L)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.suptitle("Pendulum Connected to a Hanging Mass", fontsize=12)

ln3, = plt.plot([], [], 'k')
ln4, = plt.plot([], [], 'k')
ln5, = plt.plot([], [], 'k')
ln1, = plt.plot([], [], 'rs', markersize=10+2*M)
ln2, = plt.plot([], [], 'bo', markersize=10+2*m)

# tabletop = matplotlib.patches.Rectangle((overhang, -0.125), d-2*overhang, 0.125, color='c')
# ax.add_patch(tabletop)

def update(frame):
    ln3.set_data([d, x[frame]], [0, y[frame]])
    ln4.set_data([0, 0], [0, Y[frame]])
    ln5.set_data([0, d], [0, 0])
    ln2.set_data(x[frame], y[frame])
    ln1.set_data(0, Y[frame])
    # ax.add_patch(tabletop)
    return ln1, ln2, ln3, ln4, ln5,


ani = FuncAnimation(fig, update, frames=np.arange(
    0, len(soln.t)), interval=1000/fps)  # , blit=True)
plt.show()
# matplotlib.use("Agg")
# ani.save(filename='sim.mp4', fps=fps, dpi=quality)
