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
m1 = 1.0
m2 = 2.0
m3 = 3.0
l1 = 2.0
l2 = 2.0
l3 = 2.0

# Visual Parameters
fps = 60
quality = 300  # DPI of animation

# Initial Conditions
#  V = [v1, v2, v3, dv1, dv2, dv3]
# dV = [dv1, dv2, dv3, ddv1, ddv2, ddv3]
V0 = np.array([np.pi, -0.5*np.pi, 0.5*np.pi, 0, 0, 0])


def dVF(t, V):
    v1 = V[0]
    v2 = V[1]
    v3 = V[2]
    dv1 = V[3]
    dv2 = V[4]
    dv3 = V[5]

    ddv1 = (m3*(m2+m3)*(-np.cos(v1-v2)*np.cos(v2-v3)+np.cos(v1-v3))*(g*np.sin(v3)-l1*np.sin(v1-v3)*(dv1)**2-l2*np.sin(v2-v3)*(dv2)**2)+(-m3*np.cos(v1-v3)*np.cos(v2-v3)+(m2+m3)*np.cos(v1-v2))*(g*m2*np.sin(v2)+g*m3*np.sin(v2)-l1*m2*np.sin(v1-v2)*(dv1)**2-l1*m3*np.sin(v1-v2)*(dv1)**2+l3*m3*np.sin(v2-v3)*(dv3)**2)-(m2-m3*np.cos(v2-v3)** 2+m3)*(g*m1*np.sin(v1)+g*m2*np.sin(v1)+g*m3*np.sin(v1)+l2*m2*np.sin(v1-v2)*(dv2)**2+l2*m3*np.sin(v1-v2)*(dv2)**2+l3*m3*np.sin(v1-v3)*(dv3)**2))/(l1*(2*m3*(m2+m3)*np.cos(v1-v2)*np.cos(v1-v3)*np.cos(v2-v3)-m3*(m2+m3)*np.cos(v1-v3)**2-m3*(m1+m2+m3)*np.cos(v2-v3)**2-(m2+m3)**2*np.cos(v1-v2)**2+(m2+m3)*(m1+m2+m3)))
    ddv2 = (-m3*((m2+m3)*np.cos(v1-v2)*np.cos(v1-v3)-(m1+m2+m3)*np.cos(v2-v3))*(g*np.sin(v3)-l1*np.sin(v1-v3)*(dv1)**2-l2*np.sin(v2-v3)*(dv2)**2)+(-m3*np.cos(v1-v3)*np.cos(v2-v3)+(m2+m3)*np.cos(v1-v2))*(g*m1*np.sin(v1)+g*m2*np.sin(v1)+g*m3*np.sin(v1)+l2*m2*np.sin(v1-v2)*(dv2)**2+l2*m3*np.sin(v1-v2)*(dv2)**2+l3*m3*np.sin(v1-v3)*(dv3)** 2)-(m1+m2-m3*np.cos(v1-v3)**2+m3)*(g*m2*np.sin(v2)+g*m3*np.sin(v2)-l1*m2*np.sin(v1-v2)*(dv1)**2-l1*m3*np.sin(v1-v2)*(dv1)**2+l3*m3*np.sin(v2-v3)*(dv3)**2))/(l2*(2*m3*(m2+m3)*np.cos(v1-v2)*np.cos(v1-v3)*np.cos(v2-v3)-m3*(m2+m3)*np.cos(v1-v3)**2-m3*(m1+m2+m3)*np.cos(v2-v3)**2-(m2+m3)**2*np.cos(v1-v2)**2+(m2+m3)*(m1+m2+m3)))
    ddv3 = ((m2+m3)*(-np.cos(v1-v2)*np.cos(v2-v3)+np.cos(v1-v3))*(g*m1*np.sin(v1)+g*m2*np.sin(v1)+g*m3*np.sin(v1)+l2*m2*np.sin(v1-v2)*(dv2)**2+l2*m3*np.sin(v1-v2)*(dv2)**2+l3*m3*np.sin(v1-v3)*(dv3)**2)-(m2+m3)*(g*np.sin(v3)-l1*np.sin(v1-v3)*(dv1)**2-l2*np.sin(v2-v3)*(dv2)**2)*(m1+m2+m3-(m2+m3)*np.cos(v1-v2)**2)-((m2+m3)*np.cos(v1-v2)* np.cos(v1-v3)-(m1+m2+m3)*np.cos(v2-v3))*(g*m2*np.sin(v2)+g*m3*np.sin(v2)-l1*m2*np.sin(v1-v2)*(dv1)**2-l1*m3*np.sin(v1-v2)*(dv1)**2+l3*m3*np.sin(v2-v3)*(dv3)**2))/(l3*(2*m3*(m2+m3)*np.cos(v1-v2)*np.cos(v1-v3)*np.cos(v2-v3)-m3*(m2+m3)*np.cos(v1-v3)**2-m3*(m1+m2+m3)*np.cos(v2-v3)**2-(m2+m3)**2*np.cos(v1-v2)**2+(m2+m3)*(m1+m2+m3)))

    return np.array([dv1, dv2, dv3, ddv1, ddv2, ddv3])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))


v1 = soln.y[0, :]
v2 = soln.y[1, :]
v3 = soln.y[2, :]
dv1 = soln.y[3, :]
dv2 = soln.y[4, :]
dv3 = soln.y[5, :]

x1 = l1*np.sin(v1)
y1 = -l1*np.cos(v1)
x2 = x1 + l2*np.sin(v2)
y2 = y1 + - l2*np.cos(v2)
x3 = x2 + l3*np.sin(v3)
y3 = y2 + - l3*np.cos(v3)
dx1 = l1*dv1*np.cos(v1)
dy1 = l1*dv1*np.sin(v1)
dx2 = dx1 + l2*dv2*np.cos(v2)
dy2 = dy1 + l2*dv2*np.sin(v2)
dx3 = dx2 + l3*dv3*np.cos(v3)
dy3 = dy2 + l3*dv3*np.sin(v3)

print(f'Starting Simulation At: {datetime.datetime.now()}')

print('\nParameters')
print(f'm_1 = {m1}')
print(f'm_2 = {m2}')
print(f'm_3 = {m3}')
print(f'l_1 = {l1}')
print(f'l_2 = {l2}')
print(f'l_3 = {l3}')
print(f'fps = {fps}')
print(f'DPI = {quality}')

print(f'\nInitial Conditions')
print(f'theta_1 = {V0[0]}')
print(f'theta_2 = {V0[1]}')
print(f'theta_3 = {V0[2]}')
print(f'dtheta_1 = {V0[3]}')
print(f'dtheta_2 = {V0[4]}')
print(f'dtheta_3 = {V0[5]}')

# # Phase Plots
###############################################################################
# plt.figure()
# plt.suptitle('Phase Plots', fontsize=48)

# plt.subplot(221)
# plt.plot(x1, m1*dx1, color='red', label=r'$p_{1,x}\mathrm{ vs. }x_1$')
# plt.xlabel(r'$x_1$', fontsize=16)
# plt.ylabel(r'$p_{1,x}$', fontsize=16)
# plt.title(r'$p_{1,x}\mathrm{ vs. }x_1$', fontsize=32)

# plt.subplot(222)
# plt.plot(y1, m1*dy1, color='gold', label=r'$p_{1,y}\mathrm{ vs. }y_1$')
# plt.xlabel(r'$y_1$', fontsize=16)
# plt.ylabel(r'$p_{1,y}$', fontsize=16)
# plt.title(r'$p_{1,y}\mathrm{ vs. }y_1$', fontsize=32)

# plt.subplot(223)
# plt.plot(x2, m2*dx2, color='blue', label=r'$p_{2,x}\mathrm{ vs. }x_2$')
# plt.xlabel(r'$x_2$', fontsize=16)
# plt.ylabel(r'$p_{2,x}$', fontsize=16)
# plt.title(r'$p_{2,x}\mathrm{ vs. }x_2$', fontsize=32)

# plt.subplot(224)
# plt.plot(y2, m2*dy2, 'darkorange', label=r'$p_{2,y}\mathrm{ vs. }y_2$')
# plt.xlabel(r'$y_2$', fontsize=16)
# plt.ylabel(r'$p_{2,y}$', fontsize=16)
# plt.title(r'$p_{2,y}\mathrm{ vs. }y_2$', fontsize=32)
# plt.show()
###############################################################################


# # Poincare Plots
###############################################################################
# plt.figure()
# plt.suptitle('Poincare Plots', fontsize=48)

# plt.subplot(221)
# plt.plot(x1[:-1], x1[1:], color='red')
# plt.xlabel(r'$x_{1,i}$', fontsize=16)
# plt.ylabel(r'$x_{1,i+1}$', fontsize=16)
# plt.title(r'$x_{1,i+1}\mathrm{ vs. }x_{1,i}$', fontsize=32)

# plt.subplot(222)
# plt.plot(y1[:-1], y1[1:], color='gold')
# plt.xlabel(r'$y_{1,i}$', fontsize=16)
# plt.ylabel(r'$y_{1,i+1}$', fontsize=16)
# plt.title(r'$y_{1,i+1}\mathrm{ vs. }y_{1,i}$', fontsize=32)

# plt.subplot(223)
# plt.plot(x2[:-1], x2[1:], color='blue')
# plt.xlabel(r'$x_{2,i}$', fontsize=16)
# plt.ylabel(r'$x_{2,i+1}$', fontsize=16)
# plt.title(r'$x_{2,i+1}\mathrm{ vs. }x_{2,i}$', fontsize=32)

# plt.subplot(224)
# plt.plot(y2[:-1], y2[1:], 'darkorange')
# plt.xlabel(r'$y_{2,i}$', fontsize=16)
# plt.ylabel(r'$y_{2,i+1}$', fontsize=16)
# plt.title(r'$y_{2,i+1}\mathrm{ vs. }y_{2,i}$', fontsize=32)

# plt.show()
###############################################################################

fig, ax = plt.subplots()
fig.suptitle("Triple Pendulum", fontsize=12)
#     x1data, y1data = [], []
#     x2data, y2data = [], []
#     x3data, y3data = [], []
ln4, = plt.plot([], [], 'k')
ln5, = plt.plot([], [], 'k')
ln6, = plt.plot([], [], 'k')
ln1, = plt.plot([], [], 'ro', markersize=10+2*m1)
ln2, = plt.plot([], [], 'bo', markersize=10+2*m2)
ln3, = plt.plot([], [], 'go', markersize=10+2*m3)


def init():
    ax.set_xlim(1.1*(-l1-l2-l3), 1.1*(l1+l2+l3))
    ax.set_ylim(1.1*(-l1-l2-l3), 1.1*(l1+l2+l3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ln1, ln2, ln3, ln4, ln5, ln6,


def update(frame):
    #     x1data.append(x1[frame])
    #     y1data.append(y1[frame])
    #     x2data.append(x2[frame])
    #     y2data.append(y2[frame])
    #     x3data.append(x3[frame])
    #     y3data.append(y3[frame])

    ln4.set_data([0, x1[frame]], [0, y1[frame]])
    ln5.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    ln6.set_data([x2[frame], x3[frame]], [y2[frame], y3[frame]])
    ln1.set_data(x1[frame], y1[frame])
    ln2.set_data(x2[frame], y2[frame])
    ln3.set_data(x3[frame], y3[frame])

    return ln1, ln2, ln3, ln4, ln5, ln6,


ani = FuncAnimation(fig, update, frames=np.arange(0, len(soln.t)),
                    init_func=init, interval=1000/fps)  # , blit=True)
# plt.show()
matplotlib.use("Agg")
ani.save(filename='sim.mp4', fps=fps, dpi=quality)

# del '\\left \\{
# del \\right \\}'
# del {\\left (t \\right )}
# repl frac{d^2}{d t^2}' ' -> dd
# repl frac{d}{d t}' ' -> d
# replregex ,' '\\quad' ' -> \n
# repl : -> =
# repl sin -> np.sin
# repl cos -> np.cos
# repl sqrt -> np.sqrt
# repl exp -> np.exp
# repl frac ->
# repl ^{2} -> **2
#     For each quantity '~'
#############################
# repl {(~)} -> (~)
# repl sin**2(~) -> sin(~)**2
# repl cos**2(~) -> cos(~)**2
#############################
# repl _{1} -> 1
# repl _{2} -> 2
# repl _{3} -> 3
# repl theta -> v
# repl phi -> p
# del \\
# del left
# del right
# repl ' '' ' -> ' '
# repl ' ') -> )
# repl (' ' -> (
# repl ' '} -> }
# repl {' ' -> {
# repl ' '-' ' -> -
# repl ' '+' ' -> +
# repl -' ' -> -
# repl +' ' -> +
# repl ) ( -> )*(
# repl }{ -> )/(
# repl { -> (
# repl } -> )
# repl ' ' -> *