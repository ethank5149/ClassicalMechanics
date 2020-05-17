from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# Constants
g = 9.81
tf = 100
t0 = 0.0

# Simulation Parameters
m1 = 11.0
m2 = 1.0
l01 = 3.0
l02 = 3.0
k1 = 10
k2 = 10

# Visual Parameters
fps = 60
omega_graph = np.pi/8  # Speed of rotation of the graph
quality = 300  # DPI of animation

#    Initial Conditions
#    V = [x1, y1, z1, x2, y2, z2, dx1, dy1, dz1, dx2, dy2, dz2]
#    dV = [dx1, dy1, dz1, dx2, dy2, dz2, ddx1, ddy1, ddz1, ddx2, ddy2, ddz2]
V0 = np.array([1, 1, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0])


def dVF(t, V):
    x1 = V[0]
    y1 = V[1]
    z1 = V[2]
    x2 = V[3]
    y2 = V[4]
    z2 = V[5]

    # Not used in the following equations
    # dx1 = V[6]
    # dy1 = V[7]
    # dz1 = V[8]
    # dx2 = V[9]
    # dy2 = V[10]
    # dz2 = V[11]

    ddx1 = (k1*l01*x1)/(m1*np.sqrt(x1**2+y1**2+z1**2))-(k1*x1)/(m1)+(k2*l02*x1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1 **
                                                                                            2-2*z1*z2+z2**2))-(k2*l02*x2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))-(k2*x1)/(m1)+(k2*x2)/(m1)
    ddx2 = -(k2*l02*x1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))+(k2*l02*x2) / \
        (m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2 **
                    2+z1**2-2*z1*z2+z2**2))+(k2*x1)/(m2)-(k2*x2)/(m2)
    ddy1 = -g+(k1*l01*y1)/(m1*np.sqrt(x1**2+y1**2+z1**2))-(k1*y1)/(m1)+(k2*l02*y1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2 +
                                                                                               z1**2-2*z1*z2+z2**2))-(k2*l02*y2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))-(k2*y1)/(m1)+(k2*y2)/(m1)
    ddy2 = -g-(k2*l02*y1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))+(k2*l02*y2) / \
        (m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2 **
                    2+z1**2-2*z1*z2+z2**2))+(k2*y1)/(m2)-(k2*y2)/(m2)
    ddz1 = (k1*l01*z1)/(m1*np.sqrt(x1**2+y1**2+z1**2))-(k1*z1)/(m1)+(k2*l02*z1)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1 **
                                                                                            2-2*z1*z2+z2**2))-(k2*l02*z2)/(m1*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))-(k2*z1)/(m1)+(k2*z2)/(m1)
    ddz2 = -(k2*l02*z1)/(m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2**2+z1**2-2*z1*z2+z2**2))+(k2*l02*z2) / \
        (m2*np.sqrt(x1**2-2*x1*x2+x2**2+y1**2-2*y1*y2+y2 **
                    2+z1**2-2*z1*z2+z2**2))+(k2*z1)/(m2)-(k2*z2)/(m2)
    return np.array([V[6], V[7], V[8], V[9], V[10], V[11], ddx1, ddy1, ddz1, ddx2, ddy2, ddz2])


soln = solve_ivp(fun=dVF, y0=V0, method='RK45',
                 t_span=(t0, tf), t_eval=np.linspace(t0, tf, fps*(tf-t0)))

#    Remember
#    V = [x1, y1, z1, x2, y2, z2, dx1, dy1, dz1, dx2, dy2, dz2]
x1 = soln.y[0, :]
y1 = soln.y[1, :]
z1 = soln.y[2, :]
x2 = soln.y[3, :]
y2 = soln.y[4, :]
z2 = soln.y[5, :]
dx1 = soln.y[6, :]
dy1 = soln.y[7, :]
dz1 = soln.y[8, :]
dx2 = soln.y[9, :]
dy2 = soln.y[10, :]
dz2 = soln.y[11, :]

# # # Phase Plots
# ###############################################################################
plt.figure()
plt.suptitle('Phase Plots', fontsize=48)

plt.subplot(231)
plt.plot(x1, m1*dx1, color='red', label=r'$p_{1,x}\mathrm{ vs. }x_1$')
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$p_{1,x}$', fontsize=16)
plt.title(r'$p_{1,x}\mathrm{ vs. }x_1$', fontsize=32)

plt.subplot(232)
plt.plot(y1, m1*dy1, color='gold', label=r'$p_{1,y}\mathrm{ vs. }y_1$')
plt.xlabel(r'$y_1$', fontsize=16)
plt.ylabel(r'$p_{1,y}$', fontsize=16)
plt.title(r'$p_{1,y}\mathrm{ vs. }y_1$', fontsize=32)

plt.subplot(233)
plt.plot(z1, m1*dz1, color='darkgreen', label=r'$p_{1,z}\mathrm{ vs. }z_1$')
plt.xlabel(r'$z_1$', fontsize=16)
plt.ylabel(r'$p_{1,z}$', fontsize=16)
plt.title(r'$p_{1,z}\mathrm{ vs. }z_1$', fontsize=32)

plt.subplot(234)
plt.plot(x2, m2*dx2, color='blue', label=r'$p_{2,x}\mathrm{ vs. }x_2$')
plt.xlabel(r'$x_2$', fontsize=16)
plt.ylabel(r'$p_{2,x}$', fontsize=16)
plt.title(r'$p_{2,x}\mathrm{ vs. }x_2$', fontsize=32)

plt.subplot(235)
plt.plot(y2, m2*dy2, 'darkorange', label=r'$p_{2,y}\mathrm{ vs. }y_2$')
plt.xlabel(r'$y_2$', fontsize=16)
plt.ylabel(r'$p_{2,y}$', fontsize=16)
plt.title(r'$p_{2,y}\mathrm{ vs. }y_2$', fontsize=32)

plt.subplot(236)
plt.plot(z2, m2*dz2, 'mediumpurple', label=r'$p_{2,z}\mathrm{ vs. }z_2$')
plt.xlabel(r'$z_2$', fontsize=16)
plt.ylabel(r'$p_{2,z}$', fontsize=16)
plt.title(r'$p_{2,z}\mathrm{ vs. }z_2$', fontsize=32)
plt.show()
# ###############################################################################


# # Poincare Plots
###############################################################################
# plt.figure()
# plt.suptitle('Poincare Plots', fontsize=48)

# plt.subplot(231)
# plt.plot(x1[:-1], x1[1:], color='red')
# plt.xlabel(r'$x_{1,i}$', fontsize=16)
# plt.ylabel(r'$x_{1,i+1}$', fontsize=16)
# plt.title(r'$x_{1,i+1}\mathrm{ vs. }x_{1,i}$', fontsize=32)

# plt.subplot(232)
# plt.plot(y1[:-1], y1[1:], color='gold')
# plt.xlabel(r'$y_{1,i}$', fontsize=16)
# plt.ylabel(r'$y_{1,i+1}$', fontsize=16)
# plt.title(r'$y_{1,i+1}\mathrm{ vs. }y_{1,i}$', fontsize=32)

# plt.subplot(233)
# plt.plot(z1[:-1], z1[1:], color='darkgreen')
# plt.xlabel(r'$z_{1,i}$', fontsize=16)
# plt.ylabel(r'$z_{1,i+1}$', fontsize=16)
# plt.title(r'$z_{1,i+1}\mathrm{ vs. }z_{1,i}$', fontsize=32)

# plt.subplot(234)
# plt.plot(x2[:-1], x2[1:], color='blue')
# plt.xlabel(r'$x_{2,i}$', fontsize=16)
# plt.ylabel(r'$x_{2,i+1}$', fontsize=16)
# plt.title(r'$x_{2,i+1}\mathrm{ vs. }x_{2,i}$', fontsize=32)

# plt.subplot(235)
# plt.plot(y2[:-1], y2[1:], 'darkorange')
# plt.xlabel(r'$y_{2,i}$', fontsize=16)
# plt.ylabel(r'$y_{2,i+1}$', fontsize=16)
# plt.title(r'$y_{2,i+1}\mathrm{ vs. }y_{2,i}$', fontsize=32)

# plt.subplot(236)
# plt.plot(z2[:-1], z2[1:], 'mediumpurple')
# plt.xlabel(r'$z_{2,i}$', fontsize=16)
# plt.ylabel(r'$z_{2,i+1}$', fontsize=16)
# plt.title(r'$z_{2,i+1}\mathrm{ vs. }z_{2,i}$', fontsize=32)
# plt.show()
###############################################################################

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# fig.suptitle("Double Pendulum", fontsize=12)
# #    Used in conjunction with ".append" below, can be used to make a trail in the graph.
# #    Will be slow since blitting must be false.
# #    x1data, y1data, z1data = [], [], []
# #    x2data, y2data, z2data = [], [], []


# ln1, = ax.plot((x1[0],), (y1[0],), (z1[0],), 'ro', markersize=10)
# ln2, = ax.plot((x2[0],), (y2[0],), (z2[0],), 'bo', markersize=10)
# ln3, = ax.plot([0, x1[0]], [0, y1[0]], [0, z1[0]], 'k')
# ln4, = ax.plot([x1[0], x2[0]], [y1[0], y2[0]], [z1[0], z2[0]], 'k')


# def init():
#     ax.set_xlim3d(
#         [1.1*min([min(x1), min(x2)]), 1.1*max([max(x1), max(x2)])])
#     ax.set_ylim3d(
#         [1.1*min([min(y1), min(y2)]), 1.1*max([max(y1), max(y2)])])
#     ax.set_zlim3d(
#         [1.1*min([min(z1), min(z2)]), 1.1*max([max(z1), max(z2)])])
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.view_init(30, 30)
#     return ln1, ln2, ln3, ln4,


# def update(frame):
#     #    See Above
#     #    x1data.append(x1[frame])
#     #    y1data.append(y1[frame])
#     #    x2data.append(x2[frame])
#     #    y2data.append(y2[frame])

#     ln1.set_data(x1[frame], y1[frame])
#     ln2.set_data(x2[frame], y2[frame])
#     ln3.set_data([0, x1[frame]], [0, y1[frame]])
#     ln4.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])

#     ln1.set_3d_properties(z1[frame])
#     ln2.set_3d_properties(z2[frame])
#     ln3.set_3d_properties([0, z1[frame]])
#     ln4.set_3d_properties([z1[frame], z2[frame]])

#     #    Cannot get to work
#     #    ax.relim()
#     #    ax.autoscale_view(True, True, True)
#     ax.view_init(30, 30+np.degrees(omega_graph)*frame/fps)

#     textstr = '\n'.join((
#         r'Parameters',
#         r'$m_1=%s$' % m1,
#         r'$m_2=%s$' % m2,
#         r'$l_{{0,1}}=%s$' % l01,
#         r'$l_{{0,2}}=%s$' % l02,
#         r'$k_1=%s$' % k1,
#         r'$k_2=%s$' % k2,
#         r'fps=%s' % fps,
#         r'$\omega_\mathrm{Graph}}=%s$' % omega_graph))

#     ax.text2D(-0.075, 0.95, textstr, transform=ax.transAxes,
#               fontsize=6, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

#     return ln1, ln2, ln3, ln4,


# ani = FuncAnimation(fig, update, frames=np.arange(0, len(soln.t)),
#                     init_func=init, interval=1000/fps, repeat=True)  # , blit=True)

# plt.show()
# # matplotlib.use("Agg")
# # ani.save(filename='sim.mp4', fps=fps, dpi=quality)

# # .set_alpha(.7)
