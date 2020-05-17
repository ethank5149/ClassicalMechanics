import vpython as v
import numpy as np
# from numba import jit


# ddp1 = (-g*m1*np.sin(v1)*np.cos(p1)+(k*l0*l2*np.sin(p1)*np.cos(p2)-k*l0*l2*np.sin(p2)*np.cos(v1-v2)*np.cos(p1))/(np.sqrt(l1**2-2*l1*l2*np.sin(p1)*np.sin(p2) * np.cos(v1-v2)-2*l1*l2*np.cos(p1)*np.cos(p2)+l2**2))-k*l2*np.sin(p1)*np.cos(p2)+k*l2*np.sin(p2)*np.cos(v1-v2)*np.cos(p1)+l1*m1*dv1**2*np.sin(p1)*np.cos(p1))/(l1*m1)
# ddp2 = (-g*m2*np.sin(v2)*np.cos(p2)-(k*l0*l1*np.sin(p1)*np.cos(v1-v2)*np.cos(p2)+k*l0*l1*np.sin(p2)*np.cos(p1))/(np.sqrt(l1**2-2*l1*l2*np.sin(p1)*np.sin(p2) * np.cos(v1-v2)-2*l1*l2*np.cos(p1)*np.cos(p2)+l2**2))+k*l1*np.sin(p1)*np.cos(v1-v2)*np.cos(p2)-k*l1*np.sin(p2)*np.cos(p1)+l2*m2*dv2**2*np.sin(p2)*np.cos(p2))/(l2*m2)
# ddv1 = (-(g*m1*np.cos(v1))/(np.sin(p1))+(k*l0*l2*np.sin(v1-v2)*np.sin(p2))/(np.sqrt(l1**2-2*l1*l2*np.sin(p1)*np.sin(p2)*np.cos(v1-v2) - 2*l1*l2*np.cos(p1)*np.cos(p2)+l2**2)*np.sin(p1))-(k*l2*np.sin(v1-v2)*np.sin(p2))/(np.sin(p1))-(2*l1*m1*dp1*dv1)/(np.tan(p1)))/(l1*m1)
# ddv2 = (-(g*m2*np.cos(v2))/(np.sin(p2))-(k*l0*l1*np.sin(v1-v2)*np.sin(p1))/(np.sqrt(l1**2-2*l1*l2*np.sin(p1)*np.sin(p2)*np.cos(v1-v2) - 2*l1*l2*np.cos(p1)*np.cos(p2)+l2**2)*np.sin(p2))+(k*l1*np.sin(v1-v2)*np.sin(p1))/(np.sin(p2))-(2*l2*m2*dp2*dv2)/(np.tan(p2)))/(l2*m2)


# @jit
def FV1(t, V1, V2, P1, P2):
    return(np.array([(-(g*m1*np.cos(V1[0]))/(np.sin(P1[0]))+(k*l0*l2*np.sin(V1[0]-V2[0])*np.sin(P2[0]))/(np.sqrt(l1**2-2*l1*l2*np.sin(P1[0])*np.sin(P2[0])*np.cos(V1[0]-V2[0]) - 2*l1*l2*np.cos(P1[0])*np.cos(P2[0])+l2**2)*np.sin(P1[0]))-(k*l2*np.sin(V1[0]-V2[0])*np.sin(P2[0]))/(np.sin(P1[0]))-(2*l1*m1*P1[1]*V1[1])/(np.tan(P1[0])))/(l1*m1), V1[1]]))


# @jit
def FV2(t, V1, V2, P1, P2):
    return(np.array([(-(g*m2*np.cos(V2[0]))/(np.sin(P2[0]))-(k*l0*l1*np.sin(V1[0]-V2[0])*np.sin(P1[0]))/(np.sqrt(l1**2-2*l1*l2*np.sin(P1[0])*np.sin(P2[0])*np.cos(V1[0]-V2[0]) - 2*l1*l2*np.cos(P1[0])*np.cos(P2[0])+l2**2)*np.sin(P2[0]))+(k*l1*np.sin(V1[0]-V2[0])*np.sin(P1[0]))/(np.sin(P2[0]))-(2*l2*m2*P2[1]*V2[1])/(np.tan(P2[0])))/(l2*m2), V2[1]]))


# @jit
def FP1(t, V1, V2, P1, P2):
    return(np.array([(-g*m1*np.sin(V1[0])*np.cos(P1[0])+(k*l0*l2*np.sin(P1[0])*np.cos(P2[0])-k*l0*l2*np.sin(P2[0])*np.cos(V1[0]-V2[0])*np.cos(P1[0]))/(np.sqrt(l1**2-2*l1*l2*np.sin(P1[0])*np.sin(P2[0]) * np.cos(V1[0]-V2[0])-2*l1*l2*np.cos(P1[0])*np.cos(P2[0])+l2**2))-k*l2*np.sin(P1[0])*np.cos(P2[0])+k*l2*np.sin(P2[0])*np.cos(V1[0]-V2[0])*np.cos(P1[0])+l1*m1*V1[1]**2*np.sin(P1[0])*np.cos(P1[0]))/(l1*m1), P1[1]]))


# @jit
def FP2(t, V1, V2, P1, P2):
    return(np.array([(-g*m2*np.sin(V2[0])*np.cos(P2[0])-(k*l0*l1*np.sin(P1[0])*np.cos(V1[0]-V2[0])*np.cos(P2[0])+k*l0*l1*np.sin(P2[0])*np.cos(P1[0]))/(np.sqrt(l1**2-2*l1*l2*np.sin(P1[0])*np.sin(P2[0]) * np.cos(V1[0]-V2[0])-2*l1*l2*np.cos(P1[0])*np.cos(P2[0])+l2**2))+k*l1*np.sin(P1[0])*np.cos(V1[0]-V2[0])*np.cos(P2[0])-k*l1*np.sin(P2[0])*np.cos(P1[0])+l2*m2*V2[1]**2*np.sin(P2[0])*np.cos(P2[0]))/(l2*m2), P2[1]]))


# @jit
def rk4(t, V1, V2, P1, P2):
    # K1
    kV11 = dt*FV1(t, V1, V2, P1, P2)
    kV21 = dt*FV2(t, V1, V2, P1, P2)
    kP11 = dt*FP1(t, V1, V2, P1, P2)
    kP21 = dt*FP2(t, V1, V2, P1, P2)
    # K2
    kV12 = dt*FV1(t+0.5*dt, V1+0.5*kV11, V2,          P1,          P2)
    kV22 = dt*FV2(t+0.5*dt, V1,          V2+0.5*kV21, P1,          P2)
    kP12 = dt*FP1(t+0.5*dt, V1,          V2,          P1+0.5*kP11, P2)
    kP22 = dt*FP2(t+0.5*dt, V1,          V2,          P1,          P2+0.5*kP21)
    # K3
    kV13 = dt*FV1(t+0.5*dt, V1+0.5*kV12, V2,          P1,          P2)
    kV23 = dt*FV2(t+0.5*dt, V1,          V2+0.5*kV22, P1,          P2)
    kP13 = dt*FP1(t+0.5*dt, V1,          V2,          P1+0.5*kP12, P2)
    kP23 = dt*FP2(t+0.5*dt, V1,          V2,          P1,          P2+0.5*kP22)
    # K4
    kV14 = dt*FV1(t+dt, V1+kV13, V2,      P1,      P2)
    kV24 = dt*FV2(t+dt, V1,      V2+kV23, P1,      P2)
    kP14 = dt*FP1(t+dt, V1,      V2,      P1+kP13, P2)
    kP24 = dt*FP2(t+dt, V1,      V2,      P1,      P2+kP23)
    # RK4 Step
    V1 = V1 + (kV11+2*kV12+2*kV13+kV14)/6
    V2 = V2 + (kV21+2*kV22+2*kV23+kV24)/6
    P1 = P1 + (kP11+2*kP12+2*kP13+kP14)/6
    P2 = P2 + (kP21+2*kP22+2*kP23+kP24)/6
    return(V1, V2, P1, P2)


# Constants
global g, tf, dt
g, tf, t, dt = 9.81, 100, 0, 0.0001

# Parameters
global m1, m2, l0, l1, l2, k
m1, m2 = 2, 1
l1, l2 = 5, 7
l0, k = l1*0.2*np.pi, 100
barwidth = 0.25

# Initial Conditions
V1 = np.array([-1.4*np.pi, 0])
V2 = np.array([-1.6*np.pi, 0])
P1 = np.array([0.5*np.pi, 0])
P2 = np.array([0.5*np.pi, 0])


# Display Objects
scene = v.canvas(title='Double Pendulum', width=1200, height=600,
                 center=v.vector(0, 0, 0), background=v.color.black)
# kinetic = v.gcurve(color=v.color.blue, label='Total Kinetic Energy')
# potential = v.gcurve(color=v.color.cyan, label='Total Potential Energy')
# total = v.gcurve(color=v.color.red, label='Total Energy')

# Dynamic Objects

x1 = l1*np.sin(P1[0])*np.cos(V1[0])
y1 = l1*np.sin(P1[0])*np.sin(V1[0])
z1 = l1*np.cos(P1[0])
x2 = l2*np.sin(P2[0])*np.cos(V2[0])
y2 = l2*np.sin(P2[0])*np.sin(V2[0])
z2 = l2*np.cos(P2[0])

pendula1 = v.sphere(pos=v.vector(x1, y1, z1), radius=0.25 +
                    0.1*m1, color=v.color.cyan, make_trail=True, retain=100)
rod1 = v.cylinder(pos=v.vector(0, 0, 0), axis=pendula1.pos,
                  radius=0.125, color=v.color.white)
pendula2 = v.sphere(pos=v.vector(x2, y2, z2), radius=0.25 +
                    0.1*m2, color=v.color.red, make_trail=True, retain=1000)
rod2 = v.cylinder(pos=v.vector(0, 0, 0), axis=pendula2.pos,
                  radius=0.125, color=v.color.white)
spring = v.helix(pos=pendula1.pos, axis=pendula2.pos-pendula1.pos,
                 radius=0.25, thickness=0.25/5, color=v.color.white)
up = v.arrow(pos=v.vector(0, 0, 0), axis=v.vector(0, 1, 0))

while t < tf:
    v.rate(1/dt)

    V1, V2, P1, P2 = rk4(t, V1, V2, P1, P2)

    x1 = l1*np.sin(P1[0])*np.cos(V1[0])
    y1 = l1*np.sin(P1[0])*np.sin(V1[0])
    z1 = l1*np.cos(P1[0])
    x2 = l2*np.sin(P2[0])*np.cos(V2[0])
    y2 = l2*np.sin(P2[0])*np.sin(V2[0])
    z2 = l2*np.cos(P2[0])

    # kinetic.plot(t, T(X1, X2, Y1, Y2, Z1, Z2))
    # potential.plot(t, V(X1, X2, Y1, Y2, Z1, Z2))
    # total.plot(t, T(X1, X2, Y1, Y2, Z1, Z2)+V(X1, X2, Y1, Y2, Z1, Z2))

    pendula1.pos = v.vector(x1, y1, z1)
    pendula2.pos = v.vector(x2, y2, z2)
    spring.pos = pendula1.pos
    spring.axis = pendula2.pos-pendula1.pos
    rod1.axis = pendula1.pos
    rod2.axis = pendula2.pos
    t += dt
