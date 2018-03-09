import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from numpy.matlib import zeros

from src import Plotter

# This code generates a trajectory for linear centroidal dynamics by optimizing
# the knot points for a cubic spline interpolated force trajectory to minimize the
# difference between the desired centroidal trajectory while ensuring that certain
# constraints on the final state are met
#
# defining the number of knot points.

nodes = 4
naxis = 3
xaxis = 0
yaxis = 1
zaxis = 2

# decision variables [f1, m1, .... f(nodes - 2), m(nodes - 2)]
# fi = force value mi = derivative of force
# The first and last knot points are considered fixed and removed from the optimization

# initial states [x0 v0]
x0 = np.array([[0.0], [0.0], [0.75]])  # m
v0 = np.array([[0.0], [0.0], [0.0]])  # m/s
# final states [xf vf]  typically want to impose xf as a bound rather than a value
# vf is the critical parameter that must be achieved
hDes = 0.15  # m
g = np.array([[0.0], [0.0], [-9.81]])  # m/s^2
mass = 18  # kg
# xf = 1.0
vf = np.array([[0.0], [0.2], [np.sqrt(-2.0 * g[zaxis] * hDes)]])  # m/s
# Jump timing
tLaunch = 1.0
deltaT = tLaunch / nodes  # s

mu = 0.6
fmin = mass * g * 0
fmax = -mass * g * 2.0  # assuming robot can lift 2 times its weight
mmin = np.array([[-2.0], [-2.0], [-2.0]])
mmax = np.array([[2.0], [2.0], [2.0]])
xmax = np.array([[0.1], [0.1], [0.8]])
xmin = np.array([[-0.1], [-0.1], [0.5]])

xdes = np.zeros((naxis * (nodes - 1), 1))
amp = np.array([[0.0],[0.0],[0.0]])
for i in range(nodes - 1):
    xdes[naxis * i: naxis * (i + 1)] = x0 # - amp * np.sin(np.pi * i / (nodes - 2))
    pass

f0 = - mass * g
m0 = np.array([[0.0], [0.0], [0.0]])
ff = - mass * g
mf = np.array([[0.0], [0.0], [0.0]])

# Calculate the unknown state variables in terms of the decision variables. These are the transformation matrices
# The initial conditions add on linearly as a constant (the constant is the same for the velocity matrix. i.e. v0
# but for the position the constant is i*deltaT*v0 + x0)
vi = np.zeros((naxis * nodes, naxis * 2 * (nodes - 2)))
pi = np.zeros((naxis * nodes, naxis * 2 * (nodes - 2)))

veloMultiplier = deltaT / (12 * mass)

delv0 = (6 * f0 + 1 * m0) * veloMultiplier
delvf = (6 * ff - 1 * mf) * veloMultiplier
delvg = deltaT * g

for k in range(naxis):
    vi[(0 + 1) * naxis + k, (2 * 0 + 0) * naxis + k] = 6 * veloMultiplier
    vi[(0 + 1) * naxis + k, (2 * 0 + 1) * naxis + k] = -1 * veloMultiplier
    for i in range(1, nodes - 2):
        vi[(i + 1) * naxis + k, (2 * i - 2) * naxis + k] = 6 * veloMultiplier
        vi[(i + 1) * naxis + k, (2 * i - 1) * naxis + k] = 1 * veloMultiplier
        vi[(i + 1) * naxis + k, (2 * i + 0) * naxis + k] = 6 * veloMultiplier
        vi[(i + 1) * naxis + k, (2 * i + 1) * naxis + k] = -1 * veloMultiplier
        pass
    vi[(nodes - 2 + 1) * naxis + k, (2 * (nodes - 2) - 2) * naxis + k] = 6 * veloMultiplier
    vi[(nodes - 2 + 1) * naxis + k, (2 * (nodes - 2) - 1) * naxis + k] = 1 * veloMultiplier
    pass

for k in range(naxis):
    for i in range(1, nodes):
        vi[i * naxis + k, :] += vi[(i - 1) * naxis + k, :]
        pass
    pass

posMultiplier = deltaT ** 2 / (60 * mass)
delx0 = (21 * f0 + 3 * m0) * posMultiplier
delxf = (9 * ff - 2 * mf) * posMultiplier
delxg = deltaT ** 2 * g / 2

for k in range(naxis):
    pi[(0 + 1) * naxis + k, (2 * 0 + 0) * naxis + k] = 9 * posMultiplier
    pi[(0 + 1) * naxis + k, (2 * 0 + 1) * naxis + k] = -2 * posMultiplier
    pi[(0 + 1) * naxis + k, :] += deltaT * vi[0 * naxis + k, :] + pi[0 * naxis + k, :]
    for i in range(1, nodes - 2):
        pi[(i + 1) * naxis + k, (2 * i - 2) * naxis + k] = 21 * posMultiplier
        pi[(i + 1) * naxis + k, (2 * i - 1) * naxis + k] = 3 * posMultiplier
        pi[(i + 1) * naxis + k, (2 * i + 0) * naxis + k] = 9 * posMultiplier
        pi[(i + 1) * naxis + k, (2 * i + 1) * naxis + k] = -2 * posMultiplier
        pi[(i + 1) * naxis + k, :] += deltaT * vi[i * naxis + k, :] + pi[i * naxis + k, :]
        pass
    pi[((nodes - 2) + 1) * naxis + k, (2 * (nodes - 2) - 2) * naxis + k] = 21 * posMultiplier
    pi[((nodes - 2) + 1) * naxis + k, (2 * (nodes - 2) - 1) * naxis + k] = 3 * posMultiplier
    pi[((nodes - 2) + 1) * naxis + k, :] += deltaT * vi[(nodes - 2) * naxis + k, :] + pi[(nodes - 2) * naxis + k, :]
    pass

## Setup the constraints for the nodes

# Set force limit constraints
# Since we are also imposing the friction constraint this can be written only in terms of the z force
# Reduces the total number of constraints
forceMultiplier = 2 + 2 * naxis
AinForce = np.zeros((forceMultiplier * (nodes - 2), naxis * 2 * (nodes - 2)))
binForce = np.zeros((forceMultiplier * (nodes - 2), 1))
for i in range(nodes - 2):
    AinForce[forceMultiplier * i, (2 * i) * naxis + zaxis] = 1 + mu
    binForce[forceMultiplier * i, 0] = fmax[zaxis, 0]

    AinForce[forceMultiplier * i + 1, (2 * i) * naxis + zaxis] = -1
    binForce[forceMultiplier * i + 1, 0] = -fmin[zaxis, 0]

    for j in range(naxis):
        AinForce[forceMultiplier * i + 2 * j + 2, (2 * i + 1) * naxis + j] = 1
        binForce[forceMultiplier * i + 2 * j + 2, 0] = mmax[j]
        AinForce[forceMultiplier * i + 2 * j + 3, (2 * i + 1) * naxis + j] = -1
        binForce[forceMultiplier * i + 2 * j + 3, 0] = -mmin[j]
        pass
    pass

# Setup the friction constraints at the nodes
AinFriction = np.zeros((4 * (nodes - 2), naxis * 2 * (nodes - 2)))
binFriction = np.zeros((4 * (nodes - 2), 1))
for i in range(nodes - 2):
    for k in range(4):
        AinFriction[4 * i + k, naxis * 2 * i + xaxis] = np.power(-1, int(k / 2))
        AinFriction[4 * i + k, naxis * 2 * i + yaxis] = np.power(-1, k)
        AinFriction[4 * i + k, naxis * 2 * i + zaxis] = -mu
        pass
    pass

# Setup the position limit constraints
AinPositionMax = pi[naxis:, :]
AinPositionMin = -pi[naxis:, :]
binPositionMax = np.zeros((naxis * (nodes - 1), 1))  # * (xmax - x0 - delx0 - delxg)
binPositionMin = np.zeros((naxis * (nodes - 1), 1))  # * (xmin - x0 - delx0 - delxg)
for i in range(1, nodes):
    binPositionMax[naxis * (i - 1): naxis * i, :] = +(
                xmax - x0 - delx0 - i * deltaT * v0 - (i - 1) * deltaT * delv0 - i * i * delxg)
    binPositionMin[naxis * (i - 1): naxis * i, :] = -(
                xmin - x0 - delx0 - i * deltaT * v0 - (i - 1) * deltaT * delv0 - i * i * delxg)
    pass
binPositionMax[naxis * (nodes - 2): naxis * (nodes - 1), :] -= delxf
binPositionMin[naxis * (nodes - 2): naxis * (nodes - 1), :] += delxf

# Compile all the constraints
Ain = np.vstack((AinForce, AinFriction, AinPositionMax, AinPositionMin))
Bin = np.vstack((binForce, binFriction, binPositionMax, binPositionMin))

## Setup the objective function
tasks = 0

# Add the entire x trajectory to the objective
J = pi[naxis * 1:, :]
c = xdes
for i in range(1, nodes):
    c[naxis * (i - 1): naxis * i, :] -= x0 + delx0 + i * deltaT * v0 + (i - 1) * deltaT * delv0 + (i * i) * delxg
c[naxis * (nodes - 2): naxis * (nodes - 1), :] -= delxf
tasks += naxis * (nodes - 1)

# Adding the final v value to the objective
Jvf = vi[(nodes - 1) * naxis: nodes * naxis, :]
cvf = vf - v0 - delv0 - delvf - (nodes - 1) * delvg
tasks += naxis

# Normalize the forces so that they are reduced as much as possible
Jforce = np.identity(naxis * 2 * (nodes - 2))
cforce = np.zeros((naxis * 2 * (nodes - 2), 1))
tasks += naxis * 2 * (nodes - 2)

J = np.vstack((J, Jvf, Jforce))
c = np.vstack((c, cvf, cforce))

W = np.identity(tasks)
for k in range(naxis):
    for i in range(nodes - 2):
        W[nodes*naxis + 2 * i * naxis + k] *= 1 / 10000000000000
        W[nodes*naxis + (2 * i + 1) * naxis + k] *= 1 / 100
        pass
    pass

Jt = J.transpose()
JtW = Jt.dot(W)
Q = JtW.dot(J)
f = -JtW.dot(c)

w, v = np.linalg.eig(Q)
P = matrix(Q, tc='d')
q = matrix(f, tc='d')
G = matrix(Ain, tc='d')
h = matrix(Bin, tc='d')

soln = solvers.qp(P, q, G, h)
fOpt = soln['x']
fVals = np.zeros((naxis * nodes, 1))
mVals = np.zeros((naxis * nodes, 1))
fVals[0: naxis, :] = f0 + mass * g
mVals[0: naxis, :] = m0
for i in range(1, nodes - 1):
    fVals[i * naxis: (i + 1) * naxis, :] = fOpt[2 * (i - 1) * naxis: 2 * (i - 1) * naxis + naxis, :] + mass * g
    mVals[i * naxis: (i + 1) * naxis, :] = fOpt[2 * (i - 1) * naxis + naxis: 2 * i * naxis, :]
fVals[(nodes - 1) * naxis: nodes * naxis, :] = ff + mass * g
mVals[(nodes - 1) * naxis: nodes * naxis, :] = mf

Plotter.plotResults3D(fVals, mVals, nodes, mass,deltaT, x0, v0, g)

# vOpt = vi.dot(fOpt)
# for i in range(nodes):
#     vOpt[naxis * i: naxis * (i + 1), :] += v0 + delv0 + i * delvg
# vOpt[naxis * 0: naxis * (0 + 1), :] -= delv0
# vOpt[naxis * (nodes - 1): naxis * ((nodes - 1) + 1), :] += delvf
# print(vOpt)
# exit()

# xOpt = pi.dot(fOpt)
# xOpt[0, 0] += x0
# for i in range(1, nodes):
#     xOpt[i] += x0 + delx0 + i * deltaT * v0 + (i - 1) * deltaT * delv0
# xOpt[nodes - 1, 0] += delxf
# print("Force profile:")
# print(fOpt)
# print("Position profile:")
# print(xOpt)
# print("Velocity profile: ", vf)
# print(vOpt)
