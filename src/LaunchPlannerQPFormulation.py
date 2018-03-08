import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from pip.cmdoptions import no_binary

from src import Plotter

# This code generates a trajectory for linear centroidal dynamics by optimizing
# the knot points for a cubic spline interpolated force trajectory to minimize the
# difference between the desired centroidal trajectory while ensuring that certain
# constraints on the final state are met
#
# defining the number of knot points.

nodes = 4
# decision variables [f1, m1, .... f(nodes - 2), m(nodes - 2)]
# fi = force value mi = derivative of force
# The first and last knot points are considered fixed and removed from the optimization

# initial states [x0 v0]
x0 = 0.75  # m
v0 = 0.0  # m/s
# final states [xf vf]  typically want to impose xf as a bound rather than a value
# vf is the critical parameter that must be achieved
hDes = 0.35  # m
g = -9.81  # m/s^2
mass = 1 / 60.0  # kg
xf = 1.0
vf = np.sqrt(-2.0 * g * hDes)  # m/s
deltaT = 5.0 / nodes  # s

fmin = mass * g
fmax = mass * g - mass * g * 1.5  # assuming robot can lift 2 times its weight
mmin = -2
mmax = 2
xmax = 0.8
xmin = 0.5

xnominal = 0.0
xdes = np.zeros((nodes - 1, 1))

f0 = 0
m0 = 0
ff = 0
mf = 0

# Calculate the unknown state variables in terms of the decision variables. These are the transformation matrices
# The initial conditions add on linearly as a constant (the constant is the same for the velocity matrix. i.e. v0
# but for the position the constant is i*deltaT*v0 + x0)
vi = np.zeros((nodes, 2 * nodes - 4))
xi = np.zeros((nodes, 2 * nodes - 4))

veloMultiplier = deltaT / (12 * mass)

delv0 = (6 * f0 + 1 * m0) * veloMultiplier
delvf = (6 * ff - 1 * mf) * veloMultiplier

vi[0 + 1, 2 * 0 + 0] = 6 * veloMultiplier
vi[0 + 1, 2 * 0 + 1] = -1 * veloMultiplier
for i in range(1, nodes - 2):
    vi[i + 1, 2 * i - 2] = 6 * veloMultiplier
    vi[i + 1, 2 * i - 1] = 1 * veloMultiplier
    vi[i + 1, 2 * i + 0] = 6 * veloMultiplier
    vi[i + 1, 2 * i + 1] = -1 * veloMultiplier
    pass
vi[nodes - 2 + 1, 2 * (nodes - 2) - 2] = 6 * veloMultiplier
vi[nodes - 2 + 1, 2 * (nodes - 2) - 1] = 1 * veloMultiplier

for i in range(1, nodes):
    vi[i, :] = vi[i, :] + vi[i - 1, :]
    pass

posMultiplier = deltaT ** 2 / (60 * mass)
delx0 = (21 * f0 + 3 * m0) * posMultiplier
delxf = (9 * ff - 2 * mf) * posMultiplier

xi[0 + 1, 2 * 0 + 0] = 9 * posMultiplier
xi[0 + 1, 2 * 0 + 1] = -2 * posMultiplier
xi[0 + 1, :] = xi[0 + 1, :] + deltaT * vi[0, :] + xi[0, :]
for i in range(1, nodes - 2):
    xi[i + 1, 2 * i - 2] = 21 * posMultiplier
    xi[i + 1, 2 * i - 1] = 3 * posMultiplier
    xi[i + 1, 2 * i + 0] = 9 * posMultiplier
    xi[i + 1, 2 * i + 1] = -2 * posMultiplier
    xi[i + 1, :] = xi[i + 1, :] + deltaT * vi[i, :] + xi[i, :]
    pass
xi[(nodes - 2) + 1, 2 * (nodes - 2) - 2] = 21 * posMultiplier
xi[(nodes - 2) + 1, 2 * (nodes - 2) - 1] = 3 * posMultiplier
xi[(nodes - 2) + 1, :] = xi[(nodes - 2) + 1, :] + deltaT * vi[(nodes - 2), :] + xi[(nodes - 2), :]

# Setup the constraints for the nodes, these are the min and max values for each
# Ideally want the set the initial and final forces and slopes (figure this out)

AinForce = np.zeros((4 * nodes - 8 , 2 * nodes - 4))
binForce = np.zeros((4 * nodes - 8, 1))
for i in range(nodes - 2):
    AinForce[4 * i, 2 * i] = 1
    binForce[4 * i, 0] = fmax

    AinForce[4 * i + 1, 2 * i] = -1
    binForce[4 * i + 1, 0] = -fmin

    AinForce[4 * i + 2, 2 * i + 1] = 1
    binForce[4 * i + 2, 0] = mmax

    AinForce[4 * i + 3, 2 * i + 1] = -1
    binForce[4 * i + 3, 0] = -mmin

AinPositionMax = xi[1:, :]
binPositionMax = np.ones((nodes - 1, 1)) * (xmax - x0 - delx0)
binPositionMax[nodes - 2, 0] -= delxf
AinPositionMin = -xi[1:, :]
binPositionMin = -np.ones((nodes - 1, 1)) * (xmin - x0 - delx0)
binPositionMin[nodes - 2, 0] += delxf
for i in range(0, nodes - 1):
    binPositionMax[i, 0] -= i * deltaT * (v0 + delv0)
    binPositionMin[i, 0] += i * deltaT * (v0 + delv0)
#binPositionMax[nodes - 2, 0] -= deltaT * (delvf)
#binPositionMin[nodes - 2, 0] += deltaT * (delvf)

Ain = np.vstack((AinForce, AinPositionMax, AinPositionMin))
Bin = np.vstack((binForce, binPositionMax, binPositionMin))

#Aeq = np.zeros((4, 2 * nodes))
#Aeq[0, 0] = 1
#Aeq[1, 1] = 1
#Aeq[2, 2 * nodes - 2] = 1
#Aeq[3, 2 * nodes - 1] = 1
#beq = np.zeros((4, 1))

## Setup the objective function
## This is basically the desired states, typically x and the final v
xnominal = 0.0;
xdes = np.zeros((nodes -1, 1))

# Setup the objective function
# This is basically the desired states, typically x and the final v
tasks = 0
# Add the entire x trajectory to the objective
J = xi[1:,:]
c = xdes
tasks += nodes - 1
# Adding the final v value to the objective
Jvf = vi[nodes - 1, :]
cvf = vf - v0 - delv0 - delvf
tasks += 1
# Normalize the forces so that they are reduced as much as possible
Jforce = np.identity(2 * nodes - 4)
cforce = np.zeros((2 * nodes - 4, 1))
tasks += 2 * nodes - 4

J = np.vstack((J, Jvf, Jforce))
c = np.vstack((c, cvf, cforce))

W = np.identity(tasks)
W[nodes - 1, nodes - 1] *= 3 * nodes * 100000000
Jt = J.transpose()
JtW = Jt.dot(W)
Q = JtW.dot(J)
f = -JtW.dot(c)

w, v = np.linalg.eig(Q)
P = matrix(Q, tc='d')
q = matrix(f, tc='d')
G = matrix(Ain, tc='d')
h = matrix(Bin, tc='d')
#A = matrix(Aeq, tc='d')
#b = matrix(beq, tc='d')

#soln = solvers.qp(P, q, G, h, A, b)
soln = solvers.qp(P, q, G, h)
fOpt = soln['x']

vOpt = vi.dot(fOpt) + v0
xOpt = xi.dot(fOpt)
for i in range(nodes):
    xOpt[i] += x0 + i * deltaT * v0

print("Force profile:")
print(fOpt)
print("Position profile:")
print(xOpt)
print("Velocity profile: ", vf)
print(vOpt)

fVals = np.zeros((nodes, 1))
mVals = np.zeros((nodes, 1))
fVals[0, 0] = f0
mVals[0, 0] = m0
for i in range(1, nodes - 1):
    fVals[i, 0] = fOpt[2 * i - 2, 0]
    mVals[i, 0] = fOpt[2 * i - 1, 0]
fVals[nodes - 1, 0] = ff
mVals[nodes - 1, 0] = mf

Plotter.plotResults(fVals, mVals, nodes, mass, deltaT, x0, v0)
