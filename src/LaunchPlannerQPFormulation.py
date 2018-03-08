import numpy as np
from cvxopt import solvers
from cvxopt import  matrix
from src import Plotter

# This code generates a trajectory for linear centroidal dynamics by optimizing
# the knot points for a cubic spline interpolated force trajectory to minimize the
# difference between the desired centroidal trajectory while ensuring that certain
# constraints on the final state are met
#
# defining the number of knot points.

nodes = 4
# decision variables [f0, m0, f1, m1, f2, m2, f3, m3]
# fi = force value mi = derivative of force

# initial states [x0 v0]
x0 = 0.75 # m
v0 = 0.0 # m/s
# final states [xf vf]  typically want to impose xf as a bound rather than a value
# vf is the critical parameter that must be achieved
hDes = 0.35 # m
g = -9.81 # m/s^2
mass = 18 #kg
xf = 1.0; vf = np.sqrt(-2.0 * g * hDes) # m/s
deltaT = 1.0 / nodes #s

# Calculate the unknown state variables in terms of the decision variables. These are the transformation matrices
# The initial conditions add on linearly as a constant (the constant is the same for the velocity matrix. i.e. v0
# but for the position the constant is i*deltaT*v0 + x0)
vi = np.zeros((nodes, 2 * nodes))
xi = np.zeros((nodes, 2 * nodes))

veloMultiplier = deltaT / (12 * mass)
for i in range(nodes - 1):
    vi[i + 1, 2 * i] = 6 * veloMultiplier
    vi[i + 1, 2 * i + 1] = 1 * veloMultiplier
    vi[i + 1, 2 * i + 2] = 6 * veloMultiplier
    vi[i + 1, 2 * i + 3] = -1 * veloMultiplier
    pass

for i in range(1, nodes):
    vi[i,:] = vi[i,:] + vi[i - 1,:]
    pass

posMultiplier = deltaT**2 / (60 * mass)
for i in range(nodes - 1):
    xi[i + 1, 2 * i] = 21 * posMultiplier
    xi[i + 1, 2 * i + 1] = 3 * posMultiplier
    xi[i + 1, 2 * i + 2] = 9 * posMultiplier
    xi[i + 1, 2 * i + 3] = -2 * posMultiplier
    xi[i + 1, :] = xi[i + 1,:] + deltaT * vi[i,:] + xi[i,:]
    pass

# Setup the constraints for the nodes, these are the min and max values for each
# Ideally want the set the initial and final forces and slopes (figure this out)
fmin = mass * g
fmax = mass * g - mass * g * 1.5 # assuming robot can lift 2 times its weight
mmin = -2
mmax = 2
xmax = 0.8
xmin = 0.5

AinForce = np.zeros((4 * nodes, 2 * nodes))
binForce = np.zeros((4 * nodes, 1))
for i in range(nodes):
    AinForce[4 * i, 2 * i] = 1;
    binForce[4 * i, 0] = fmax

    AinForce[4 * i + 1, 2 * i] = -1;
    binForce[4 * i + 1, 0] = -fmin

    AinForce[4 * i + 2, 2 * i + 1] = 1;
    binForce[4 * i + 2, 0] = mmax

    AinForce[4 * i + 3, 2 * i + 1] = -1;
    binForce[4 * i + 3, 0] = -mmin

AinPositionMax = xi
binPositionMax = np.ones((nodes, 1)) * (xmax - x0)
AinPositionMin = -xi
binPositionMin = -np.ones((nodes, 1)) * (xmin - x0)
for i in range(nodes):
    binPositionMax[i, 0] -= i * deltaT * v0
    binPositionMin[i, 0] += i * deltaT * v0

Ain = np.vstack((AinForce, AinPositionMax, AinPositionMin))
bin = np.vstack((binForce, binPositionMax, binPositionMin))

Aeq = np.zeros((4, 2 * nodes))
Aeq[0, 0] = 1
Aeq[1, 1] = 1
Aeq[2, 2 * nodes - 2] = 1
Aeq[3, 2 * nodes - 1] = 1
beq = np.zeros((4, 1))

## Setup the objective function
## This is basically the desired states, typically x and the final v
xnominal = 0.0;
xdes = np.zeros((nodes, 1))

## Add the entire x trajectory to the objcetive
J = xi
c = xdes
## Adding the final v value to the objective
Jvf = vi[nodes - 1, :]
cvf = vf - v0

## Normailze the forces so that they are reduced as much as possible
Jforce = np.identity( 2 * nodes)
cforce = np.zeros(( 2 * nodes, 1))

J = np.vstack((J, Jvf, Jforce))
c = np.vstack((c, cvf, cforce))

W = np.identity(nodes * 3 + 1)
W[nodes, nodes] *= 3 * nodes * 100000000
Jt = J.transpose()
JtW = Jt.dot(W)
Q = JtW.dot(J)
f = -JtW.dot(c)

w,  v = np.linalg.eig(Q)
P = matrix(Q, tc='d')
q = matrix(f, tc='d')
G = matrix(Ain, tc='d')
h = matrix(bin, tc='d')
A = matrix(Aeq, tc='d')
b = matrix(beq, tc='d')

soln = solvers.qp(P, q, G, h, A, b)
fOpt = soln['x']

vOpt = vi.dot(fOpt) + v0
xOpt = xi.dot(fOpt)
for i in range(nodes):
    xOpt[i]  += x0 + i * deltaT * v0

print("Force profile:")
print(fOpt)
print("Position profile:")
print(xOpt)
print("Velocity profile: ", vf)
print(vOpt)

fVals = np.zeros((nodes, 1))
mVals = np.zeros((nodes, 1))
for i in range(nodes):
    fVals[i, 0] = fOpt[2*i, 0]
    mVals[i, 0] = fOpt[2*i + 1, 0]

Plotter.plotResults(fVals, mVals, nodes, mass, deltaT, x0, v0)