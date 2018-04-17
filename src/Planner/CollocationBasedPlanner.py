from cvxopt import solvers, matrix
import numpy as np
import scipy as sci
from src.Planner.HelperFuncs import *
from matplotlib import pyplot as plt

h = 0.30
g = -9.81
omegaSq = -g / h
T = 0.1
v = 0.2
dF = 0.025
# Decision variables [x0, x1, x2, x3, v0, v1, v2, v3]

T0 = 1.0
T1 = T
T2 = T*T
T3 = T*T*T

objJ = np.array([[T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [T0, T1, T2, T3, 0.0, 0.0, 0.0, 0.0],
                 [0.0, T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, T0, 2.0 * T1, 3.0 * T2, 0.0, 0.0, 0.0, 0.0]])
objC = np.array([[0.008],
                 [T*v/2.0],
                 [0.0],
                 [0.0]])

dynJ = np.array([[0.0, 0.0, 0.0, omegaSq, 0.0, 0.0, 0.0, -omegaSq],
                 [0.0, 0.0, omegaSq, 0.0, 0.0, 0.0, -omegaSq, 0.0],
                 [0.0, -omegaSq, 0.0, 6.0, 0.0, omegaSq, 0.0, 0.0],
                 [-omegaSq, 0.0, 2.0, 0.0, omegaSq, 0.0, 0.0, 0.0]])
dynC = np.array([[0.0],
                 [0.0],
                 [0.0],
                 [0.0]])

copJ = np.array([[0.0, 0.0, 0.0, 0.0, T0, 0.0 * T1, 0.0 * T2, 0.0 * T3],
                 [0.0, 0.0, 0.0, 0.0, T0, (0.25 ** 1) * T1, (0.25 **2) * T2, (0.25 ** 3) * T3],
                 [0.0, 0.0, 0.0, 0.0, T0, (0.50 ** 1) * T1, (0.50 **2) * T2, (0.50 ** 3) * T3],
                 [0.0, 0.0, 0.0, 0.0, T0, (0.75 ** 1) * T1, (0.75 **2) * T2, (0.75 ** 3) * T3],
                 [0.0, 0.0, 0.0, 0.0, T0, (1.00 ** 1) * T1, (1.00 **2) * T2, (1.00 ** 3) * T3]])

objW = np.identity(4)

H, f = getQuadProgCostFunction(objJ, objC, objW)
Aeq = dynJ
beq = dynC
Ain = np.vstack((copJ, -copJ))
bin = np.ones((10,1)) * dF

P = matrix(H, tc='d')
q = matrix(f, tc='d')
G = matrix(Ain, tc='d')
h = matrix(bin, tc='d')
A = matrix(Aeq, tc='d')
b = matrix(beq, tc='d')

soln = solvers.qp(P, q, G, h, A, b)
optX = soln['x']

coeffX = optX[0:4]
coeffV = optX[4:8]

xInitial = coeffX[0]
xFinal = coeffX[0] + coeffX[1] * T1 + coeffX[2] * T2 + coeffX[3] * T3
xdInitial = coeffX[1]
xdFinal = coeffX[1] + 2 * coeffX[2] * T1 + 3 * coeffX[3] * T2

print(xInitial)
print(xFinal)
print(xdInitial)
print(xdFinal)

cop = np.matmul(copJ, optX)
print(cop)

deltaT = 0.001
xRange = int(T / deltaT)
t = [deltaT * i for i in range(xRange)]
x = [coeffX[0] + coeffX[1] * dt + coeffX[2] * dt**2 + coeffX[3] * dt**3 for dt in t]
copX = [coeffV[0] + coeffV[1] * dt + coeffV[2] * dt**2 + coeffV[3] * dt**3 for dt in t]
xddot = [coeffX[2] * 2 + coeffX[3] * 6 * dt for dt in t]
cErr = [xddot[i] - omegaSq * (x[i] - copX[i]) for i in range(xRange)]

fig = plt.figure()
plotX = fig.add_subplot(311)
plotX.plot(t, x, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
plotX.set_ylabel('X')
plotX.grid()
plotV = fig.add_subplot(312)
plotV.plot(t, copX, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
plotV.set_ylabel('copX')
plotV.grid()

plotErr = fig.add_subplot(313)
plotErr.plot(t, cErr, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
plotErr.set_ylabel('Err')
plotErr.grid()
fig.show()