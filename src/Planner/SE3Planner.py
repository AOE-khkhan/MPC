from cvxopt import matrix, log, exp, solvers
from src.Planner.HelperFuncs import *
import numpy as np

nodes = 10
deltaT = 0.1
T = 2.0

mass = 18.0
I = np.array([[0.5, 0.0, 0.0], [0.0, 0.34, 0.0], [0.0, 0.0, 0.15]])
Iinv = np.linalg.inv(I)

nQP = int(T / deltaT)
x0 = np.array([[0.0],[0.0],[0.5]])
v0 = np.array([[0.0],[0.0],[0.0]])
f0 = np.array([[0.0],[0.0],[0.0]])
R0 = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
w0 = np.array([[1.0],[2.0],[3.0]])
cop0 = np.array([[0.0], [0.0], [0.0]])

x = [x0 for i in range(nQP)]
v = [v0 for i in range(nQP)]
R = [R0 for i in range(nQP)]
w = [w0 for i in range(nQP)]
cop = [cop0 for i in range(nQP)]
f = [f0 for i in range(nQP)]

delx = np.array([[0.0], [0.0], [0.0]])
delv = np.array([[0.0], [0.0], [0.0]])
delw = np.array([[0.0], [0.0], [0.0]])
delR = np.array([[0.0], [0.0], [0.0]])
Jx = np.array([[1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0]])
Jv = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass]])

for i in range(nQP  - 1):
    im = nQP - i - 1
    errx = x[im] - x[im - 1] - deltaT * v[im - 1]
    errv = v[im] - v[im - 1] - deltaT / mass * f[im - 1]
    M = getMomentumAboutCoM(x[im - 1], cop[im - 1], f[im - 1])
    errw = w[im] - w[im - 1] - deltaT * Iinv * M - getAngularMomentumInstability(w[im - 1], I)
    errR = getLogDiff(R[im], R[im - 1]) - w[im - 1] * deltaT
    objx = delx + errx
    objv = delv + errv
    objw = delw + errw
    objR = delR + errR

    pass
