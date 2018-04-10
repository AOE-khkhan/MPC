from builtins import print
from cvxopt import matrix, log, exp, solvers
import numpy as np
from scipy.linalg import expm, logm, det

nodes = 10
deltaT = 0.1
T = 2.0

mass = 18.0
I = np.array([[0.5, 0.0, 0.0], [0.0, 0.34, 0.0], [0.0, 0.0, 0.15]])

nQP = int(T / deltaT)
x0 = np.array([[0.0],[0.0],[0.5]])
v0 = np.array([[0.0],[0.0],[0.0]])
f0 = np.array([[0.0],[0.0],[0.0]])
R0 = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
w0 = np.array([[0.0],[0.0],[0.0]])

x = [x0 for i in range(nQP)]
v = [v0 for i in range(nQP)]
f = [f0 for i in range(nQP)]
R = [R0 for i in range(nQP)]
w = [w0 for i in range(nQP)]


delx = np.array([[0.0], [0.0], [0.0]])
delv = np.array([[0.0], [0.0], [0.0]])

Jx = np.array([[1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0]])
Jv = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass])


for i in range(nQP  - 1):
    im = nQP - i - 1
    errx = x[im] - x[im - 1] - deltaT * v[im - 1]
    errv = v[im] - v[im - 1] - deltaT / mass * f[im - 1]
    objx = delx + errx
    objv = delv + errv

    pass
