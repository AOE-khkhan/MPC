from cvxopt import matrix, log, exp, solvers
from matplotlib import pyplot as plt
from src.Planner.HelperFuncs import *
import numpy as np
solvers.options['show_progress'] = True

nodes = 10
deltaT = 0.1
T = 2.0
gz = -9.81

mass = 18.0
I = np.array([[0.5, 0.0, 0.0], [0.0, 0.34, 0.0], [0.0, 0.0, 0.15]])
Iinv = np.linalg.inv(I)

nQP = int(T / deltaT)
x0 = np.array([[0.0], [0.0], [0.40]])
v0 = np.array([[0.0], [0.0], [0.0]])
f0 = np.array([[0.0], [0.0], [0.0]])
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
w0 = np.array([[0.0], [0.0], [0.0]])
cop0 = np.array([[0.0], [0.0], [0.0]])

r = [x0 for i in range(nQP)]
v = [v0 for i in range(nQP)]
R = [R0 for i in range(nQP)]
w = [w0 for i in range(nQP)]
cop = [cop0 for i in range(nQP)]
f = [f0 for i in range(nQP)]

Jx = np.array([[1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
Jv = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
JR = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0]])

fd = [f0 for i in range(nQP)]
for i in range(4):
    fd[i + 7][2] = mass * gz

# Forward pass
for i in range(nQP - 1):
    r[i + 1] = r[i] + deltaT * v[i]
    v[i + 1] = v[i] + deltaT / mass * fd[i]
    R[i + 1] = np.matmul(getRotationMatrix(w[i], deltaT), R[i])
    M = getMomentumAboutCoM(r[i], cop[i], fd[i])
    w[i + 1] = w[i] + deltaT * np.matmul(Iinv, M + getAngularMomentumInstability(w[i], I))
pass

xf = x0
vf = v0
wf = w0
Rf = R0

delx = xf - r[nQP - 1]
delv = vf - v[nQP - 1]
delw = wf - w[nQP - 1]
delR = getLogDiff(Rf, R[nQP - 1])

Wx = np.identity(3) * 1000
Wv = np.identity(3) * 50
WR = np.identity(3) * 1
Ww = np.identity(3) * 1

# Backward pass
for i in range(nQP - 1):
    im = nQP - i - 1
    errx = r[im] - r[im - 1] - deltaT * v[im - 1]
    errv = v[im] - v[im - 1] - deltaT / mass * f[im - 1]
    M = getMomentumAboutCoM(r[im - 1], cop[im - 1], f[im - 1])
    errw = w[im] - w[im - 1] - deltaT * np.matmul(Iinv, M - getAngularMomentumInstability(w[im - 1], I))
    errR = getLogDiff(R[im], R[im - 1]) - w[im - 1] * deltaT
    objx = delx + errx
    print(delx)
    objv = delv + errv
    objw = delw + errw
    objR = delR + errR
    Ixx = I.item((0, 0))
    Iyy = I.item((1, 1))
    Izz = I.item((2, 2))
    Fx = f[im - 1].item(0)
    Fy = f[im - 1].item(1)
    Fz = f[im - 1].item(2)
    x = r[im - 1].item(0)
    y = r[im - 1].item(1)
    z = r[im - 1].item(2)
    xcop = cop[im - 1].item(0)
    ycop = cop[im - 1].item(1)
    zcop = cop[im - 1].item(2)
    Jw = np.array([[0.0, -Fz, Fy, 0.0, 0.0, 0.0, 0.0, z - zcop, ycop - y, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, Fz, -Fy],
                   [Fz, 0.0, -Fx, 0.0, 0.0, 0.0, zcop - z, 0.0, x - xcop, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -Fz, 0.0, Fx],
                   [-Fy, 0.0, Fx, 0.0, 0.0, 0.0, y - ycop, 0.0, xcop - x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, Fy, 0.0, -Fx]]) * deltaT
    Jw[0, :] *= Ixx
    Jw[1, :] *= Iyy
    Jw[2, :] *= Izz
    Q = np.identity(18) * 0.0001
    F = np.zeros((18, 1))
    Qd, fd = getQuadProgCostFunction(Jx, objx, Wx)
    Q +=  Qd
    F += fd
    Qd, fd = getQuadProgCostFunction(Jv, objv, Wv)
    Q += Qd
    F += fd
    Qd, fd = getQuadProgCostFunction(Jw, objw, Ww)
    Q += Qd
    F += fd
    Qd, fd = getQuadProgCostFunction(JR, objR, WR)
    Q += Qd
    F += fd
    P = matrix(Q, tc='d')
    q = matrix(F, tc='d')
    #G = matrix(Ain, tc='d')
    #h = matrix(Bin, tc='d')

    soln = solvers.qp(P, q) #, G, h)
    dOpt = soln['x']
    delx = dOpt[0:3]
    delv = dOpt[3:6]
    delf = dOpt[6:9]
    delR = dOpt[9:12]
    delw = dOpt[12:15]
    delCoP = dOpt[15:18]
    f[im - 1] += delf
    cop[im - 1] += delCoP
    pass

# Forward pass
for i in range(nQP - 1):
    r[i + 1] = r[i] + deltaT * v[i]
    v[i + 1] = v[i] + deltaT / mass * f[i]
    R[i + 1] = np.matmul(getRotationMatrix(w[i], deltaT), R[i])
    M = getMomentumAboutCoM(r[i], cop[i], f[i])
    w[i + 1] = w[i] + deltaT * np.matmul(Iinv, M + getAngularMomentumInstability(w[i], I))
pass

t = [0.1 * i for i in range(nQP)]
rz = [r[i].item(2) for i in range(nQP)]
plt.plot(t, rz)
plt.show()