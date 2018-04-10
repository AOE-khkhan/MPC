from builtins import print
from cvxopt import matrix, log, exp, solvers
from matplotlib import pyplot as plt
from src.Planner.HelperFuncs import *
import numpy as np
solvers.options['show_progress'] = False

nodes = 10
deltaT = 0.1
T = 2.0
gz = -9.81

mass = 18.0
I = np.array([[0.5, 0.0, 0.0], [0.0, 0.34, 0.0], [0.0, 0.0, 0.15]])
Iinv = np.linalg.inv(I)

rmin = np.array([[-0.1], [-0.1], [0.3]])
rmax = np.array([[0.1], [0.1], [0.5]])
fmin = np.array([[-10.0], [-10.0], [mass * gz]])
fmax = np.array([[10.0], [10.0], [mass * -gz]])

nQP = int(T / deltaT)
x0 = np.array([[0.0], [0.0], [0.40]])
v0 = np.array([[0.0], [0.0], [0.0]])
f0 = np.array([[0.0], [0.0], [0.0]])
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
w0 = np.array([[0.0], [0.0], [0.0]])
cop0 = np.array([[0.0], [0.0], [0.0]])

r = [np.copy(x0) for i in range(nQP)]
v = [np.copy(v0) for i in range(nQP)]
R = [np.copy(R0) for i in range(nQP)]
w = [np.copy(w0) for i in range(nQP)]
cop = [np.copy(cop0) for i in range(nQP)]
f = [np.copy(f0) for i in range(nQP)]

Jx = np.array([[1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
Jv = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT / mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
JR = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, deltaT, 0.0, 0.0, 0.0]])
Jf = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
fd = [np.copy(f0) for i in range(nQP)]
for i in range(4):
    fd[i + 7][2] = mass * gz

# Initial forward pass
for i in range(nQP - 1):
    r[i + 1] = r[i] + deltaT * v[i]
    v[i + 1] = v[i] + deltaT / mass * fd[i]
    R[i + 1] = np.matmul(getRotationMatrix(w[i], deltaT), R[i])
    M = getMomentumAboutCoM(r[i], cop[i], fd[i])
    w[i + 1] = w[i] + deltaT * np.matmul(Iinv, M + getAngularMomentumInstability(w[i], I))
pass

xf = np.copy(x0)
vf = np.copy(v0)
wf = np.copy(w0)
Rf = np.copy(R0)

delx = xf - r[nQP - 1]
delv = vf - v[nQP - 1]
delw = wf - w[nQP - 1]
delR = getLogDiff(Rf, R[nQP - 1])

Wx = np.identity(3) * 50
Wv = np.identity(3) * 50
WR = np.identity(3) * 1
Ww = np.identity(3) * 1
Wf = np.identity(3) * 0.005

Ainx = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
Ainf = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
nIterations = 2
fz = [0 for i in range(nIterations)]

for j in range(nIterations):
    # Backward pass
    for i in range(nQP - 1):
        im = nQP - i - 1
        errx = r[im] - r[im - 1] - deltaT * v[im - 1]
        errv = v[im] - v[im - 1] - deltaT / mass * f[im - 1]
        M = getMomentumAboutCoM(r[im - 1], cop[im - 1], f[im - 1])
        errw = w[im] - w[im - 1] - deltaT * np.matmul(Iinv, M - getAngularMomentumInstability(w[im - 1], I))
        errR = getLogDiff(R[im], R[im - 1]) - w[im - 1] * deltaT
        objx = delx + errx
        objv = delv + errv
        objw = delw + errw
        objR = delR + errR
        objf = np.array([[0.0], [0.0], [0.0]])
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
        Jw[0, :] *= 1.0/Ixx
        Jw[1, :] *= 1.0/Iyy
        Jw[2, :] *= 1.0/Izz
        Q = np.identity(18) * 0.000001
        F = np.zeros((18, 1))
        Qd, fd = getQuadProgCostFunction(Jf, objf, Wf)
        Q +=  Qd
        F += fd
        Qd, fd = getQuadProgCostFunction(Jx, objx, Wx)
        Q +=  Qd
        F += fd
        binx = np.vstack((rmax - r[im], -rmin + r[im]))
        Qd, fd = getQuadProgCostFunction(Jv, objv, Wv)
        Q += Qd
        F += fd
        Qd, fd = getQuadProgCostFunction(Jw, objw, Ww)
        Q += Qd
        F += fd
        Qd, fd = getQuadProgCostFunction(JR, objR, WR)
        Q += Qd
        F += fd
        binf = np.vstack((fmax - f[im], -fmin + f[im]))
        Ain = np.vstack((Ainx, Ainf))
        Bin = np.vstack((binx, binf))
        P = matrix(Q, tc='d')
        q = matrix(F, tc='d')
        G = matrix(Ain, tc='d')
        h = matrix(Bin, tc='d')

        soln = solvers.qp(P, q, G, h)
        dOpt = soln['x']
        delx = np.array(dOpt[0:3])
        delv = np.array(dOpt[3:6])
        delf = np.array(dOpt[6:9])
        delR = np.array(dOpt[9:12])
        delw = np.array(dOpt[12:15])
        delCoP = np.array(dOpt[15:18])
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
    fz[j] = f[nQP - 1].item(2)
pass

tData = [0.1 * i for i in range(nQP)]
pxData = [r[i].item(0) for i in range(nQP)]
pyData= [r[i].item(1) for i in range(nQP)]
pzData = [r[i].item(2) for i in range(nQP)]
vxData = [v[i].item(0) for i in range(nQP)]
vyData= [v[i].item(1) for i in range(nQP)]
vzData = [v[i].item(2) for i in range(nQP)]
fxData = [f[i].item(0) for i in range(nQP)]
fyData = [f[i].item(1) for i in range(nQP)]
fzData = [f[i].item(2) for i in range(nQP)]

fig = plt.figure()
fxPlot = fig.add_subplot(331)
fxPlot.plot(tData, fxData, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
fxPlot.set_ylabel('ForceX')
fxPlot.grid()
fyPlot = fig.add_subplot(332)
fyPlot.plot(tData, fyData, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
fyPlot.set_ylabel('ForceY')
fyPlot.grid()
fzPlot = fig.add_subplot(333)
fzPlot.plot(tData, fzData, color='red', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
fzPlot.set_ylabel('ForceZ')
fzPlot.grid()

vxPlot = fig.add_subplot(334)
vxPlot.plot(tData, vxData, color='green', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
vxPlot.set_ylabel('VelocityX')
vxPlot.grid()
vyPlot = fig.add_subplot(335)
vyPlot.plot(tData, vyData, color='green', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
vyPlot.set_ylabel('VelocityY')
vyPlot.grid()
vzPlot = fig.add_subplot(336)
vzPlot.plot(tData, vzData, color='green', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
vzPlot.set_ylabel('VelocityZ')
vzPlot.grid()

pxPlot = fig.add_subplot(337)
pxPlot.plot(tData, pxData, color='blue', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
pxPlot.set_xlabel('time')
pxPlot.set_ylabel('PositionX')
pxPlot.grid()
pyPlot = fig.add_subplot(338)
pyPlot.plot(tData, pyData, color='blue', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
pyPlot.set_xlabel('time')
pyPlot.set_ylabel('PositionY')
pyPlot.grid()
pzPlot = fig.add_subplot(339)
pzPlot.plot(tData, pzData, color='blue', marker='o', linestyle='dashed',
            linewidth=1, markersize=1)
pzPlot.set_xlabel('time')
pzPlot.set_ylabel('PositionZ')
pzPlot.grid()
fig.show()

fig = plt.figure()
conPlot = fig.add_subplot(111)
n = [i for i in range(nIterations)]
conPlot.plot(n, fz)
fig.show()
