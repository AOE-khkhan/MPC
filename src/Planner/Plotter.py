from matplotlib import pyplot as plt
import numpy as np

sampling = 10
naxis = 3
xaxis = 0
yaxis = 1
zaxis = 2
def plotResults3D(f, m, n, mass, deltaT, x0, v0, g):
    fX = np.zeros((n, 1))
    fY = np.zeros((n, 1))
    fZ = np.zeros((n, 1))
    mX = np.zeros((n, 1))
    mY = np.zeros((n, 1))
    mz = np.zeros((n, 1))

    for i in range(n):
        fX[i, 0] = f[naxis * i + xaxis]
        fY[i, 0] = f[naxis * i + yaxis]
        fZ[i, 0] = f[naxis * i + zaxis]

        mX[i, 0] = m[naxis * i + xaxis]
        mY[i, 0] = m[naxis * i + yaxis]
        mz[i, 0] = m[naxis * i + zaxis]
        pass

    tData = getTimeData(n, deltaT)
    fxData, vxData, pxData = getData(deltaT, fX, m, mass, n, v0[xaxis], x0[xaxis])
    fyData, vyData, pyData = getData(deltaT, fY, m, mass, n, v0[yaxis], x0[yaxis])
    fzData, vzData, pzData = getData(deltaT, fZ, m, mass, n, v0[zaxis], x0[zaxis])

    for i in range(tData.size):
        fxData[i] -= mass * g[xaxis]
        fyData[i] -= mass * g[yaxis]
        fzData[i] -= mass * g[zaxis]

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
    pass

def plotResults(f, m, n, mass, deltaT, x0, v0):
    tData = getTimeData(n, deltaT)
    fData, vData, xData = getData(deltaT, f, m, mass, n, v0, x0)
    fig = plt.figure()
    fPlot = fig.add_subplot(311)
    fPlot.plot(tData, fData, color='red', marker='o', linestyle='dashed',
        linewidth=1, markersize=1)
    fPlot.grid()
    vPlot = fig.add_subplot(312)
    vPlot.plot(tData, vData, color='green', marker='o', linestyle='dashed',
        linewidth=1, markersize=1)
    vPlot.grid()
    xPlot = fig.add_subplot(313)
    xPlot.plot(tData, xData, color='blue', marker='o', linestyle='dashed',
        linewidth=1, markersize=1)
    xPlot.grid()
    fig.show()
    pass


def getData(deltaT, f, m, mass, n, v0, x0):
    fData = getForceData(f, m, n, deltaT)
    vData = getVelocityData(f, m, n, mass, deltaT, v0)
    xData = getPositionData(f, m, n, mass, deltaT, x0, v0)
    return fData, vData, xData


def getTimeData(n, deltaT):
    tDataList = [[] for i in range(n-1)]
    for i in range(n - 1):
        coeffs = np.vstack((0, 1))
        tDataList[i] = generateData(coeffs, i * deltaT, (i + 1) * deltaT, sampling)
    fData = np.vstack(tDataList)
    return fData

def getForceData(f, m, n, deltaT):
    fDataList = [[] for i in range(n-1)]
    for i in range(n - 1):
        f0 = f[i]
        f1 = f[i + 1]
        m0 = m[i]
        m1 = m[i + 1]
        coeffs = np.vstack(((f0) / (deltaT**0),
                            (m0) / (deltaT ** 1),
                            (-3 * f0 + 3 * f1 -2 * m0 - m1) / (deltaT ** 2),
                            (m0 + m1 - 2 * f1 + 2 * f0) / (deltaT ** 3)))
        fDataList[i] = generateData(coeffs, 0, deltaT, sampling)
    fData = np.vstack(fDataList)
    return fData


def getVelocityData(f, m, n, mass, deltaT, v0):
    vDataList = [[] for i in range(n-1)]
    vi = v0
    for i in range(n - 1):
        f0 = f[i]
        f1 = f[i + 1]
        m0 = m[i]
        m1 = m[i + 1]
        coeffs = np.vstack(((vi * mass) / (deltaT**0),
                            (f0) / (deltaT**0),
                            (m0) / (2 * deltaT ** 1),
                            (-3 * f0 + 3 * f1 -2 * m0 - m1) / (3 * deltaT ** 2),
                            (m0 + m1 - 2 * f1 + 2 * f0) / (4 * deltaT ** 3))) / mass
        vDataList[i] = generateData(coeffs, 0, deltaT, sampling)
        vi = vDataList[i][sampling-1, 0]
    vData = np.vstack(vDataList)
    return vData

def getPositionData(f, m, n, mass, deltaT, x0, v0):
    xDataList = [[] for i in range(n-1)]
    xi = x0
    vi = v0
    for i in range(n - 1):
        f0 = f[i]
        f1 = f[i + 1]
        m0 = m[i]
        m1 = m[i + 1]
        coeffs = np.vstack(((xi * mass) / (deltaT**0),
                            (vi * mass) / (deltaT**0),
                            (f0) / (2 * 1 * deltaT**0),
                            (m0) / (3 * 2 * deltaT ** 1),
                            (-3 * f0 + 3 * f1 -2 * m0 - m1) / (4 * 3 * deltaT ** 2),
                            (m0 + m1 - 2 * f1 + 2 * f0) / (5 * 4 * deltaT ** 3))) / mass
        xDataList[i] = generateData(coeffs, 0, deltaT, sampling)
        vi = vi + deltaT * (6 * f0 + 6 * f1 - m1 + m0) / (12 * mass)
        xi = xDataList[i][sampling-1, 0]
    xData = np.vstack(xDataList)
    return xData

def generateData(coeffs, tmin, tmax, nPoints):
    data = np.zeros((nPoints, 1))
    for i in range(nPoints):
        t = tmin + (tmax - tmin) * i / (nPoints)
        data[i,0] = 0.0
        for j in range(coeffs.size):
            data[i] += np.power(t, j) * coeffs[j, 0]
    return data

def plotDataCollocationPlanner(nodes, nodeT, D, nx, nv, nu, i):
    nNode = 2 * nx + nv + nu
    tSoln = []
    xSoln = []
    zSoln = []
    vxSoln = []
    vzSoln = []
    axSoln = []
    azSoln = []
    axpSoln = []
    azpSoln = []
    uSoln = []
    vSoln = []
    xErrSoln = []
    zErrSoln = []
    MySoln = []
    plotDt = 0.001
    for j in range(nodes - 1):
        tInitial = nodeT[j]
        tFinal = nodeT[j + 1]
        coeffX = D[j * nNode: j * nNode + nx]
        coeffZ = D[j * nNode + nx: j * nNode + 2 * nx]
        coeffV = D[j * nNode + 2 * nx: j * nNode + 2 * nx + nv]
        coeffU = D[j * nNode + 2 * nx + nv: j * nNode + 2 * nx + nv + nu]
        pointsToCalc = int((tFinal - tInitial) / plotDt)
        for l in range(pointsToCalc):
            tSoln.append((tInitial + plotDt * l))
            x = 0.0
            xd = 0.0
            xdd = 0.0
            for k in range(nx):
                x += coeffX[k] * (plotDt * l) ** k
                if k - 2 >= 0:
                    xdd += coeffX[k] * (plotDt * l) ** (k - 2) * k * (k - 1)
                if k - 1 >= 0:
                    xd += coeffX[k] * (plotDt * l) ** (k - 1) * k
                pass
            xSoln.append(x.item((0)))
            vxSoln.append(xd.item((0)))
            z = 0.0
            zd = 0.0
            zdd = 0.0
            for k in range(nx):
                z += coeffZ[k] * (plotDt * l) ** k
                if k - 2 >= 0:
                    zdd += coeffZ[k] * (plotDt * l) ** (k - 2) * k * (k - 1)
                if k - 1 >= 0:
                    zd += coeffZ[k] * (plotDt * l) ** (k - 1) * k
                pass
            zSoln.append(z.item((0)))
            vzSoln.append(zd.item((0)))
            v = 0.0
            for k in range(nv):
                v += coeffV[k] * (plotDt * l) ** k
                pass
            vSoln.append(v.item((0)))
            u = 0.0
            for k in range(nu):
                u += coeffU[k] * (plotDt * l) ** k
                pass
            uSoln.append(u.item((0)))
            ax = u * (x - v)
            az = u * (z - 0) - 9.81
            xErr = xdd - ax
            zErr = zdd - az
            My = (zdd + 9.81) * (x - v) - xdd * (z - 0)
            xErrSoln.append(xErr.item((0)))
            zErrSoln.append(zErr.item((0)))
            axSoln.append(ax.item((0)))
            azSoln.append(az.item((0)))
            axpSoln.append(xdd.item((0)))
            azpSoln.append(zdd.item((0)))
            MySoln.append(My.item((0)))
        pass
    pass
    rows = 6
    cols = 2
    fig = plt.figure()
    fig.suptitle("Iteration " + str(i))
    plotX = fig.add_subplot(rows, cols, 1)
    plotX.plot(tSoln, xSoln)
    plotX.plot(tSoln, vSoln, 'r--')
    plotX.set_ylabel("X")
    plotX.grid()
    for tval in nodeT:
        plotX.axvline(x=tval, color="red", linewidth=0.2)
    plotZ = fig.add_subplot(rows, cols, 2)
    plotZ.plot(tSoln, zSoln)
    plotZ.set_ylabel("Z")
    plotZ.grid()
    for tval in nodeT:
        plotZ.axvline(x=tval, color="red", linewidth=0.2)
    plotV = fig.add_subplot(rows, cols, 3)
    plotV.plot(tSoln, vSoln)
    plotV.set_ylabel("V")
    plotV.grid()
    for tval in nodeT:
        plotV.axvline(x=tval, color="red", linewidth=0.2)
    plotU = fig.add_subplot(rows, cols, 4)
    plotU.plot(tSoln, uSoln)
    plotU.set_ylabel("U")
    plotU.grid()
    for tval in nodeT:
        plotU.axvline(x=tval, color="red", linewidth=0.2)
    plotVx = fig.add_subplot(rows, cols, 5)
    plotVx.plot(tSoln, vxSoln)
    plotVx.set_ylabel("Xdot")
    plotVx.grid()
    for tval in nodeT:
        plotVx.axvline(x=tval, color="red", linewidth=0.2)
    plotVz = fig.add_subplot(rows, cols, 6)
    plotVz.plot(tSoln, vzSoln)
    plotVz.set_ylabel("Zddot")
    plotVz.grid()
    for tval in nodeT:
        plotVz.axvline(x=tval, color="red", linewidth=0.2)
    plotAx = fig.add_subplot(rows, cols, 7)
    plotAx.plot(tSoln, axSoln)
    plotAx.plot(tSoln, axpSoln, 'r--')
    plotAx.set_ylabel("Xddot")
    plotAx.grid()
    for tval in nodeT:
        plotAx.axvline(x=tval, color="red", linewidth=0.2)
    plotAz = fig.add_subplot(rows, cols, 8)
    plotAz.plot(tSoln, azSoln)
    plotAz.plot(tSoln, azpSoln, 'r--')
    plotAz.set_ylabel("Zddot")
    plotAz.grid()
    for tval in nodeT:
        plotAz.axvline(x=tval, color="red", linewidth=0.2)
    plotXErr = fig.add_subplot(rows, cols, 9)
    plotXErr.plot(tSoln, xErrSoln)
    plotXErr.set_ylabel("XErr")
    plotXErr.grid()
    for tval in nodeT:
        plotXErr.axvline(x=tval, color="red", linewidth=0.2)
    plotZErr = fig.add_subplot(rows, cols, 10)
    plotZErr.plot(tSoln, zErrSoln)
    plotZErr.set_ylabel("ZErr")
    plotZErr.grid()
    for tval in nodeT:
        plotZErr.axvline(x=tval, color="red", linewidth=0.2)
    plotMy = fig.add_subplot(rows, cols, 11)
    plotMy.plot(tSoln, MySoln)
    plotMy.set_ylabel("My")
    plotMy.grid()
    for tval in nodeT:
        plotMy.axvline(x=tval, color="red", linewidth=0.2)
    fig.show()
