from matplotlib import pyplot as plt
import numpy as np

sampling = 10

def plotResults(f, m, n, mass, deltaT, x0, v0):
    tData = getTimeData(n, deltaT)
    fData = getForceData(f, m, n, deltaT)
    plt.plot(tData, fData)
    plt.show()
    pass

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

def generateData(coeffs, tmin, tmax, nPoints):
    data = np.zeros((nPoints, 1))
    for i in range(nPoints):
        t = tmin + (tmax - tmin) * i / (nPoints)
        data[i,0] = 0.0
        for j in range(coeffs.size):
            data[i] += np.power(t, j) * coeffs[j, 0]
    return data


print(getTimeData(3, 0.1))