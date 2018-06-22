import numpy as np
from scipy.linalg import logm, expm

# Converts the objective Jq = c with weight W into a quadratic cost
# Q = Jt * W * J, f = -Jt * c  Jt is J transpose
def getQuadProgCostFunction(J, c, W):
    Jt = J.transpose()
    JtW = Jt.dot(W)
    Q = JtW.dot(J)
    f = -JtW.dot(c)
    return Q, f

def getAngularMomentumInstability(w, I):
    return np.matmul(skew(w), np.matmul(I, w))

def skew(A):
    skewA = np.zeros((3,3))
    skewA[0, 1] = -A[2, 0]
    skewA[0, 2] = A[1, 0]
    skewA[1, 0] = A[2, 0]
    skewA[1, 2] = -A[0, 0]
    skewA[2, 0] = -A[1, 0]
    skewA[2, 1] = A[0, 0]
    return skewA

def askew(A):
    return np.array([[-A.item(1, 2)], [A.item(0, 2)], [-A.item(0, 1)]])


def getMomentumAboutCoM(x, cop, F):
    r = cop - x
    return np.array([[r[1, 0] * F[2, 0] - r[2, 0] * F[1, 0]], [r[2, 0] * F[0, 0] - r[0, 0] * F[2, 0]], [r[0, 0] * F[1, 0] - r[1, 0] * F[0, 0]]])

def getLogDiff(R1, R2):
    R2inv = np.linalg.inv(R2)
    return askew(logm(np.matmul(R1, R2inv)))

def getRotationMatrix(w, deltaT):
    return expm(skew(w)*deltaT)

def removeNullConstraints(J, c, threshold):
    rows = J.shape[0]
    cols = J.shape[1]
    rowRemovalList = []
    for i in range(rows):
        remove = True
        for j in range(cols):
            remove = remove and (abs(J.item(i,j)) < threshold)
        if(remove):
            if abs(c.item(i, 0)) > threshold:
                print("Removing constraint with zero coefficient and non zero objective")
            rowRemovalList.append(i)
    print("Removing " + str(len(rowRemovalList)) + " constraints")
    return np.delete(J, rowRemovalList, axis=0), np.delete(c, rowRemovalList, axis=0)
