import numpy as np

# Converts the objective Jq = c with weight W into a quadratic cost
# Q = Jt * W * J, f = -Jt * c  Jt is J transpose
def getQuadProgCostFunction(J, c, W):
    Jt = J.transpose()
    JtW = Jt.dot(W)
    Q = JtW.dot(J)
    f = -JtW.dot(c)
    return Q, f