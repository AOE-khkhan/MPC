from builtins import print
from cvxopt import solvers, matrix
import scipy as sci
from src.Planner.HelperFuncs import *
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

'''This code solves for a CoM trajectory when the forces exerted on the system can only be along the line
joining the CoP and CoM'''

T = 1.0
tFlight = 0.4
tLand = 0.6
g = -9.81
xi = np.array([0.0, 0.35])
vi = np.array([0.0, 0.0])
ai = np.array([0.0, 0.0])
xf = np.array([0.4, 0.35])
vf = np.array([0.0, 0.0])
af = np.array([0.0, 0.0])

dF = 0.025
vLbi = xi[0] - dF
vUbi = xi[0] + dF
vLbf = xf[0] - dF
vUbf = xf[0] + dF

nv = 8  # number of coefficients for CoP trajectory
nx = nv  # number of coefficients for CoM trajectory
nu = 4  # number of coefficients for parameter trajectory
mdx = 3  # number of trajectory derivatives to match
mdv = 1  # number of CoP derivatives to match
mdu = 1  # number of scalar derivatives to match

nodes = 5  # number of nodes into which the system is divided
# D = [... nxi nzi nvi nui ...]
deltaT = T / (nodes - 1)

# Setup nominal trajectory
dx = (xf - xi) / (nodes - 1)
D = np.zeros(((nodes - 1) * (2 * nx + nv + nu)))
u0 = 2  # FIXME
nodeT = [i * deltaT for i in range(nodes)]
for i in range(nodes - 1):
    t = nodeT[i]
    D[i * (2 * nx + nv + nu) + 0] = xi[0] + i * dx[0]
    D[i * (2 * nx + nv + nu) + 1] = dx[0] / deltaT
    D[i * (2 * nx + nv + nu) + nx] = xi[1] + i * dx[1]
    D[i * (2 * nx + nv + nu) + nx + 1] = dx[1] / deltaT
    if t <= tFlight:
        D[i * (2 * nx + nv + nu) + 2 * nx] = xi[0]
    elif t >= tLand:
        D[i * (2 * nx + nv + nu) + 2 * nx] = xf[0]
    else:
        D[i * (2 * nx + nv + nu) + 2 * nx] = 0.5 * (xf[0] + xi[0])
    D[i * (2 * nx + nv + nu) + 2 * nx + nv] = u0
    pass

nIterations = 1
nNode = (2 * nx + nv + nu)

for i in range(nIterations):
    ## Setup end point constraints
    endJ = np.zeros((12, (nodes - 1) * nNode))
    endC = np.zeros((12, 1))
    # Setup initial point constraints
    for j in range(2):
        endJ[3 * j + 0, j * nx + 0] = 1.0
        endJ[3 * j + 1, j * nx + 1] = 1.0
        endJ[3 * j + 2, j * nx + 2] = 2.0
        endC[3 * j + 0, 0] = xi[j] - D[0 + j * nx]
        endC[3 * j + 1, 0] = vi[j] - D[0 + j * nx + 1]
        endC[3 * j + 2, 0] = ai[j] - 2.0 * D[0 + j * nx + 2]
        pass
    # Setup final point constraints
    for j in range(2):
        for k in range(nx):
            endJ[6 + 3 * j + 0, (nodes - 2) * nNode + j * nx + k] = deltaT ** k
            endJ[6 + 3 * j + 1, (nodes - 2) * nNode + j * nx + k] = k * deltaT ** k
            endJ[6 + 3 * j + 2, (nodes - 2) * nNode + j * nx + k] = k * (k - 1) * deltaT ** k
            pass
        endC[6 + 3 * j + 0, 0] = xf[j] - D[(nodes - 2) * nNode + j * nx]
        endC[6 + 3 * j + 1, 0] = vf[j] - D[(nodes - 2) * nNode + j * nx + 1]
        endC[6 + 3 * j + 2, 0] = af[j] - 2.0 * D[(nodes - 2) * nNode + j * nx + 2]
        pass
    ## Setup collocation constraints
    # Trajectory collocation
    collJx = np.zeros(((nodes - 2) * mdx * 2, (nodes - 1) * nNode))
    collCx = np.zeros(((nodes - 2) * mdx * 2, 1))
    for j in range(0, nodes - 2):
        for k in range(2):
            for m in range(nx):
                fact = 1.0
                nDerivatives = min(m + 1, mdx)
                for l in range(nDerivatives):
                    collJx[j * (2 * mdx) + k * mdx + l, j * nNode + k * nx + m] = fact * (deltaT ** (m - l)) * 1.0
                    collJx[j * (2 * mdx) + k * mdx + l, (j + 1) * nNode + k * nx + m] = -fact * (0 ** (m - l)) * 1.0
                    collCx[j * (2 * mdx) + k * mdx + l, 0] = - fact * (deltaT ** (m - l)) * D[j * nNode + k * nx + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + k * nx + m]
                    fact = fact * (m - l)
                pass
            pass
        pass
    # CoP collocation
    collJv = np.zeros(((nodes - 2) * mdv, (nodes - 1) * nNode))
    collCv = np.zeros(((nodes - 2) * mdv, 1))
    for j in range(0, nodes - 2):
        for m in range(nv):
            fact = 1.0
            nDerivatives = min(m + 1, mdv)
            for l in range(nDerivatives):
                collJv[j * mdv + l, j * nNode + 2 * nx + m] = fact * (deltaT ** (m - l)) * 1.0
                collJv[j * mdv + l, (j + 1) * nNode + 2 * nx + m] = -fact * (0 ** (m - l)) * 1.0
                collCv[j * mdv + l, 0] = -fact * (deltaT ** (m - l)) * D[j * nNode + 2 * nx + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + 2 * nx + m]
                fact = fact * (m - l)
            pass
        pass

    # Scalar multiplier collocation
    collJu = np.zeros(((nodes - 2) * mdu, (nodes - 1) * nNode))
    collCu = np.zeros(((nodes - 2) * mdu, 1))
    for j in range(0, nodes - 2):
        for m in range(nu):
            fact = 1.0
            nDerivatives = min(m + 1, mdu)
            for l in range(nDerivatives):
                collJu[j * mdu + l, j * nNode + 2 * nx + nv + m] = fact * (deltaT ** (m - l)) * 1.0
                collJu[j * mdu + l, (j + 1) * nNode + 2 * nx + nv + m] = -fact * (0 ** (m - l)) * 1.0
                collCu[j * mdu + l, 0] = - fact * (deltaT ** (m - l)) * D[j * nNode + 2 * nx + nv + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + 2 * nx + nv + m]
                fact = fact * (m - l)
            pass
        pass

    ## Setup equality differential dynamic constraints
    dynJ = np.zeros((2 * (nodes - 1) * (nx + nu), (nodes - 1) * nNode))
    dynC = np.zeros((2 * (nodes - 1) * (nx + nu), 1))
    gvec = np.array([[0.0], [0.0], [g]])
    for j in range(nodes - 1):
        for k in range(2):
            for l in range(nx - 2):
                dynJ[j * 2 * (nx + nu) + k * (nx + nu) + l, j * nNode + k * nx + l + 2] = (l + 1) * (l + 2)
                dynC[j * 2 * (nx + nu) + k * (nx + nu) + l, 0] = -D[j * nNode + k * nx + l + 2] * (l + 1) * (l + 2)
                nConv = max(0, l - nu)
                for m in range(nConv, l + 1):
                    n = l - m
                    dynJ[j * 2 * (nx + nu) + k * (nx + nu) + l, j * nNode + k * nx + m] = -D[j * nNode + 2 * nx + nv + n]
                    dynJ[j * 2 * (nx + nu) + k * (nx + nu) + l, j * nNode + 2 * nx + m] = D[j * nNode + 2 * nx + nv + n]
                    dynJ[j + 2 * (nx + nu) + k * (nx + nu) + l, j * nNode + 2 * nx + nv + n] = -D[j * nNode + k * nx + m] + D[j * nNode + 2 * nx + m]
                    dynC[j * 2 * (nx + nu) + k * (nx + nu) + l, 0] += D[j * nNode + 2 * nx + nv + n] * (D[j * nNode + k * nx + m] - D[j * nNode + 2 * nx + m])
                    pass
            pass
        pass
    print(dynJ)
    print(dynC)

    ## Setup scalar multiplier constraints

    ## Setup cop location constraints

    for j in range(nodes):
        pass
pass


