from builtins import print
from cvxopt import solvers, matrix
import scipy as sci
import math
from matplotlib import pyplot as plt
from src.Planner.HelperFuncs import *
from src.Planner.Plotter import plotDataCollocationPlanner
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)
'''This code solves for a CoM trajectory when the forces exerted on the system can only be along the line
joining the CoP and CoM'''

solvers.options['refinement'] = 1
solvers.options['reltol'] = 1e-10
solvers.options['abstol'] = 1e-11
solvers.options['feastol'] = 1e-11
solvers.options['maxiters'] = 500

T = 1.0
tFlight = 1.4
tLand = 1.6
g = -9.81
xi = np.array([-0.01, 0.35])
vi = np.array([0.0, 0.0])
ai = np.array([0.0, 0.0])
xf = np.array([0.01, 0.35])
vf = np.array([0.0, 0.0])
af = np.array([0.0, 0.0])

nvi = -0.01
nvf = 0.01

nui = -g / xi[1]
nuf = -g / xf[1]

dF = 0.025
vLbi = xi[0] - dF
vUbi = xi[0] + dF
vLbf = xf[0] - dF
vUbf = xf[0] + dF

nv = 8  # number of coefficients for CoP trajectory
nx = nv  # number of coefficients for CoM trajectory
nu = 3  # number of coefficients for parameter trajectory
mdx = 2  # number of trajectory derivatives to match
mdv = 2  # number of CoP derivatives to match
mdu = 2  # number of scalar derivatives to match
ncv = nv - 1 # number of support polygon constraints segment
ndyn = 4 # number of points in a segment to enforce dynamics constraints at
ncu = 3 # number of points in a segment to enforce the scalar equality and inequality constraints

nodes = 6  # number of nodes into which the system is divided
# D = [... nxi nzi nvi nui ...]
deltaT = T / (nodes - 1)

# Setup nominal trajectory
dx = (xf - xi) / (nodes - 1)
D = np.zeros(((nodes - 1) * (2 * nx + nv + nu), 1))
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
    D[i * (2 * nx + nv + nu) + 2 * nx + nv] = nui

nIterations = 14
nNode = (2 * nx + nv + nu)
plotDataCollocationPlanner(nodes, nodeT, D, nx, nv, nu, -1)

for i in range(nIterations):
    ## Setup end point constraints
    # Setup initial trajectory constraints
    endJ = np.zeros((12, (nodes - 1) * nNode))
    endC = np.zeros((12, 1))
    for j in range(2):
        endJ[3 * j + 0, j * nx + 0] = 1.0
        endJ[3 * j + 1, j * nx + 1] = 1.0
        endJ[3 * j + 2, j * nx + 2] = 2.0
        endC[3 * j + 0, 0] = xi[j] - D[0 + j * nx]
        endC[3 * j + 1, 0] = vi[j] - D[0 + j * nx + 1]
        endC[3 * j + 2, 0] = ai[j] - 2.0 * D[0 + j * nx + 2]

    # Setup final trajectory constraints
    for j in range(2):
        endC[6 + 3 * j + 0, 0] = xf[j]
        endC[6 + 3 * j + 1, 0] = vf[j]
        endC[6 + 3 * j + 2, 0] = af[j]
        for k in range(nx):
            endJ[6 + 3 * j + 0, (nodes - 2) * nNode + j * nx + k] = deltaT ** k
            endJ[6 + 3 * j + 1, (nodes - 2) * nNode + j * nx + k] = k * deltaT ** k
            endJ[6 + 3 * j + 2, (nodes - 2) * nNode + j * nx + k] = k * (k - 1) * deltaT ** k
            endC[6 + 3 * j + 0, 0] -= D[(nodes - 2) * nNode + j * nx + k] * deltaT ** k
            endC[6 + 3 * j + 1, 0] -= k * D[(nodes - 2) * nNode + j * nx + k] * deltaT ** k
            endC[6 + 3 * j + 2, 0] -= k * (k - 1) * D[(nodes - 2) * nNode + j * nx + k] * deltaT ** k

    # Setup initial CoP constraints
    endJv = np.zeros((2, (nodes - 1) * nNode))
    endCv = np.zeros((2, 1))
    endJv[0, 0 + 2 * nx + 0] = 1.0
    endCv[0, 0] = nvi - D[0 + 2 * nx + 0]

    # Setup final CoP constraints
    endCv[1, 0] = nvf
    for k in range(nv):
        endJv[1, (nodes - 2) * nNode + 2 * nx + k] = deltaT ** k
        endCv[1, 0] -= D[(nodes - 2) * nNode + 2 * nx + k] * deltaT ** k

    # Setup initial scalar constraints
    endJu = np.zeros((2, (nodes - 1) * nNode))
    endCu = np.zeros((2, 1))
    endJu[0, 0 + 2 * nx + nv + 0] = 1.0
    endCu[0, 0] = nui - D[0 + 2 * nx + nv]

    # Setup final scalar constraints
    endCu[1, 0] = nuf
    for k in range(nu):
        endJu[1, (nodes - 2) * nNode + 2 * nx + nv + k] = deltaT ** k
        endCu[1, 0] -= D[(nodes - 2) * nNode + 2 * nx + nv + k] * deltaT ** k

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
                    collCx[j * (2 * mdx) + k * mdx + l, 0] += -fact * (deltaT ** (m - l)) * D[j * nNode + k * nx + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + k * nx + m]
                    fact = fact * (m - l)

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
                collCv[j * mdv + l, 0] += -fact * (deltaT ** (m - l)) * D[j * nNode + 2 * nx + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + 2 * nx + m]
                fact = fact * (m - l)

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
                collCu[j * mdu + l, 0] += -fact * (deltaT ** (m - l)) * D[j * nNode + 2 * nx + nv + m] + fact * (0 ** (m - l)) * D[(j + 1) * nNode + 2 * nx + nv + m]
                fact = fact * (m - l)

    ## Setup equality differential dynamic constraints
    dynJ = np.zeros((2 * (nodes - 1) * (ndyn), (nodes - 1) * nNode))
    dynC = np.zeros((2 * (nodes - 1) * (ndyn), 1))
    ddt = deltaT / (ndyn + 1)
    for j in range(nodes - 1):
        for k in range(ndyn):
            uVal = 0.0
            ddt_k = ddt * (k + 1)
            for l in range(nu):
                uVal += D[j * nNode + 2 * nx + nv + l] * (ddt_k) ** l
            xVal = 0.0
            vVal = 0.0
            xddVal = 0.0
            zVal = 0.0
            zddVal = 0.0
            for l in range(nx):
                xVal += D[j * nNode + l] * (ddt_k) ** l
                xddVal += D[j * nNode + l] * (ddt_k) ** (l - 2) * l * (l - 1)
                zVal += D[j * nNode + nx + l] * (ddt_k) ** l
                zddVal += D[j * nNode + nx + l] * (ddt_k) ** (l - 2) * l * (l - 1)
                vVal += D[j * nNode + 2 * nx +  l] * (ddt_k) ** l
            ## X axis
            for l in range(nx):
                dynJ[j * 2 * ndyn + k, j * nNode + 0 * nx + l] = (ddt_k ** (l - 2)) * l * (l - 1) - uVal * (ddt_k ** l)
                dynJ[j * 2 * ndyn + k, j * nNode + 2 * nx + l] = uVal * (ddt_k **l)
            for l in range(nu):
                dynJ[j * 2 * ndyn + k, j * nNode + 2 * nx + nv + l] = (vVal - xVal) * (ddt_k ** l)
            dynC[j * 2 * ndyn + k, 0] = uVal * (xVal - vVal) - xddVal
            ## Z axis
            for l in range(nx):
                dynJ[j * 2 * ndyn + ndyn + k, j * nNode + 1 * nx + l] = (ddt_k ** (l - 2)) * l * (l - 1) - uVal * (ddt_k ** l)
            for l in range(nu):
                dynJ[j * 2 * ndyn + ndyn + k, j * nNode + 2 * nx + nv + l] = (-zVal) * (ddt_k ** l)
            dynC[j * 2 * ndyn + ndyn + k, 0] = uVal * zVal - zddVal + g

    ## Setup scalar multiplier constraints
    conJinu = np.zeros(((nodes - 1) * ncu, (nodes - 1) * nNode))
    conCinu = np.zeros(((nodes - 1) * ncu, 1))
    ddt = deltaT / (ncu + 1)
    for j in range(nodes - 1):
        for k in range(ncu):
            ddt_k = ddt * (k + 1)
            uVal = 0.0
            for l in range(nu):
                coeff = (ddt_k) ** l
                uVal += D[j * nNode + 2 * nx + nv + l] * coeff
                conJinu[j * ncu + k, j * nNode + 2 * nx + nv + l] = -coeff
            conCinu[j * ncu + k, 0] = uVal

    ## Setup cop location constraints
    indicesToDelete = []
    suppPolJ = np.zeros(((nodes - 1) * (ncv + 1) * 2, (nodes - 1)* nNode ))
    suppPolC = np.zeros(((nodes - 1) * (ncv + 1) * 2, 1))
    for j in range(nodes - 1):
        t = nodeT[j]
        dt = 1.0 / ncv
        for k in range(ncv + 1):
            if t < tFlight:
                suppPolC[j * (ncv + 1) * 2 + k * 2, 0] = vUbi
                suppPolC[j * (ncv + 1) * 2 + k * 2 + 1, 0] = -vLbi
            elif t > tLand:
                suppPolC[j * (ncv + 1) * 2 + k * 2, 0] = vUbf
                suppPolC[j * (ncv + 1) * 2 + k * 2 + 1, 0] = -vLbf
            else:
                suppPolC[j * (ncv + 1) * 2 + k * 2, 0] = math.inf
                suppPolC[j * (ncv + 1) * 2 + k * 2 + 1, 0] = math.inf
                indicesToDelete.append(j * (ncv + 1) * 2 + k * 2)
                indicesToDelete.append(j * (ncv + 1) * 2 + k * 2 + 1)
            for l in range(nv):
                coeff = (dt * k * deltaT) ** l
                suppPolJ[j * (ncv + 1) * 2 + k * 2, j * nNode + 2 * nx + l] = coeff
                suppPolJ[j * (ncv + 1) * 2 + k * 2 + 1, j * nNode + 2 * nx + l] = -coeff
                suppPolC[j * (ncv + 1) * 2 + k * 2, 0] -= coeff * D[j * nNode + 2 * nx + l]
                suppPolC[j * (ncv + 1) * 2 + k * 2 + 1, 0] += coeff * D[j * nNode + 2 * nx + l]

    suppPolJF = np.delete(suppPolJ, indicesToDelete, axis=0)
    suppPolCF = np.delete(suppPolC, indicesToDelete, axis=0)

    ## Setup acceleration minimization objective
    objJacc = np.zeros((2 , (nodes - 1) * nNode))
    objCacc = np.zeros((2 , 1))
    for j in range(2):
        for k in range(nodes - 1):
            for l in range(nx):
                for m in range(l, nx + l):
                    coeff = (deltaT ** m + 1) / (m + 1) * l * (l - 1) * (m - l) * (m - l - 1) * D[k * nNode + j * nx + m - l]
                    objJacc[j,  k * nNode + j * nx + l] += coeff
                    objCacc[j, 0] += coeff * D[k * nNode + j * nx + l]

    rho = np.identity(D.size) * 1e-3
    Wx = np.identity(12)
    Wv = np.identity(2)
    Wu = np.identity(2)
    Wacc = np.identity(2)
    #Hx, fx = getQuadProgCostFunction(endJ, endC, Wx)
    #Hv, fv = getQuadProgCostFunction(endJv, endCv, Wv)
    Hacc, facc = getQuadProgCostFunction(objJacc, -objCacc, Wacc)
    #Hdyn, fdyn = getQuadProgCostFunction(dynJ, dynC, dynW)
    H = Hacc + rho
    f = facc
    # Aeq = np.vstack((dynJ, collJx, collJv, collJu))
    # beq = np.vstack((dynC, collCx, collCv, collCu))
    Aeq = np.vstack((endJ, endJv, endJu, dynJ, collJx, collJv, collJu))
    beq = np.vstack((endC, endCv, endCu, dynC, collCx, collCv, collCu))
    Aeq, beq = removeNullConstraints(Aeq, beq, 1e-10)
    Ain = np.vstack((suppPolJF, conJinu))
    bin = np.vstack((suppPolCF, conCinu))

    P = matrix(H, tc='d')
    q = matrix(f, tc='d')
    G = matrix(Ain, tc='d')
    h = matrix(bin, tc='d')
    A = matrix(Aeq, tc='d')
    b = matrix(beq, tc='d')

    Dmat = np.vstack((H, Aeq, Ain))
    #print(np.linalg.matrix_rank(dynJ))
    #print(dynJ.shape)
    #print(np.linalg.matrix_rank(Dmat))
    #print(D.shape)
    #print(np.linalg.matrix_rank(Ain))
    #print(Ain.shape)
    #print(np.linalg.matrix_rank(Aeq))
    #print(Aeq.shape)

    soln = solvers.qp(P, q, G, h, A, b)
    output = soln['status']
    print("Iteration " + str(i) + " terminated with status " + output)
    optX = np.array(soln['x'])
    D = D + optX
    plotDataCollocationPlanner(nodes, nodeT, D, nx, nv, nu, i)
