from cvxopt import solvers, matrix
import numpy as np
import scipy as sci
from src.Planner.HelperFuncs import *
from matplotlib import pyplot as plt

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

nv = 8 # number of coefficients for CoP trajectory
nx = nv # number of coefficients for CoM trajectory
nu = 4 # number of coefficients for parameter trajectory

nodes = 11 # number of nodes into which the system is divided
# D = [... nxi nzi nvi nui ...]
deltaT = T / (nodes - 1)

# Setup nominal trajectory
dx = (xf - xi) / (nodes - 1)
D = np.zeros((nodes * (2 * nx + nv + nu)))
u0 = 2
nodeT = [i * deltaT for i in range(nodes)]
for i in range(nodes):
    t = nodeT[i]
    D[i * (2 * nx + nv + nu) + 0] = xi[0] + i * dx[0]
    D[i * (2 * nx + nv + nu) + 1] = dx[0] / deltaT
    D[i * (2 * nx + nv + nu) + nx] = xi[1] + i * dx[1]
    D[i * (2 * nx + nv + nu) + nx + 1] = dx[1] / deltaT
    if(t <= tFlight):
        D[i * (2 * nx + nv + nu) + 2 * nx] = xi[0]
    elif (t >= tLand):
        D[i * (2 * nx + nv + nu) + 2 * nx] = xf[0]
    else:
        D[i * (2 * nx + nv + nu) + 2 * nx] = 0.5 * (xf[0] + xi[0])
    D[i * (2 * nx + nv + nu) + 2 * nx + nv] = u0
    pass

nIterations = 1
nNode = (2 * nx + nv + nu)

for i in range(nIterations):
    ## Setup end point constraints
    endJ = np.zeros((12, nodes * nNode))
    endC = np.zeros((12, 1))
    # Setup initial point constraints
    for j in range(2):
        endJ[3 * j + 0, j * nx + 0] = 1.0
        endJ[3 * j + 1, j * nx + 1] = 1.0
        endJ[3 * j + 2, j * nx + 2] = 2.0
        endC[3 * j + 0, 1] = xi[j] - D[0 + j * nx]
        endC[3 * j + 1, 1] = vi[j] - D[0 + j * nx + 1]
        endC[3 * j + 2, 1] = ai[j] - 2.0 * D[0 + j * nx + 2]
        pass
    # Setup final point constraints
    for j in range(2):
        endJ[6 + 3 * j + 0, (nodes - 1) * nNode + j * nx + 0] = 1.0
        endJ[6 + 3 * j + 1, (nodes - 1) * nNode + j * nx + 1] = 1.0
        endJ[6 + 3 * j + 2, (nodes - 1) * nNode + j * nx + 2] = 2.0
        endC[6 + 3 * j + 0, 1] = xf[j] - D[(nodes - 1) * nNode + j * nx]
        endC[6 + 3 * j + 1, 1] = vf[j] - D[(nodes - 1) * nNode + j * nx + 1]
        endC[6 + 3 * j + 2, 1] = af[j] - 2.0 * D[(nodes - 1) * nNode +  j * nx + 2]
        pass

    ## Setup collocation constraints

    ## Setup equality differential dynamic constraints

    ## Setup scalar multiplier constraints

    ## Setup cop location constraints

    for i in range(nodes):
        pass
    pass


