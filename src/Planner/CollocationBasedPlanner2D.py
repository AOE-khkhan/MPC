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

for i in range(nIterations):
    # Setup end point constraints
    
    # Setup scalar multiplier constraints

    # Setup equality dynamic constraints

    # Setup cop location constraints

    for i in range(nodes):
        pass
    pass


