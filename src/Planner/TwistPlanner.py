import numpy as np
import scipy as sci
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=1000)
nodes = 4

tLift = 0.4
tFlight = 0.15
tLand = 0.4

tLiftOff = tLift
tTouchdown = tLift + tFlight
tEnd = tTouchdown + tLand

upperBodyIzz = 0.4
lowerBodyIzz = 0.1

thetaI = 0.0
thetaF = np.pi / 4
print(thetaF)
## Generate feet trajectory
feetTrajectory = np.zeros((nodes - 1, 6))
# Only need to compute the flight trajectory, others will be zero
coeffEq = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [1.0, pow(tFlight, 1), pow(tFlight, 2), pow(tFlight, 3), pow(tFlight, 4), pow(tFlight, 5)],
                    [0.0, pow(tFlight, 1), 2.0 * pow(tFlight, 1), 3.0 * pow(tFlight, 2), 4.0 * pow(tFlight, 3), 5.0 * pow(tFlight, 4)],
                    [0.0, 0.0, 2.0, 6.0 * pow(tFlight, 1), 12.0 * pow(tFlight, 2), 20.0 * pow(tFlight, 3)]])
coeffAr = np.array([[thetaI],
                    [0.0],
                    [0.0],
                    [thetaF],
                    [0.0],
                    [0.0]])

ans = np.linalg.solve(coeffEq, coeffAr)
feetTrajectory[1:2, :] = ans.T
print(feetTrajectory)

T = [tFlight * i / 10 for i in range(11)]
xFeet = [sum([feetTrajectory[1, i] * pow(t, i) for i in range(6)]) for t in T]


