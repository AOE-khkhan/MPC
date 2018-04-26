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
thetaF = np.pi / 8
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

T = [tFlight * i / 10 for i in range(11)]
thetaFeet = [sum([feetTrajectory[1, i] * pow(t, i) for i in range(6)]) for t in T]
tauFeet = lowerBodyIzz * np.array([[feetTrajectory[i, j + 2] * (j + 1) * (j + 2) for j in range(4)] for i in range(nodes - 1)])

thetaDDUpperBodyCon = tauFeet[1, :]/ upperBodyIzz
print(thetaDDUpperBodyCon)
delOmega = sum([thetaDDUpperBodyCon[i] * pow(tFlight, i + 1) / (i + 1) for i in range(4)])
delTheta = sum([thetaDDUpperBodyCon[i] * pow(tFlight, i + 2) / ((i + 1) * (i + 2)) for i in range(4)])

# Create the optimization problem
# Decision variables are the 12 coefficients for the upper body trajectory in launch and land phases

conJdyn = np.array([[           0.0, -pow(tLift, 0), -pow(tLift, 1), -pow(tLift, 2), -pow(tLift, 3), -pow(tLift, 4), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                 [-pow(tLift, 0), -pow(tLift, 1) - tFlight * pow(tLift, 0), -pow(tLift, 2) - tFlight * pow(tLift, 1), -pow(tLift, 3)- tFlight * pow(tLift, 2), -pow(tLift, 4)- tFlight * pow(tLift, 3), -pow(tLift, 5)- tFlight * pow(tLift, 4), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
conCdyn = np.array([[delTheta], [delOmega]])

conJstate = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pow(tLand, 0), pow(tLand, 1), pow(tLand, 2), pow(tLand, 3),
                       pow(tLand, 4), pow(tLand, 5)],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pow(tLand, 0), 2.0 * pow(tLand, 1), 3.0 * pow(tLand, 2),
                       4.0 * pow(tLand, 3), 5.0 * pow(tLand, 4)]])
conCstate = np.array([[thetaI],
                      [0.0],
                      [thetaF],
                      [0.0]])

conAeq = np.vstack((conJdyn, conJstate))
conbeq = np.vstack((conCdyn, conCstate))

