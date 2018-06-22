from matplotlib import pyplot as plt
import numpy as np
import scipy as sci

# points = "(-0.05249999999999999, 0.12),\
# (0.015000000000000013, 0.15),\
# (0.037500000000000006, 0.11125),\
# (0.037500000000000006, 0.08875),\
# (0.015000000000000013, 0.05),\
# (-0.05249999999999999, 0.08),\
# (-0.05249999999999999, -0.08),\
# (0.015000000000000013, -0.05),\
# (0.037500000000000006, -0.08875),\
# (0.037500000000000006, -0.11125),\
# (0.015000000000000013, -0.15),\
# (-0.05249999999999999, -0.12)"git

points = "(-0.042749999999999996, 0.019), \
 (-0.042749999999999996, -0.019), \
 (0.042749999999999996, 0.010687499999999999), \
 (0.042749999999999996, -0.010687499999999999), \
 (0.021375000000000005, -0.0475), \
 (0.021375000000000005, 0.0475)"

list = points.split("),")
x = []
y = []
for str in list:
    xStr, yStr = str.replace("(", "").replace(")","").split(",")
    x.append(float(xStr))
    y.append(float(yStr))

print(sum(x) / 6)
plt.plot(x, y, marker='o')
plt.grid()
plt.show()

# t0 = 0.0
# tF = 1.0
# x0 = np.array([0.0, 0.0, 0.0])
# xF = np.array([1.0, 0.0, 0.0])
#
# dt = tF - x0
# n = 6