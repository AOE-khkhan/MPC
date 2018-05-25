from matplotlib import  pyplot as plt

points = "-0.0175, 0.12000000000000001),\
(0.036250000000000004, 0.14),\
(0.07250000000000001, 0.12250000000000001),\
(0.07250000000000001, -0.12250000000000001),\
(0.036250000000000004, -0.14),\
(-0.0175, -0.12000000000000001"

list = points.split("),")
x = []
y = []
for str in list:
    xStr, yStr = str.replace("(", "").replace(")","").split(",")
    x.append(float(xStr))
    y.append(float(yStr))

plt.plot(x, y, marker='o')
plt.grid()
plt.show()
