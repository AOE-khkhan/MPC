from sympy import *

f0, f1, m0, m1 = symbols('f0 f1 m0 m1')

a = m0 + m1 - 2*f1 + 2* f0
b = 3 * f1 - 3 * f0 - m1 - 2 * m0
c = m0
d = f0

ans = simplify ((2 * b * b * b - 9 * a * b * c + 27 * a* a* d) / (27 * a * a))
print(ans)