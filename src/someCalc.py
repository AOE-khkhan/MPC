import sympy as sym
ixx, ixy, ixz, iyy, iyz, izz = sym.symbols(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'))
izy = iyz
iyx = ixy
izx = ixz

I = sym.Matrix([[ixx, ixy, ixz],[iyx, iyy, iyz],[izx, izy, izz]])
print(I)
Idet = sym.simplify(I.det())
Iinv = sym.simplify(I.inv() * I.det())
print(Iinv)
print(Idet)