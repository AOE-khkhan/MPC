import sympy as sym
dt , t, gx, gy, gz, m= sym.symbols(('dt', 't', 'gx', 'gy', 'gz', 'm'))
fx0, fx1, mx0, mx1, vx0, px0 = sym.symbols(('fx0', 'fx1', 'mx0', 'mx1', 'vx0', 'px0'))
fy0, fy1, my0, my1, vy0, py0 = sym.symbols(('fy0', 'fy1', 'my0', 'my1', 'vy0', 'py0'))
fz0, fz1, mz0, mz1, vz0, pz0 = sym.symbols(('fz0', 'fz1', 'mz0', 'mz1', 'vz0', 'pz0'))
tnorm = t / dt
tnorm5 = tnorm * tnorm * tnorm * tnorm * tnorm
tnorm4 = tnorm * tnorm * tnorm * tnorm
tnorm3 = tnorm * tnorm * tnorm
tnorm2 = tnorm * tnorm
tnorm1 = tnorm

Fx = tnorm3 * (mx0 + mx1 - 2 * fx1 + 2 * fx0) + tnorm2 * (-2 * mx0 - mx1 + 3 * fx1 - 3 * fx0) + tnorm1 * (mx0) + fx0
Fy = tnorm3 * (my0 + my1 - 2 * fy1 + 2 * fy0) + tnorm2 * (-2 * my0 - my1 + 3 * fy1 - 3 * fy0) + tnorm1 * (my0) + fy0
Fz = tnorm3 * (mz0 + mz1 - 2 * fz1 + 2 * fz0) + tnorm2 * (-2 * mz0 - mz1 + 3 * fz1 - 3 * fz0) + tnorm1 * (mz0) + fz0

px = tnorm5 * (mx0 + mx1 - 2 * fx1 + 2 * fx0) * dt*dt/ (20 * m) +\
     tnorm4 * (-2 * mx0 - mx1 + 3 * fx1 - 3 * fx0) * dt*dt/ (12 * m) +\
     tnorm3 * (mx0) * dt*dt/ (6 * m) +\
     tnorm2 * (fx0 + m*gx) * dt*dt/ (2 * m) +\
     tnorm1 * vx0 * dt +\
     px0
py = tnorm5 * (my0 + my1 - 2 * fy1 + 2 * fy0) * dt*dt/ (20 * m) +\
     tnorm4 * (-2 * my0 - my1 + 3 * fy1 - 3 * fy0) * dt*dt/ (12 * m) +\
     tnorm3 * (my0) * dt*dt/ (6 * m) +\
     tnorm2 * (fy0 + m*gy) * dt*dt/ (2 * m) +\
     tnorm1 * vy0 * dt +\
     py0
pz = tnorm5 * (mz0 + mz1 - 2 * fz1 + 2 * fz0) * dt*dt/ (20 * m) +\
     tnorm4 * (-2 * mz0 - mz1 + 3 * fz1 - 3 * fz0) * dt*dt/ (12 * m) +\
     tnorm3 * (mz0) * dt*dt/ (6 * m) +\
     tnorm2 * (fz0 + m*gz) * dt*dt/ (2 * m) +\
     tnorm1 * vz0 * dt +\
     pz0

tx = sym.expand(py * Fz - pz * Fy)
ty = sym.expand(pz * Fx - px * Fz)
tz = sym.expand(px * Fy - py * Fx)

txPoly = sym.Poly(tx, t)
txCoeffs = txPoly.coeffs()
for i in range(8):
    print(sym.simplify(txCoeffs[i]))
    pass

tx = tx.subs(t, dt)
sym.simplify(tx)
print(tx)