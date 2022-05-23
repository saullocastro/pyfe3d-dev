"""
Internal force vector
"""
import numpy as np
import sympy
from sympy import var, Matrix, symbols, simplify

num_nodes = 4
cpu_count = 6
DOF = 10

var('xi, eta, lex, ley, rho, weight')
var('R')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')

#ley calculated from nodal positions and radius

ONE = sympy.Integer(1)

# shape functions
# - from Reference:
#     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
# cubic
Hi = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
Hxi = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
Hyi = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
Hxyi = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

# node 1 (-1, -1)
# node 2 (+1, -1)
# node 3 (+1, +1)
# node 4 (-1, +1)

Nu = sympy.Matrix([[
   #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
    Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
    Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
    Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
    Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
    ]])
Nv = sympy.Matrix([[
   #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
    0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
    0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
    0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
    0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
    ]])
Nw = sympy.Matrix([[
   #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
    0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
    0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
    0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
    0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
    ]])

Nu_x = (2/lex)*Nu.diff(xi)
Nu_y = (2/ley)*Nu.diff(eta)
Nv_x = (2/lex)*Nv.diff(xi)
Nv_y = (2/ley)*Nv.diff(eta)

Bm = Matrix([
    Nu_x, # epsilon_xx
    Nv_y + 1/R*Nw, # epsilon_yy
    Nu_y + Nv_x # gamma_xy
    ])
Bms = []
for i in range(Bm.shape[0]):
    Bmis = []
    for j in range(Bm.shape[1]):
        Bmij = Bm[i, j]
        if Bmij != 0:
            Bmis.append(symbols('Bm%d_%02d' % (i+1, j+1)))
        else:
            Bmis.append(0)
    Bms.append(Bmis)
Bm = sympy.Matrix(Bms)

Nw_x = (2/lex)*Nw.diff(xi)
Nw_y = (2/ley)*Nw.diff(eta)
v = var('v')
w_x = var('w_x')
w_y = var('w_y')
BmL = Matrix([
    w_x*Nw_x,
    w_y*Nw_y + 1/R**2*v*Nv - 1/R*v*Nw_y - 1/R*w_y*Nv,
    w_x*Nw_y + w_y*Nw_x - 1/R*v*Nw_x - 1/R*w_x*Nv
    ])
BmLs = []
for i in range(BmL.shape[0]):
    BmLis = []
    for j in range(BmL.shape[1]):
        BmLij = BmL[i, j]
        if BmLij != 0:
            BmLis.append(symbols('BmL%d_%02d' % (i+1, j+1)))
        else:
            BmLis.append(0)
    BmLs.append(BmLis)
BmL = Matrix(BmLs)

Nphix = -(2/lex)*Nw.diff(xi)
Nphiy = -(2/ley)*Nw.diff(eta)
Nphix_x = (2/lex)*Nphix.diff(xi)
Nphix_y = (2/ley)*Nphix.diff(eta)
Nphiy_x = (2/lex)*Nphiy.diff(xi)
Nphiy_y = (2/ley)*Nphiy.diff(eta)
Bb = sympy.Matrix([
    Nphix_x,
    Nphiy_y + 1/R*Nv_y,
    Nphix_y + Nphiy_x + 1/R*Nv_x
    ])
Bbs = []
for i in range(Bb.shape[0]):
    Bbis = []
    for j in range(Bb.shape[1]):
        Bbij = Bb[i, j]
        if Bbij != 0:
            Bbis.append(symbols('Bb%d_%02d' % (i+1, j+1)))
        else:
            Bbis.append(0)
    Bbs.append(Bbis)
Bb = Matrix(Bbs)

A = Matrix([
    [A11, A12, A16],
    [A12, A22, A26],
    [A16, A26, A66]])
B = Matrix([
    [B11, B12, B16],
    [B12, B22, B26],
    [B16, B26, B66]])
D = Matrix([
    [D11, D12, D16],
    [D12, D22, D26],
    [D16, D26, D66]])

ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, Bb.shape[1])])
N = A*(Bm + BmL)*ue + B*Bb*ue
M = B*(Bm + BmL)*ue + D*Bb*ue
print('Nxx =', N[0])
print('Nyy =', N[1])
print('Nxy =', N[2])
print('Mxx =', M[0])
print('Myy =', M[1])
print('Mxy =', M[2])

N = Matrix([[Nxx, Nyy, Nxy]]).T
M = Matrix([[Mxx, Myy, Mxy]]).T

fint_terms = Bm.T*N + BmL.T*N + Bb.T*M
fint = weight*(lex*ley)/4.*(fint_terms)

def name_ind(i):
    if i >=0 and i < DOF:
        return 'c1'
    elif i >= DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

for i, fi in enumerate(fint):
    if fi == 0:
        continue
    si = name_ind(i)
    print('fint[%d + %s] +=' % (i%DOF, si), fi)

