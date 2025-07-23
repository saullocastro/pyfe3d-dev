import numpy as np
import sympy
from sympy import simplify, Matrix, var, symbols
from sympy.vector import CoordSys3D, cross

r"""

   3
   |\
   | \    positive normal in CCW
   |  \
   |___\
   1    2

"""

DOF = 6
NUM_NODES = 3

var('h', positive=True, real=True)
var('x, y, x1, y1, x2, y2, x3, y3', real=True, positive=True)
var('A, K6ROT')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E44, E45, E55')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

ONE = sympy.Integer(1)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r = x*R.i + y*R.j

Aexpr = cross(r2 - r1, r3 - r1).components[R.k]/2
print('A =', Aexpr)

AN1 = cross(r2 - r, r3 - r).components[R.k]/2
AN2 = cross(r3 - r, r1 - r).components[R.k]/2
N1 = simplify(AN1/A)
N2 = simplify(AN2/A)
N3 = simplify((A - AN1 - AN2)/A)

N1x = N1.diff(x)
N2x = N2.diff(x)
N3x = N3.diff(x)
N1y = N1.diff(y)
N2y = N2.diff(y)
N3y = N3.diff(y)

# Jacobian
# N1 = N1(x, y)
# N2 = N2(x, y)
# dN1 = [dN1/dx  dN1/dy] dx
# dN2   [dN2/dx  dN2/dy] dy
#
Jinv = Matrix([[N1x, N1y],
               [N2x, N2y]])
detJ = Jinv.inv().det().simplify()
print('detJ =', detJ)
detJ = sympy.var('detJ')

print('N1x =', N1x)
print('N2x =', N2x)
print('N3x =', N3x)
print('N1y =', N1y)
print('N2y =', N2y)
print('N3y =', N3y)

N1x, N2x, N3x = sympy.var('N1x, N2x, N3x')
N1y, N2y, N3y = sympy.var('N1y, N2y, N3y')

# d/dx = dN1/dx*d/dN1 + dN2/dx*d/dN2 + dN3/dx*d/dN3

#NOTE evaluating at 1 integration point at the centre
N1, N2 = sympy.var('N1, N2')
N3 = 1 - N1 - N2

# u v w  rx  ry  rz  (rows are node 1, node2, node3)

#exx = u,x
BLexx = Matrix([[N1x, 0, 0, 0, 0, 0,
                 N2x, 0, 0, 0, 0, 0,
                 N3x, 0, 0, 0, 0, 0]])
#eyy = v,y
BLeyy = Matrix([[0, N1y, 0, 0, 0, 0,
                 0, N2y, 0, 0, 0, 0,
                 0, N3y, 0, 0, 0, 0]])
#gxy = u,y + v,x
BLgxy = Matrix([[N1y, N1x, 0, 0, 0, 0,
                 N2y, N2x, 0, 0, 0, 0,
                 N3y, N3x, 0, 0, 0, 0]])
#kxx = phix,x
#kxx = ry,x
BLkxx = Matrix([[0, 0, 0, 0, N1x, 0,
                 0, 0, 0, 0, N2x, 0,
                 0, 0, 0, 0, N3x, 0]])
#kyy = phiy,y
#kyy = -rx,y
BLkyy = Matrix([[0, 0, 0, -N1y, 0, 0,
                 0, 0, 0, -N2y, 0, 0,
                 0, 0, 0, -N3y, 0, 0]])
#kxy = phix,y + phiy,x
#kxy = ry,y + (-rx),x
BLkxy = Matrix([[0, 0, 0, -N1x, N1y, 0,
                 0, 0, 0, -N2x, N2y, 0,
                 0, 0, 0, -N3x, N3y, 0]])
#gyz = phiy + w,y
#    = -rx + w,y
BLgyz = Matrix([[0, 0, N1y, -N1, 0, 0,
                 0, 0, N2y, -N2, 0, 0,
                 0, 0, N3y, -N3, 0, 0]])
#gxz = phix + w,x
#    = ry + w,x
BLgxz = Matrix([[0, 0, N1x, 0, N1, 0,
                 0, 0, N2x, 0, N2, 0,
                 0, 0, N3x, 0, N3, 0]])
# for drilling stiffness
#   see Eq. 2.20 in F.M. Adam, A.E. Mohamed, A.E. Hassaballa, Degenerated Four Nodes Shell Element with Drilling Degree of Freedom, IOSR J. Eng. 3 (2013) 10–20. www.iosrjen.org (accessed April 20, 2020).
#BLdrilling = Matrix([[N1y/2., -N1x/2., 0, 0, 0, N1,
                      #N2y/2., -N2x/2., 0, 0, 0, N2,
                      #N3y/2., -N3x/2., 0, 0, 0, N3]])

BL = Matrix([BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy, BLgyz, BLgxz])

ABDE = Matrix(
        [[A11, A12, A16, B11, B12, B16,   0,   0],
         [A12, A22, A26, B12, B22, B26,   0,   0],
         [A16, A26, A66, B16, B26, B66,   0,   0],
         [B11, B12, B16, D11, D12, D16,   0,   0],
         [B12, B22, B26, D12, D22, D26,   0,   0],
         [B16, B26, B66, D16, D26, D66,   0,   0],
         [  0,   0,   0,   0,   0,   0, E44, E45],
         [  0,   0,   0,   0,   0,   0, E45, E55]])

var('wij')

# Constitutive linear stiffness matrix
#NOTE reduced integration of stiffness to remove shear locking
KC0e = wij*detJ*(BL.T*ABDE*BL
                 #+ alphat*A66/h*BLdrilling.T*BLdrilling
                 )
for node_i in range(NUM_NODES):
    print(node_i*DOF + 5)
    KC0e[node_i*DOF + 5, node_i*DOF + 5] = K6ROT

nonzero = set()
for ind, val in np.ndenumerate(KC0e):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    name = 'KC0e%02d%02d' % (i, j)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(NUM_NODES*DOF):
    cols = []
    for j in range(NUM_NODES*DOF):
        if j >= i:
            name = 'KC0e%02d%02d' % (i, j)
        else:
            name = 'KC0e%02d%02d' % (j, i)
        if name in nonzero:
            cols.append(var(name))
        else:
            cols.append(0)
    rows.append(cols)
KC0e = Matrix(rows)

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])

finte = KC0e*ue

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
for i in range(2*NUM_NODES):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

nonzero = set()
for ind, val in np.ndenumerate(finte):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    name = 'finte[%d]' % (i)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(NUM_NODES*DOF):
    name = 'finte[%d]' % (i)
    if name in nonzero:
        rows.append(var(name))
    else:
        rows.append(0)
finte = Matrix(rows)
fint = R*finte


def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    else:
        raise

print()
print()
print('_______________________________________')
print()
print('printing fint')
print('_______________________________________')
print()
print()
for ind, val in np.ndenumerate(fint):
    i, j = ind
    si = name_ind(i)
    if sympy.expand(val) == 0:
        continue
    print('            fint[%d+self.%s] +=' % (i%DOF, si), val)
print()
print()
