r"""
Constitutive stiffness matrix (linear) for Quad4 element

w, phi_x and phi_y DOFs based on:

    Hugues T.J.R., Taylor R.L., Kanoknukulchai W. "A simple and efficient
    finite element for plate bending". International Journal of  Numerical
    Methods in Engineering, Volume 11, 1977.


    Reduced integration for bending, and full integration for transverse shear
    terms.




    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""
import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var
from sympy.vector import CoordSys3D, cross


DOF = 6
num_nodes = 4

var('h', positive=True, real=True)
var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)
var('xi, eta, A, alphat')
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
r4 = x4*R.i + y4*R.j

rbottom = r1 + (r2 - r1)*(xi + 1)/2.
rtop = r4 + (r3 - r4)*(xi + 1)/2.
r = rbottom + (rtop - rbottom)*(eta + 1)/2.
xfunc = r.components[R.i]
yfunc = r.components[R.j]

# Jacobian
# http://kis.tu.kielce.pl/mo/COLORADO_FEM/colorado/IFEM.Ch17.pdf
#NOTE for linear element these derivatives are constant
# xi = xi(x, y)
# eta = eta(x, y)
#J = [dx/dxi  dy/dxi ]
#    [dx/deta dy/deta]
# dx   = J.T dxi
# dy         deta
#
# dxi   = Jinv.T dx
# deta           dy
#
# Jinv:
# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi  =  [j11  j12] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta =  [j21  j22] d/deta
#
J11 = xfunc.diff(xi)
J12 = yfunc.diff(xi)
J21 = xfunc.diff(eta)
J22 = yfunc.diff(eta)
print('J11 =', J11)
print('J12 =', J12)
print('J21 =', J21)
print('J22 =', J22)
J11, J12, J21, J22 = sympy.var('J11, J12, J21, J22')
J = Matrix([[J11, J12],
            [J21, J22]])
detJ = J.det().simplify()
print('detJ =', detJ.simplify())

j = J.inv()
j11 = j[0, 0].simplify()
j12 = j[0, 1].simplify()
j21 = j[1, 0].simplify()
j22 = j[1, 1].simplify()

print('j11 =', j11.simplify())
print('j12 =', j12.simplify())
print('j21 =', j21.simplify())
print('j22 =', j22.simplify())

j11, j12, j21, j22 = var('j11, j12, j21, j22')

N1 = (eta*xi - eta - xi + 1)/4.
N2 = -(eta*xi + eta - xi - 1)/4.
N3 = (eta*xi + eta + xi + 1)/4.
N4 = -(eta*xi - eta + xi - 1)/4.

N1xi = N1.diff(xi)
N2xi = N2.diff(xi)
N3xi = N3.diff(xi)
N4xi = N4.diff(xi)

N1eta = N1.diff(eta)
N2eta = N2.diff(eta)
N3eta = N3.diff(eta)
N4eta = N4.diff(eta)

N1x = j11*N1xi + j12*N1eta
N2x = j11*N2xi + j12*N2eta
N3x = j11*N3xi + j12*N3eta
N4x = j11*N4xi + j12*N4eta

N1xxi = N1x.diff(xi)
N1xeta = N1x.diff(eta)
N2xxi = N2x.diff(xi)
N2xeta = N2x.diff(eta)
N3xxi = N3x.diff(xi)
N3xeta = N3x.diff(eta)
N4xxi = N4x.diff(xi)
N4xeta = N4x.diff(eta)

N1xy = j21*N1xxi + j22*N1xeta
N2xy = j21*N2xxi + j22*N2xeta
N3xy = j21*N3xxi + j22*N3xeta
N4xy = j21*N4xxi + j22*N4xeta

N1y = j21*N1xi + j22*N1eta
N2y = j21*N2xi + j22*N2eta
N3y = j21*N3xi + j22*N3eta
N4y = j21*N4xi + j22*N4eta

N1yxi = N1y.diff(xi)
N1yeta = N1y.diff(eta)
N2yxi = N2y.diff(xi)
N2yeta = N2y.diff(eta)
N3yxi = N3y.diff(xi)
N3yeta = N3y.diff(eta)
N4yxi = N4y.diff(xi)
N4yeta = N4y.diff(eta)

N1yx = j11*N1yxi + j12*N1yeta
N2yx = j11*N2yxi + j12*N2yeta
N3yx = j11*N3yxi + j12*N3yeta
N4yx = j11*N4yxi + j12*N4yeta

print('N1 =', N1.simplify())
print('N2 =', N2.simplify())
print('N3 =', N3.simplify())
print('N4 =', N4.simplify())

print('N1x =', N1x.simplify())
print('N2x =', N2x.simplify())
print('N3x =', N3x.simplify())
print('N4x =', N4x.simplify())

print('N1y =', N1y.simplify())
print('N2y =', N2y.simplify())
print('N3y =', N3y.simplify())
print('N4y =', N4y.simplify())
print('')
print('N1xy =', N1xy.simplify())
print('N2xy =', N2xy.simplify())
print('N3xy =', N3xy.simplify())
print('N4xy =', N4xy.simplify())
print('')
print('Niyx only for checking purposes')
print('')
print('N1yx =', N1yx.simplify())
print('N2yx =', N2yx.simplify())
print('N3yx =', N3yx.simplify())
print('N4yx =', N4yx.simplify())
print('')
print('')
print('xi=0, eta=0')
print('N1yx =', N1yx.subs(dict(xi=0, eta=0)).simplify())
print('N2yx =', N2yx.subs(dict(xi=0, eta=0)).simplify())
print('N3yx =', N3yx.subs(dict(xi=0, eta=0)).simplify())
print('N4yx =', N4yx.subs(dict(xi=0, eta=0)).simplify())

print('N1 =', N1.subs(dict(xi=0, eta=0)).simplify())
print('N2 =', N2.subs(dict(xi=0, eta=0)).simplify())
print('N3 =', N3.subs(dict(xi=0, eta=0)).simplify())
print('N4 =', N4.subs(dict(xi=0, eta=0)).simplify())

print('N1x =', N1x.subs(dict(xi=0, eta=0)).simplify())
print('N2x =', N2x.subs(dict(xi=0, eta=0)).simplify())
print('N3x =', N3x.subs(dict(xi=0, eta=0)).simplify())
print('N4x =', N4x.subs(dict(xi=0, eta=0)).simplify())

print('N1y =', N1y.subs(dict(xi=0, eta=0)).simplify())
print('N2y =', N2y.subs(dict(xi=0, eta=0)).simplify())
print('N3y =', N3y.subs(dict(xi=0, eta=0)).simplify())
print('N4y =', N4y.subs(dict(xi=0, eta=0)).simplify())
print('')
print('N1xy =', N1xy.subs(dict(xi=0, eta=0)).simplify())
print('N2xy =', N2xy.subs(dict(xi=0, eta=0)).simplify())
print('N3xy =', N3xy.subs(dict(xi=0, eta=0)).simplify())
print('N4xy =', N4xy.subs(dict(xi=0, eta=0)).simplify())
print('')

detJ = var('detJ')
N1, N2, N3, N4 = var('N1, N2, N3, N4')
N1x, N2x, N3x, N4x = var('N1x, N2x, N3x, N4x')
N1y, N2y, N3y, N4y = var('N1y, N2y, N3y, N4y')

#Nu =  Matrix([[N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0, 0]])
#Nv =  Matrix([[0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0]])
#Nw =  Matrix([[0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0]])
#Nrx = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0]])
#Nry = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0]])
#Nrz = Matrix([[0, 0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4]])

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#exx = u,x = (dxi/dx)*u,xi + (deta/dx)*u,eta = j11 u,xi + j12 u,eta
BLexx = Matrix([[N1x, 0, 0, 0, 0, 0,
                 N2x, 0, 0, 0, 0, 0,
                 N3x, 0, 0, 0, 0, 0,
                 N4x, 0, 0, 0, 0, 0]])
#eyy = v,y = (dxi/dy)*v,xi + (deta/dy)*v,eta = j21 v,xi + j22 v,eta
BLeyy = Matrix([[0, N1y, 0, 0, 0, 0,
                 0, N2y, 0, 0, 0, 0,
                 0, N3y, 0, 0, 0, 0,
                 0, N4y, 0, 0, 0, 0]])
#gxy = u,y + v,x = (dxi/dy)*u,xi + (deta/dy)*u,eta + (dxi/dx)*v,xi + (deta/dy)*v,eta
BLgxy = Matrix([[N1y, N1x, 0, 0, 0, 0,
                 N2y, N2x, 0, 0, 0, 0,
                 N3y, N3x, 0, 0, 0, 0,
                 N4y, N4x, 0, 0, 0, 0]])
#kxx = phix,x = (dxi/dx)*phix,xi + (deta/dx)*phix,eta
#kxx = ry,x = (dxi/dx)*ry,xi + (deta/dx)*ry,eta
BLkxx = Matrix([[0, 0, 0, 0, N1x, 0,
                 0, 0, 0, 0, N2x, 0,
                 0, 0, 0, 0, N3x, 0,
                 0, 0, 0, 0, N4x, 0]])
#kyy = phiy,y = (dxi/dy)*phiy,xi + (deta/dy)*phiy,eta
#kyy = -rx,y = (dxi/dy)*(-rx),xi + (deta/dy)*(-rx),eta
BLkyy = Matrix([[0, 0, 0, -N1y, 0, 0,
                 0, 0, 0, -N2y, 0, 0,
                 0, 0, 0, -N3y, 0, 0,
                 0, 0, 0, -N4y, 0, 0]])
#kxy = phix,y + phiy,x = (dxi/dy)*phix,xi + (deta/dy)*phix,eta
#                       +(dxi/dx)*phiy,xi + (deta/dx)*phiy,eta
#kxy = ry,y + (-rx),x = (dxi/dy)*ry,xi + (deta/dy)*ry,eta
#                       +(dxi/dx)*(-rx),xi + (deta/dx)*(-rx),eta
BLkxy = Matrix([[0, 0, 0, -N1x, N1y, 0,
                 0, 0, 0, -N2x, N2y, 0,
                 0, 0, 0, -N3x, N3y, 0,
                 0, 0, 0, -N4x, N4y, 0]])
#gyz = phiy + w,y
#    = -rx + w,y
#    = rot + grad
BLgyz_rot = Matrix([[0, 0, 0, -N1, 0, 0,
                     0, 0, 0, -N2, 0, 0,
                     0, 0, 0, -N3, 0, 0,
                     0, 0, 0, -N4, 0, 0]])
BLgyz_grad = Matrix([[0, 0, N1y, 0, 0, 0,
                      0, 0, N2y, 0, 0, 0,
                      0, 0, N3y, 0, 0, 0,
                      0, 0, N4y, 0, 0, 0]])
#gxz = phix + w,x
#    = ry + w,x
#    = rot + grad
BLgxz_rot = Matrix([[0, 0, 0, 0, N1, 0,
                     0, 0, 0, 0, N2, 0,
                     0, 0, 0, 0, N3, 0,
                     0, 0, 0, 0, N4, 0]])
BLgxz_grad = Matrix([[0, 0, N1x, 0, 0, 0,
                      0, 0, N2x, 0, 0, 0,
                      0, 0, N3x, 0, 0, 0,
                      0, 0, N4x, 0, 0, 0]])
# for drilling stiffness
#   see Eq. 2.20 in F.M. Adam, A.E. Mohamed, A.E. Hassaballa, Degenerated Four Nodes Shell Element with Drilling Degree of Freedom, IOSR J. Eng. 3 (2013) 10–20. www.iosrjen.org (accessed April 20, 2020).
BLdrilling = Matrix([[N1y/2., -N1x/2., 0, 0, 0, N1,
                      N2y/2., -N2x/2., 0, 0, 0, N2,
                      N3y/2., -N3x/2., 0, 0, 0, N3,
                      N4y/2., -N4x/2., 0, 0, 0, N4]])

ZERO = sympy.zeros(*BLexx.shape)
BLmembrane = Matrix([BLexx, BLeyy, BLgxy, ZERO, ZERO, ZERO])
BLbending = Matrix([ZERO, ZERO, ZERO, BLkxx, BLkyy, BLkxy])
BLtransvshear_grad = Matrix([BLgyz_grad, BLgxz_grad])
BLtransvshear_rot = Matrix([BLgyz_rot, BLgxz_rot])


ABD = Matrix(
        [[A11, A12, A16, B11, B12, B16],
         [A12, A22, A26, B12, B22, B26],
         [A16, A26, A66, B16, B26, B66],
         [B11, B12, B16, D11, D12, D16],
         [B12, B22, B26, D12, D22, D26],
         [B16, B26, B66, D16, D26, D66]])
E = Matrix(
    [[E44, E45],
     [E45, E55]])

var('wij')

# Constitutive linear stiffness matrix
#NOTE reduced integration of stiffness to remove shear locking
#subs(xi=0, eta=0) in many places above was used
KC0e_membrane = wij*detJ*(BLmembrane.T*ABD*BLmembrane)
KC0e_bending = wij*detJ*(BLbending.T*ABD*BLbending)
KC0e_coupled = wij*detJ*(BLmembrane.T*ABD*BLbending +
                         BLbending.T*ABD*BLmembrane)
KC0e_transvshear_rot = wij*detJ*(BLtransvshear_rot.T*E*BLtransvshear_rot)
KC0e_transvshear_grad = wij*detJ*(BLtransvshear_grad.T*E*BLtransvshear_grad)
KC0e_transvshear_coupled = wij*detJ*(BLtransvshear_rot.T*E*BLtransvshear_grad +
                                     BLtransvshear_grad.T*E*BLtransvshear_rot)
KC0e_drilling = wij*detJ*(alphat*A66/h*BLdrilling.T*BLdrilling)

# KC0 represents the global linear stiffness matrix
print()
print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

#NOTE line below to visually check the Rmatrix
#np.savetxt('Rmatrix.txt', R, fmt='% 3s')

KC0_membrane = R*KC0e_membrane*R.T
KC0_bending = R*KC0e_bending*R.T
KC0_coupled = R*KC0e_coupled*R.T
KC0_transvshear_rot = R*KC0e_transvshear_rot*R.T
KC0_transvshear_grad = R*KC0e_transvshear_grad*R.T
KC0_transvshear_coupled = R*KC0e_transvshear_coupled*R.T
KC0_drilling = R*KC0e_drilling*R.T

KC0 = KC0_membrane + KC0_bending + KC0_coupled + KC0_transvshear_rot + KC0_transvshear_grad + KC0_transvshear_coupled + KC0_drilling

def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

print()
print()
print('_______________________________________')
print()
print('printing code for sparse implementation')
print('_______________________________________')
print()
print()
KC0_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    KC0_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KC0r[k] = %d+%s' % (i%DOF, si))
    print('        KC0c[k] = %d+%s' % (j%DOF, sj))
print('KC0_SPARSE_SIZE', KC0_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        KC0v[k] +=', val)
print()
print()
