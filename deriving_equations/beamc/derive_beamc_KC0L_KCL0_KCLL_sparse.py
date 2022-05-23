"""
Constitutive nonlinear stiffness matrix for BFS cylinder with Sanders-type kinematics
"""
import numpy as np
import sympy
from sympy import var, Matrix, symbols, simplify

num_nodes = 4
cpu_count = 6
DOF = 10

def main():
    var('xi, eta, lex, ley, rho, weight')
    var('R')
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

    # membrane
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
                print('                Bm%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(Bmij))))
                Bmis.append(symbols('Bm%d_%02d' % (i+1, j+1)))
            else:
                Bmis.append(0)
        Bms.append(Bmis)
    Bm = sympy.Matrix(Bms)

    print()
    print()
    print()

    uG = Matrix([symbols(r'ue[%d]' % i) for i in range(0, Bm.shape[1])])
    Nw_x = (2/lex)*Nw.diff(xi)
    Nw_y = (2/ley)*Nw.diff(eta)
    v = Nv*uG
    w_x = Nw_x*uG
    w_y = Nw_y*uG
    print('v =', simplify(v)[0, 0])
    print('w_x =', simplify(w_x)[0, 0])
    print('w_y =', simplify(w_y)[0, 0])
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
                print('                BmL%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(BmLij))))
                BmLis.append(symbols('BmL%d_%02d' % (i+1, j+1)))
            else:
                BmLis.append(0)
        BmLs.append(BmLis)
    BmL = Matrix(BmLs)

    print()
    print()
    print()

    # bending
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
                print('                Bb%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(Bbij))))
                Bbis.append(symbols('Bb%d_%02d' % (i+1, j+1)))
            else:
                Bbis.append(0)
        Bbs.append(Bbis)
    Bb = Matrix(Bbs)

    print()
    print()
    print()

    # Constitutive nonlinear stiffness matrix in element coordinate KCNLe
    KC0Le = Bm.T*A*BmL + Bb.T*B*BmL
    KCL0e = BmL.T*A*Bm + BmL.T*B*Bb
    KCLLe = BmL.T*A*BmL
    KCNLe = weight*(lex*ley)/4.*(KC0Le + KCL0e + KCLLe)

    # KCNL represents the global constitutive nonlinear stiffness matrix
    # in case we want to apply coordinate transformations
    KCNL = KCNLe

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

    print('printing code for sparse implementation')
    for ind, val in np.ndenumerate(KCNL):
        if val == 0:
            continue
        print('                k += 1')
        print('                KCNLv[k] +=', KCNL[ind])

    print()
    print()
    print()
    KCNL_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(KCNL):
        if val == 0:
            continue
        KCNL_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        KCNLr[k] = %d+%s' % (i%DOF, si))
        print('        KCNLc[k] = %d+%s' % (j%DOF, sj))
    print('KCNL_SPARSE_SIZE', KCNL_SPARSE_SIZE)

if __name__ == '__main__':
    main()
