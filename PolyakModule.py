# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#                               Polyak Algorithm with constant coefficients
'''

from numpy.linalg import norm


def Polyak(u0, v0, h, eps, E, gradE, N, gamma)  :
    n = 0
    itLst = [n]
    u = [u0]
    v = [v0]
    ok = 0
    while ok == 0 :
        n = n + 1
        itLst.append( n )
        if N == 2 :
            modifGrad_u = [i * h for i in gradE(u[-1])]
        else :
            modifGrad_u = h * gradE(u[-1])
        v.append( ( 1 - h * gamma ) * v[-1] - modifGrad_u )
        u.append( u[-1] + h * v[-1] )
        if norm(gradE(u[-1])) <= eps :
            ok = 1
    return itLst, u, v   