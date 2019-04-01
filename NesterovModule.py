# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#                               Nesterov Algorithm
'''

from numpy.linalg import norm


def Nesterov(u0, v0, h, eps, E, gradE, N) :
    n = 0
    itLst = [n]
    u = [u0]
    v = [v0]
    ok = 0
    while ok == 0 :
        n = n + 1
        itLst.append( n )
        beta = (n-1)/(n+2)
        y = u[-1] + h * beta * v[-1]
        if N == 2 :
            modifGrad_y = [i * h for i in gradE(y)]
        else :
            modifGrad_y = h * gradE(y)
        v.append( beta * v[-1] - modifGrad_y )
        u.append( u[-1] + h * v[-1] )
        if norm(gradE(u[-1])) <= eps :
            ok = 1
    return itLst, u, v   
