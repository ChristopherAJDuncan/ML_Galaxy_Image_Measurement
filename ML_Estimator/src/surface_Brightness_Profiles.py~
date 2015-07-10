'''
Module containing the general routines for the production of a non-Pixelised (etc.) surface brightness profile and its derivatives. This particualr version uses sympy to producederivatives of an anlytic function.

Author: cajd
Touch Date: 16 June 2015
'''

import numpy as np

def gaussian_SBProfile_Sympy(xy, cen, isigma, ie1, ie2, iItot, der = []):
    import sympy as sp
    '''

    ---der: Sets the parameters to be differentiated with respect to. The size of each sub-list sets the order of differentiation. e.g. to return SB profile, use der = [] (or leave at default. To get d2I/dr2: der = ['size', 'size'], an To get d2I/drde1: der = ['size', 'e1'] etc.

    '''
    ## Edit this to pass in model parameters
    from math import pi

    ## Set up sympy definition of the profile

    #Symbols
    size = sp.Symbol('size')
    e1 = sp.Symbol('e1')
    e2 = sp.Symbol('e2')
    flux = sp.Symbol('flux')
    dx = sp.Symbol('dx')
    dy = sp.Symbol('dy')

    ##Matrices
    Q = (size**2)*sp.Matrix([[(1-e1), e2],[e2, 1+e1]])
    rvec = sp.Matrix([dx,dy])

    Pref = flux/(2.*pi*(Q.det()**0.5)) ##Sqrt detQ
    Surf = Pref*sp.exp(-0.5*rvec.transpose()*Q.inv()*rvec)

    ##Create function of surface brightness profile to specified order of differentiation
    dSurf = Surf
    for dd in range(len(der)):
        print 'Differentiating wrt:', der[dd]
        dSurf = dSurf.diff(der[dd])
    fSurf = sp.lambdify((dx,dy,size,e1,e2,flux), dSurf)

    ##Set up grid on which to evaluate
    delR = [xy[0]-cen[0], xy[1]-cen[1]] #[delX, delY]
    SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))

    ## Can this be sped up?
    for i in range(SB.shape[0]):
        for j in range(SB.shape[1]):
            SB[i,j] = fSurf(delR[0][i], delR[1][j], isigma, ie1, ie2, iItot)

    return SB
    


