"""
Module containing the general routines for the production of a non-Pixelised (etc.) surface brightness profile and its derivatives. If run as a stand-alone program, the code will output the functional form of the profile or it's derivatives as specified, as well as C++-ified output to enable easy implementation of Weave for new profiles. See the bottom of the file for details.

Author: cajd
Touch Date: 16 June 2015
"""

import numpy as np

def gaussian_SBProfile_Sympy(xy, cen, isigma, ie1, ie2, iItot, K = 0., g1 = 0., g2 = 0., der = [], printOnly = False, suppressLensing = True, printStyle = 'CPP'):
    import sympy as sp
    """
    Uses symbolic python pacakge (SymPy) to evaluate function and its derivatives analytically, including symbolic output. As symbolic, this is noticably slow, so should not be used as part of main routine except to debug.

    Requires:
    -- xy: Grid [x,y] over which SB profile is evaluated
    -- cen: Centroid [x,y]
    -- isigma: Size of SB profile
    -- ie1, ie2: Ellipticity components of SB profile
    -- iTOT: Total integrated flux, corresponding to amplitude of profile in Gaussian case.
    -- K, g1, g2: Lensing Parameters
    ---der: Sets the parameters to be differentiated with respect to. The size of each sub-list sets the order of differentiation. e.g. to return SB profile, use der = [] (or leave at default. To get d2I/dr2: der = ['size', 'size'], an To get d2I/drde1: der = ['size', 'e1'] etc.
    -- printOnly: default False - If true, only output the symbolic result to screen. If false, evaluate and return the result
    --suppressLensing: default True -  if True, ignore the lensing contribution
    -- printStyle: default CPP - If CPP, output to screen in C++ form

    """
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
    K = sp.Symbol('K')
    g1 = sp.Symbol('g1')
    g2 = sp.Symbol('g2')

    ##Matrices
    Q = (size**2)*sp.Matrix([[(1-e1), e2],[e2, 1+e1]])
    rvec = sp.Matrix([dx,dy])
    A = (1.-K)*sp.Matrix([[1-g1,-g2],[-g2,1+g1]])

    if(not suppressLensing):
        ##Translate the distrotion matrix to a lensed version
        Q = A.T*Q*A

    Pref = flux/(2.*pi*(Q.det()**0.5)) ##Sqrt detQ
    Surf = Pref*sp.exp(-0.5*rvec.transpose()*Q.inv()*rvec)

    ##Create function of surface brightness profile to specified order of differentiation
    dSurf = Surf
    for dd in range(len(der)):
        print 'Differentiating wrt:', der[dd]
        dSurf = dSurf.diff(der[dd])

    strdSurf = str(dSurf)
    strdSurf = strdSurf[9:-3] ##Cut offf the 'Matrix([[' and ]]) parts

    if printOnly:

        print 'Python Result is:', strdSurf
        raw_input('Check')

        if(printStyle.lower() == 'cpp'):
            strdSurf = convert_CPP(strdSurf)
        
            print '\n Final Result is:', strdSurf
    else:
        
        fSurf = sp.lambdify((dx,dy,size,e1,e2,flux), dSurf)
        
        ##Set up grid on which to evaluate
        delR = [xy[0]-cen[0], xy[1]-cen[1]] #[delX, delY]
        SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))
        
        ## Can this be sped up?
        for i in range(SB.shape[0]):
            for j in range(SB.shape[1]):
                SB[i,j] = fSurf(delR[0][i], delR[1][j], isigma, ie1, ie2, iItot)
                
        return SB

### SymPy helper routines - For C++-style output
def iterativeStringFind(string, sub):
    """
    Helper routine used in converting SymPy output to C++ output. Interatvely search a string for a sub string
    """
    from string import find

    res = []
    index = 0

    end = False
    while not end:
        found = find(string,sub,index)
        if(found == -1):
            end = True
        else:
            res.append(found)
            index = res[-1]+1
    return res

def convert_CPP(string):
    """
    Routine that takes the output from SymPy and converts it to CPP code. This is really taking x**y -> pow(x,y) in a whole load of complecating cases. It was an absolute bugger to code (seriously the worse code Ive had to produce), however it allows one to simply run the SymPy code with printType = CPP, and copy and paste the result directly into a C++ program (including weave implementation used here. You`re welcome.

    author: cajd
    Date:7Aug2015

    Has been verified to work for all combinations up to second derivative for Gaussian SB profile, for parameter set [e1,e2,size]
    """
    
    from string import replace, find
    import numpy as np

    #Find instances of ** -> pow
    subloc = iterativeStringFind(string, '**')

    opSearch = np.array(['*','+', '-', '/', ','])

    ##Search to the left for all the cases where a whole term in brackets is reaised to the power
    Ended = False; counter = 0
    while not Ended:
        #for ii in range(len(subloc)):
        counter += 1
        subloc = iterativeStringFind(string, '**')

        found = 0
        for ii in range(len(subloc)):
            i = subloc[ii] # + 2 to account for the fact we are replacing **


            if(string[i] != '*'):
                raise ValueError('convert_CPP - Error in setting subloc (search to left):',string[i], string[i-2:i+3])

            ##If first character to the left is a close backet, assum that it contains all the terms to be raised to the power

#            print 'looking at:', string[i-1:i+1], i>0, string[i-1] == ')'
#            raw_input('Check this')
            
            if(i > 0 and string[i-1] == ')'):
                found += 1
                j = 0;
                
                inParanthesis = True
                
                nParanthesis = 0

                ##Search to left
                while i+j >= 0 and inParanthesis:
                    j -= 1
                    
                    if(string[i+j] == ')'):
                        nParanthesis += 1
                        
                    elif(string[i+j] == '('):
                        #Found start of paranethesis: replace string
                        #inParanthesis = False
                        nParanthesis -=1
                        
                    if(nParanthesis == 0):
                        inParanthesis == False
                        
                        string = string[:i+j]+'pow'+string[i+j:i-1]+','+string[i+2:]#+')'
                        
                        subloc[ii+1:] += 3*np.ones(len(subloc[ii+1:]), dtype = int)
                        
                        
                        ##Add Paranthesis to right:
                        
                        string =  operatorSearchRight(string, opSearch, i+3)

#                        print 'found check:', found
                        
                        break

        if found == 0:
            ##Exit when there are no more to be found
            Ended = True
            break

    subloc = iterativeStringFind(string, '**')

    ##Search right to the next operator
    for ii in range(len(subloc)):
        i = subloc[ii]+2 # + 2 to account for the fact we are replacing **

        if(string[i-2] != '*'):
            raise ValueError('subloc index issue in search to right', string[i], string[i-2:i+1])

        string =  operatorSearchRight(string, opSearch, i)
        subloc =  iterativeStringFind(string, '**')


    ##Subloc is not out of date

    ##Do initial replace to left for most obvious cases
    expSearch = ['e1', 'e2', 'size', 'dy', 'dx']
    for expVal in expSearch:
        string = string.replace( expVal+'**', 'pow('+expVal+',')
            

    ## Finally, replace dx -> dx[i], dy -> dy[j]
    string = string.replace( 'dx', 'dx[i]')
    string = string.replace('dy', 'dy[j]')


    if(find(string, '**') != -1):
        print 'Error - ** found in string'
        
    return string


def operatorSearchRight(string, opSearch, index):
    """
    Helper routine in converting SymPy output to C++ style output.
    Routine that starts from index entered, searches to right to the next operator, as defined in opSearch, and then places an extra `)` before that operator. Used to bracket pow functions to the right, accounting for the case were x**(some function) [parantheses important here]
    """


    j = -1; found = False
    i = index
    inParanthesis = False
    #Search forward form position
    while i+j <= len(string)-2 and not found:
        j += 1
        
        if(string[i+j] == '('):
            #Susped search until paranthesis closed
            inParanthesis = True
        if(inParanthesis):
            if(string[i+j] == ')'):
                inParanthesis = False
            continue
        
        match = np.array(opSearch == string[i+j])
        
        if(sum(match) == 1):
            string = string[:i+j]+')'+string[i+j:]
            #string[i+j] = ')'+ string[i+j]
            found = True

    if(not found):
        #Close on end of string
        string = string+')'

    return string


def init_SRCPath():
    """                                                                               
    Initialises the sys path to include the directory where the run modules are stored. Ideally this should not be necessary                                                                                 
    """
    import os
    import sys

    #Define src path as current files path, up on level, src/
    srcPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    srcPath = os.path.join('/',srcPath,'src/')
    
    if not srcPath in sys.path:
        sys.path.insert(1,srcPath)
    del srcPath

    
### C++ implemnations of surface brightness profiles and derivatives
def gaussian_SBProfile_CXX(xy, cen, isigma, ie1, ie2, iItot, der = []):
    """
    Routine that produces the SB profile and its derivatives using C++ code..

    Note: As the code is compiled for the first run and then run from precompiled code, it may be the case that the code will be quickest when compiled in seperate routines

    The evaluation of the SB profile and its derivatives in the presence of a lensing field is involved using this method, therefore it has been neglected for now (The application of the method itself does not require lensing paramaters where the shear is taken as the average across a sample [e.g. ring test], however the simulation of sheared images may require this. In this case, only the lensed surface brightness profile itself is required, and this may be more easily implemented in the full matrix formalism of gaussian_SBProfile_Py in Python (numpy)

    Required:
    -- xy: Grid [x,y] over which SB profile is evaluated
    -- cen: Centroid [x,y]
    -- isigma: Size of SB profile
    -- ie1, ie2: Ellipticity components of SB profile
    -- iTOT: Total integrated flux, corresponding to amplitude of profile in Gaussian case.
    ---der: Sets the parameters to be differentiated with respect to. The size of each sub-list sets the order of differentiation. e.g. to return SB profile, use der = [] (or leave at default. To get d2I/dr2: der = ['size', 'size'], an To get d2I/drde1: der = ['size', 'e1'] etc.

    Returns:
    --SB: surface brightness profile evaluated according to input parameters and derivative labels, on grid specified by xy.
    """
    
    import numpy as np

    dx = xy[0]-cen[0]
    dy = xy[1]-cen[1]

    if(cen[0] > xy[0].max() or cen[0] < xy[0].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (x) lies outwith the range of the PS, FATAL :'+str(cen))
    if(cen[1] > xy[1].max() or cen[1] < xy[1].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (y) lies outwith the range of the PS, FATAL :'+str(cen))

    ##Rename variables to fit in with Sympy output
    e1 = ie1
    e2 = ie2
    size = isigma
    flux = iItot

    #Depreacted?
    Var = ['SB', 'flux', 'e1', 'e2', 'size', 'dx', 'dy']
    Args = [float(flux), float(e1), float(e2), float(size), dx, dy]

    #-- NOTE : If this throws an exception, its likely that the addition of the src path to sys in init did not go well
    init_SRCPath()
    import surface_Brightness_Profiles_CXX as cxxSB
    
    if(len(der) == 0):
        ##SB profile directly
        SB = cxxSB.cxx_GaussSB(*Args)

    elif(der == ['flux']):
        Args[0] = 1.
        SB = cxxSB.cxx_GaussSB(*Args)

    elif(der == ['size']):
        SB = cxxSB.cxx_GaussSB_dT(*Args)

    elif(der == ['e1']):
        SB = cxxSB.cxx_GaussSB_de1(*Args)

    elif(der == ['e2']):
        SB = cxxSB.cxx_GaussSB_de2(*Args)

    elif(der == ['e1','e1']):
        SB = cxxSB.cxx_GaussSB_dde1(*Args)

    elif(der == ['e2','e2']):
        SB = cxxSB.cxx_GaussSB_dde2(*Args)

    elif(der == ['size','size']):
        SB = cxxSB.cxx_GaussSB_ddT(*Args)

    elif(der == ['flux', 'flux']):
        #Do nothing as SB is zeros
        SB = np.zeros((dx.shape[0], dy.shape[0]))

    elif(der == ['size', 'flux'] or der == ['flux', 'size']):
        #Use the fact that d/dflux just takes flux -> 1 (for guassian which is linear in flux)
        Args[0] = 1.
        SB = cxxSB.cxx_GaussSB_dT(*Args)

    elif(der == ['size','e1'] or der == ['e1','size'] ):
        SB = cxxSB.cxx_GaussSB_de1dT(*Args)

    elif(der == ['size','e2'] or der == ['e2','size'] ):
        SB = cxxSB.cxx_GaussSB_de2dT(*Args)

    elif(der == ['e1','e2'] or der == ['e2','e1'] ):
        SB = cxxSB.cxx_GaussSB_de1de2(*Args)

    else:
        raise RuntimeError('CXX run not coded up for derivative:',str(der))

    #Cast into numpy form
    SB = np.array(SB)
    SB = SB.reshape((dx.shape[0], dy.shape[0]))
    
    return SB.copy()



### Weave Declarations -- C++ Implementation ###

def gaussian_SBProfile_Weave(xy, cen, isigma, ie1, ie2, iItot, der = []):
    """
    Routine that produces the SB profile and its derivatives using C++ code called through Weave inline.

    Note: As the code is compiled for the first run and then run from precompiled code, it may be the case that the code will be quickest when compiled in seperate routines

    The evaluation of the SB profile and its derivatives in the presence of a lensing field is involved using this method, therefore it has been neglected for now (The application of the method itself does not require lensing paramaters where the shear is taken as the average across a sample [e.g. ring test], however the simulation of sheared images may require this. In this case, only the lensed surface brightness profile itself is required, and this may be more easily implemented in the full matrix formalism of gaussian_SBProfile_Py in Python (numpy)

    Required:
    -- xy: Grid [x,y] over which SB profile is evaluated
    -- cen: Centroid [x,y]
    -- isigma: Size of SB profile
    -- ie1, ie2: Ellipticity components of SB profile
    -- iTOT: Total integrated flux, corresponding to amplitude of profile in Gaussian case.
    ---der: Sets the parameters to be differentiated with respect to. The size of each sub-list sets the order of differentiation. e.g. to return SB profile, use der = [] (or leave at default. To get d2I/dr2: der = ['size', 'size'], an To get d2I/drde1: der = ['size', 'e1'] etc.

    Returns:
    --SB: surface brightness profile evaluated according to input parameters and derivative labels, on grid specified by xy.
    """
    
    from scipy import weave
    import numpy as np

    raise RuntimeError("Weave implemnation has been deprecated. CXX implemation should be used instead")
    
    nX = xy[0].shape[0]; nY = xy[1].shape[0]

    dx = xy[0]-cen[0]
    dy = xy[1]-cen[1]
    SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))

    if(cen[0] > xy[0].max() or cen[0] < xy[0].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (x) lies outwith the range of the PS, FATAL :'+str(cen))
    if(cen[1] > xy[1].max() or cen[1] < xy[1].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (y) lies outwith the range of the PS, FATAL :'+str(cen))

    ##Rename variables to fit in with Sympy output
    e1 = ie1
    e2 = ie2
    size = isigma
    flux = iItot

    weaveVar = ['SB', 'flux', 'e1', 'e2', 'size', 'dx', 'dy', 'nX', 'nY']

    ##Set up initial bit of code
    #Pos here accounts for the fact that apparently weave works on flattened arrays
    code = r"""
    int i;
    int j;
    int Pos;
    for(i = 0; i < nX; i++){
    for(j = 0; j < nY; j++){
    Pos = i*nX + j;
    SB[Pos] = """

    codeTail = r"""}}"""

    weaveArgs = [SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail]

    if(len(der) == 0):
        ##SB profile directly
        runWeave_GaussSB(*weaveArgs)

    elif(der == ['flux']):
        weaveArgs[1] = 1.
        runWeave_GaussSB(*weaveArgs)

    elif(der == ['size']):
        runWeave_GaussSB_dT(*weaveArgs)

    elif(der == ['e1']):
        runWeave_GaussSB_de1(*weaveArgs)

    elif(der == ['e2']):
        runWeave_GaussSB_de2(*weaveArgs)

    elif(der == ['e1','e1']):
        runWeave_GaussSB_dde1(*weaveArgs)

    elif(der == ['e2','e2']):
        runWeave_GaussSB_dde2(*weaveArgs)

    elif(der == ['size','size']):
        runWeave_GaussSB_ddT(*weaveArgs)

    elif(der == ['flux', 'flux']):
        #Do nothing as SB is zeros
        SB[:,:] = 0.

    elif(der == ['size', 'flux'] or der == ['flux', 'size']):
        #Use the fact that d/dflux just takes flux -> 1 (for guassian which is linear in flux)
        weaveArgs[1] = 1.
        runWeave_GaussSB_dT(*weaveArgs)

    elif(der == ['size','e1'] or der == ['e1','size'] ):
        runWeave_GaussSB_de1dT(*weaveArgs)

    elif(der == ['size','e2'] or der == ['e2','size'] ):
        runWeave_GaussSB_de2dT(*weaveArgs)

    elif(der == ['e1','e2'] or der == ['e2','e1'] ):
        runWeave_GaussSB_de1de2(*weaveArgs)

    else:
        raise RuntimeError('weave run not coded up for derivative:',str(der))
        
    return SB.copy()

### ~~~~~~~~~ Support routines for Weave: Take the common code and append the C++ code to calcualte the SB and its derivatives. These are kept in seperate routines so that the correct precompiled version is used every time and not overwritten with subsequent calls

### --- Probably want to do this for lensing parameters also...

### --- To convert from SymPy output to C++ code, one must replace x**y -> pow(x,y), and dx -> dx[i], dy -> dy[j]

## Could a dictionary be used for input, since weaveVar is essentially the `key` of the entered parameters?


''' TEMPLATE FOR RUN WEAVE DECLARATIONS 

def runWeave_GaussSB_{...}(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave
    ##SB profile directly
    codeSB = r""" ... ;"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

'''

#### ---- 0th order

def runWeave_GaussSB(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave
    ##SB profile directly
    codeSB = r"""0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4) - pow(e2,2)*pow(size,4) + pow(size,4),-0.5)*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2) - 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2) + pow(e2,2) - 1.0)));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])


#### ---- 1st order

def runWeave_GaussSB_dT(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    ##This has been compared to direct SymPy output and is fine
    from scipy import weave

    codeSB = r"""0.159154943091895*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*pow((-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4)),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+pow(e2,2 )- 1.0)); """

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

def runWeave_GaussSB_de1(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.159154943091895*e1*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))+ 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) ;"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

def runWeave_GaussSB_de2(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.159154943091895*e2*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

#### ---- 2nd Order

## o-o-o-o Auto-terms

def runWeave_GaussSB_dde1(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.477464829275686*pow(e1,2)*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.318309886183791*e1*flux*pow(size,4)*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)),2)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(4.0*pow(e1,2)*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 2.0*e1*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )- 1.0*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

def runWeave_GaussSB_dde2(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.477464829275686*pow(e2,2)*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.318309886183791*e2*flux*pow(size,4)*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(size,4)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)),2)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(4.0*dx[i]*dy[j]*e2/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 4.0*pow(e2,2)*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 1.0*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])


def runWeave_GaussSB_ddT(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.159154943091895*flux*(6.0*pow(e1,2)*pow(size,2 )+ 6.0*pow(e2,2)*pow(size,2 )- 6.0*pow(size,2))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.318309886183791*flux*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 0.477464829275686*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,4)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*pow(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2),2)*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,6)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])


#-o-o-o-o Cross terms 2nd order

def runWeave_GaussSB_de1dT(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r""" 0.159154943091895*e1*flux*pow(size,4)*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.636619772367581*e1*flux*pow(size,3)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*e1*flux*size*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(e1,2 )+ pow(e2,2 )- 1.0) + 0.159154943091895*flux*(2.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,3)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )- 1.0*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])


def runWeave_GaussSB_de2dT(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.159154943091895*e2*flux*pow(size,4)*(6.0*pow(e1,2)*pow(size,3 )+ 6.0*pow(e2,2)*pow(size,3 )- 6.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.636619772367581*e2*flux*pow(size,3)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*e2*flux*size*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(e1,2 )+ pow(e2,2 )- 1.0) + 0.159154943091895*flux*(2.0*dx[i]*dy[j]/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) + 2.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,3)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*(2.0*pow(e1,2)*pow(size,3 )+ 2.0*pow(e2,2)*pow(size,3 )- 2.0*pow(size,3))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) - 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))/(pow(size,3)*(pow(e1,2 )+ pow(e2,2 )- 1.0));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

def runWeave_GaussSB_de1de2(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.477464829275686*e1*e2*flux*pow(size,8)*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-2.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*e1*flux*pow(size,4)*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*e2*flux*pow(size,4)*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-1.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*(-1.0*dx[i]*dy[j]/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)) - 1.0*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*(-1.0*e1*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 0.5*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)))*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0))) + 0.159154943091895*flux*pow(-pow(e1,2)*pow(size,4 )- pow(e2,2)*pow(size,4 )+ pow(size,4),(-0.5))*(2.0*dx[i]*dy[j]*e1/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2) )+ 4.0*e1*e2*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,3) )- 1.0*e2*(pow(dx[i],2 )- pow(dy[j],2))/(pow(size,2)*pow(pow(e1,2 )+ pow(e2,2 )- 1.0,2)))*exp(0.5*(pow(dx[i],2)*e1 + pow(dx[i],2 )- 2.0*dx[i]*dy[j]*e2 - pow(dy[j],2)*e1 + pow(dy[j],2))/(pow(size,2)*(pow(e1,2 )+ pow(e2,2 )- 1.0)));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])

def runWeave_GaussSB_dTdF(SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail):
    from scipy import weave

    codeSB = r"""0.159154943091895*(2.0*e1**2*size**3 + 2.0*e2**2*size**3 - 2.0*size**3)*(-e1**2*size**4 - e2**2*size**4 + size**4)**(-1.5)*exp(0.5*(dx**2*e1 + dx**2 - 2.0*dx*dy*e2 - dy**2*e1 + dy**2)/(size**2*(e1**2 + e2**2 - 1.0))) - 0.159154943091895*(-e1**2*size**4 - e2**2*size**4 + size**4)**(-0.5)*(dx**2*e1 + dx**2 - 2.0*dx*dy*e2 - dy**2*e1 + dy**2)*exp(0.5*(dx**2*e1 + dx**2 - 2.0*dx*dy*e2 - dy**2*e1 + dy**2)/(size**2*(e1**2 + e2**2 - 1.0)))/(size**3*(e1**2 + e2**2 - 1.0));"""

    code += codeSB + codeTail
    ## Run the code through weave
    weave.inline(code, weaveVar, headers = ['<cmath>'])


#

##### ----------------------------------- Command Line Run: Set up to run SymPy version with print output, thus allowing the scalar form for the SB profile and its derivatives to be calcualted easily for inclusion as C++ code -------------------------------#####

if __name__ == "__main__":
     ##Contains the routines if the code is run from command line, most likely to output SymPy result

    import sys
    nargs = len(sys.argv)
    if(nargs >= 5):
        isigma, ie1, ie2, iItot = sys.argv[1:5]
    else:
        isigma, ie1, ie2, iItot = 0.0, 0.0, 0.0, 0.0
    if(nargs >= 6):
        der = sys.argv[5:]
    else:
        der = ['flux', 'size']

    print 'Producing SB profile for Size, E1, E2, Flux:', isigma, ie1, ie2, iItot
    print 'With derivative:', der
    
    gaussian_SBProfile_Sympy(0.0, 0.0, isigma, ie1, ie2, iItot, der = der, printOnly = True, printStyle = 'CPP')
