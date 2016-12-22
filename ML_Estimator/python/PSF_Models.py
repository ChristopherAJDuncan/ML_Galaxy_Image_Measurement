### Weave Declarations -- C++ Implementation ###
"""
Module containing routines defining the PSF model used in the production of images. Uses a C++ back-end using ScipY Weave. 
Author: cajd
"""

def PSFModel_CXX(xy, cen, Param, der = []):
    from copy import deepcopy
    """
    Routine that produces the PSF profile and its derivatives using C++ code called through Weave inline.
    
    Note: As the code is compiled for the first run and then run from precompiled code, it may be the case that the code will be quickest when compiled in seperate routines
    -- Uses the surface_brightness_Profile definitions of Gaussian profiles to evaluate the Guassian PSF model

    Requires:
    -- xy: [2,nGrid] list/array specifying the grid over which the model is evaluated.
    -- cen: centroid position [x,y], defining the centre of the model
    -- Param: model Parameter dictionary specifying PSF model parameters. NOTE: As opposed to the top-level model parameter dictionary, this should be the sub-level `PSF` dictionary
    -- der: List of parameters with with to take derivatives. Number of elements specifies the order of derivative to return.

    """
    
    import numpy as np
    
    nX = xy[0].shape[0]; nY = xy[1].shape[0]
    
    dx = xy[0]-cen[0]
    dy = xy[1]-cen[1]
    SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))
    
    if(cen[0] > xy[0].max() or cen[0] < xy[0].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (x) lies outwith the range of the PS, FATAL :'+str(cen))
    if(cen[1] > xy[1].max() or cen[1] < xy[1].min()):
        raise ValueError('gaussian_SBProfile_Weave - Centroid (y) lies outwith the range of the PS, FATAL :'+str(cen))

    iParam = deepcopy(Param)

    if(iParam['PSF_Type'] == 1 or str(iParam['PSF_Type']).lower() == 'gaussian'):
        ## Elliptical Gaussian PSF: Use known output for SB profile
        ##Rename variables to fit in with Sympy output

        ''' These variables are not used, as the Guassian SB profile routines are instead used, but are left here as instructional
        e1 = ie1
        e2 = ie2
        size = isigma
        flux = 1.0 #Set to one to enusre int(P) = 1 (normalised)
        '''

        ##Use SB profile code
        from surface_Brightness_Profiles import gaussian_SBProfile_CXX
            
        psf = gaussian_SBProfile_CXX(xy, cen, iParam['PSF_size'], iParam['PSF_Gauss_e1'], iParam['PSF_Gauss_e2'], 1.0, der = der)


    return psf.copy()
        




# def PSFModel_Weave(xy, cen, Param, der = []):
#     from copy import deepcopy
#     """
#     Routine that produces the PSF profile and its derivatives using C++ code called through Weave inline.
    
#     Note: As the code is compiled for the first run and then run from precompiled code, it may be the case that the code will be quickest when compiled in seperate routines
#     -- Uses the surface_brightness_Profile definitions of Gaussian profiles to evaluate the Guassian PSF model

#     Requires:
#     -- xy: [2,nGrid] list/array specifying the grid over which the model is evaluated.
#     -- cen: centroid position [x,y], defining the centre of the model
#     -- Param: model Parameter dictionary specifying PSF model parameters. NOTE: As opposed to the top-level model parameter dictionary, this should be the sub-level `PSF` dictionary
#     -- der: List of parameters with with to take derivatives. Number of elements specifies the order of derivative to return.

#     """
    
#     from scipy import weave
#     import numpy as np
    
#     nX = xy[0].shape[0]; nY = xy[1].shape[0]
    
#     dx = xy[0]-cen[0]
#     dy = xy[1]-cen[1]
#     SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))
    
#     if(cen[0] > xy[0].max() or cen[0] < xy[0].min()):
#         raise ValueError('gaussian_SBProfile_Weave - Centroid (x) lies outwith the range of the PS, FATAL :'+str(cen))
#     if(cen[1] > xy[1].max() or cen[1] < xy[1].min()):
#         raise ValueError('gaussian_SBProfile_Weave - Centroid (y) lies outwith the range of the PS, FATAL :'+str(cen))

#     iParam = deepcopy(Param)

#     ##Set up initial bit of code, generic for all PSF models
#     #Pos here accounts for the fact that apparently weave works on flattened arrays
#     code = r"""
#     int i;
#     int j;
#     int Pos;
#     for(i = 0; i < nX; i++){
#     for(j = 0; j < nY; j++){
#     Pos = i*nX + j;
#     SB[Pos] = """
    
#     codeTail = r"""}}"""


#     if(iParam['PSF_Type'] == 1 or str(iParam['PSF_Type']).lower() == 'gaussian'):
#         ## Elliptical Gaussian PSF: Use known output for SB profile
#         ##Rename variables to fit in with Sympy output

#         ''' These variables are not used, as the Guassian SB profile routines are instead used, but are left here as instructional
#         e1 = ie1
#         e2 = ie2
#         size = isigma
#         flux = 1.0 #Set to one to enusre int(P) = 1 (normalised)
    
#         weaveVar = ['SB', 'flux', 'e1', 'e2', 'size', 'dx', 'dy', 'nX', 'nY']

#         weaveArgs = [SB, flux, e1, e2, size, dx, dy, nX, nY, weaveVar, code, codeTail]
#         '''

#         ##Use SB profile code
#         from surface_Brightness_Profiles import gaussian_SBProfile_Weave
            
#         psf = gaussian_SBProfile_Weave(xy, cen, iParam['PSF_size'], iParam['PSF_Gauss_e1'], iParam['PSF_Gauss_e2'], 1.0, der = der)


#     return psf.copy()
        

