"""
Module that contains the general routines for production of the pixelised (and potentially noisy) surface brightness models and its derivatives. 
Includes routines to utilise GALSIM, however testing confirms that this does not behave well in producing derivatives, as well as a `user-defined` 
pixelised model production routine that performs the convolutions natively. Testing confirms that the latter is well behaved and produced sensible results.

NOTE: This routine sets up the default model parameters as well as the means to produce the model images. 
It does this using individual subroutines and function, however this is a situation ripe for the use of a defined class. 
This is left to future work if desired.

Author: cajd
Touch Date: 28 May 2015
"""
import numpy as np
from copy import deepcopy

debug = False

##--------------Model Dictionary Manipulation------------------------##

def default_ModelParameter_Dictionary(**setters):
    """
    Returns the default parameter dictionary used in setting up the model image. Dictionary keyword-value pairs can be passed in individually to overwrite defaults. This is essentially a *setter*.

    Model Parameters:
    -- centroid
    -- noise
    -- SNR
    -- stamp_size
    -- pixel_scale: Used in GALSIM implementation, assumed to be one in user implementation (where everything is done in pixels)
    
    --PSF Dictionary (PSF):
    --- PSF_Type: Corresponds to the PSF model used in producing the image
    ____ 0: No PSF
    ____ `Gaussian` or 1: Gaussian
    --- PSF_size
    --- PSF_Gauss_e1
    --- PSF_Gauss_e2

    -- Surface Brightness Profile Dictionary (SB):
    --- modelType: (case insensitive)
    ___ 'Gaussian'
    --- size
    --- e1
    --- e2
    --- flux
    --- bg : Background *per pixel*
    LENSING PARAMETERS (IGNORED IN THIS ITERATION)
    --- magnification
    --- shear
    """

    ### PSF Declaration
    PSFDict = dict(PSF_Type = 0, PSF_size = 0.05, PSF_Gauss_e1 = 0.0, PSF_Gauss_e2 = 0.0)

    ## SB Declaration
    SBDict = dict(modelType = 'gaussian', size = 2, e1 = 0., e2 = 0., flux = 10, magnification = 1., shear = [0., 0.], bg = 0.)

    imgshape = np.array([30, 30]) # Changed  orginally ([10,10])           
    dct = dict(centroid = (np.array(imgshape)+1)/2., noise = 1., SNR = 20, stamp_size = imgshape, pixel_scale = 1., SB = SBDict, PSF = PSFDict) 

    ## Use this to set SB and PSF parameters - RECURSIVE
    ##set_modelParameter(dct, setters.keys(), setters.values())

    update_Dictionary(dct, setters) # Combines the dictionaries

    return dct


def unpack_Dictionary(dic, requested_keys = None):
    from generalManipulation import makeIterableList
    """
    Helper routine which returns a list of dictionary values corresponding to the list of requested keys input. If no keys are input, 
    the full list of values corresponding to the full dictionary keys list (in stored order) is returned. Used to extract model parameters. 
    Automatically searchs all sub-directory levels defined (hardwired to SB and PSF only)
    """

    ##This could be generalised if a list of subdicts (currently SB and PSF could be passed), or inferred

    if(requested_keys is None):
        #This could be improved by querying intelligently, e.g. getting list of keys, then checking each as dictionary, then replacing (remove + concat) if dictionary
        requested_keys = dic['SB'].keys()
        requested_keys += dic['PSF'].keys()
    elif(not hasattr(requested_keys, "__iter")):
        requested_keys = makeIterableList(requested_keys)

    ## Set SB Keys
    rescount = 0; res = ['F']*len(requested_keys)
    for k in requested_keys:
        if((np.array(dic['SB'].keys()) == k).sum() > 0): #SB Parameter
            res[rescount] = dic['SB'][k]; rescount += 1
        elif((np.array(dic['PSF'].keys()) == k).sum() > 0): #PSF Parameter
            res[rescount] = dic['PSF'][k]; rescount += 1
        #elif( (np.array(dic.keys()) == k).sum() > 0): #Other Parameter
        #    res[rescount] = dic[k]; rescount += 1
    return res[:rescount]

def print_ModelParameters(dic):

    subList = ["SB", "PSF"]

    for sub in subList:
        #!! Generalise this
        requested_keys = dic[sub].keys()
        vals = unpack_Dictionary(dic,requested_keys)

        print "\n --", sub, ":"
        for i,par in enumerate(requested_keys):
            print "-----", par, " : ", vals[i]

            
def update_Dictionary(d, u):
    """
    Recurively updates a dictionary (d) and all subdictionaries with input dictionary (u). Taken from StackExchange
    """
    import collections
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_Dictionary(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def seperate_Keys_byModel(der, vals = None, refParam = None):
    from copy import deepcopy
    """
    Takes as input a list which contains the labels of all the parameters considered, and seperates into two lists corresponding to SurfaceBrightness (SB), 
    and PSF models (PSF)
    Ignores *others* for now (anything not PSF or SB)

    Requires:
    -- der: list of parameter labels beign queried
    -- vals: optional (default None): Values for the parameters being queried. If not None, these are similarly split into lists corresponding to SB and PSF. Output is then SBPar, PSFPar, SBVal, PSFVal where each list contains identified parameters
    -- refParam: optional (default None) Reference dictionary used to identify the parameters (i.e. parameters in der checked against this dictionary). If None, default dictionary is used.
    """

    if(refParam is None):
        ##Use default dictionary
        Params = default_ModelParameter_Dictionary()
    else:
        Params = deepcopy(refParam)

    ##Get Reference PSF Keys
    refPSF = np.array(Params['PSF'].keys())
    ##Get Reference SB Keys
    refSB = np.array(Params['SB'].keys())

    PSFDer, SBDer = ['F']*len(der), ['F']*len(der)
    PSFVal, SBVal = ['F']*len(der), ['F']*len(der)
    PSFcount, SBcount = 0, 0
    for dd in range(len(der)):
        if( sum((refPSF == der[dd])) > 0):
            ##der is a PSF parameter
            PSFDer[PSFcount] = der[dd]
            if(vals is not None):
                PSFVal[PSFcount] = vals[dd]
            PSFcount += 1
        elif( sum((refSB == der[dd])) > 0):
            SBDer[SBcount] = der[dd]
            if(vals is not None):
                SBVal[SBcount] = vals[dd]
            SBcount += 1

    ##Isolate those subsets of der which have been assigned
    PSFDer = PSFDer[:PSFcount]
    SBDer = SBDer[:SBcount]
    PSFVal = PSFVal[:PSFcount]
    SBVal = SBVal[:SBcount]
    

    if(vals is not None):
        return SBDer, PSFDer, SBVal, PSFVal
    else:
        return SBDer, PSFDer


def set_modelParameter(Dict, param_labels, param_values):
    from copy import deepcopy
    from generalManipulation import isIterableList, makeIterableList
    """
    Sets model Parameters (as defined above) by input lists. As opposed to update_Dictionary, this routine allows for the keys to be put in without 
    defining which sub-dictionary they belong to, as seperate_Keys_byModel seperates out into sub-dictionaries specified in the default declaration

    Requires:
    -- Dict: model Parameter dictionary to be modified
    -- param_labels: list of parameter labels to be modified in Dict
    -- param_values: list of values corresponding to parameters in param_labels.
    """

    if(not (isIterableList(param_labels) == isIterableList(param_values))):
        raise ValueError('set_modelParameter - Both  param_labels, param_values must be lists or scalar')

    iparam_labels = makeIterableList(param_labels); iparam_values = makeIterableList(param_values)

    if(len(iparam_labels) != len(iparam_values)):
        raise RuntimeError('set_modelParameter - labels and lists not conformal')

    ## Seperate out into respective parts (SB/PSF)
    SBLab, PSFLab, SBVals, PSFVals = seperate_Keys_byModel(iparam_labels, iparam_values, refParam = default_ModelParameter_Dictionary())

    for i, s in enumerate(SBLab):
        Dict['SB'][s] = SBVals[i]
    for i, s in enumerate(PSFLab):
        Dict['SB'][s] = PSFVals[i]

##-------------------------Model Production-----------------------------------------##
def get_Pixelised_Model_wrapFunction(x, Params, xKey, returnOrder = 1, **kwargs):
    from copy import deepcopy
    """
    Wrapper function for get Pixelised model, which returns the image according to Params, where parameter with key 'xKey' is set to value 'x'. 
    By default uses the native pixelised model image production routine.

    Requires:
    --x: Values to be set in dictionary Params
    --Params: model dictionary used to specify model image to be produced
    --xKey: label, or list of labels, corresponding to values in x, to be modified in Params. Must be defined as in default dictionary.
    --returnOrder: if == 1, only the porduced image is returned (default). If == 2, both the image and modified parameter dictionary returned.

    Returns:
    -- model image (always), altered model Params (if returnOrder /= 1)

    """

    ##Store params value so that original is not overwritten - should this be the case?
    #iParams = Params.copy() ## DEPRECATED FOR NOW as it makes sense that we would want to overwrite this
    if xKey is not None and x is not None:
        ##Make iterabel if not already
        try:
            iter(xKey)
        except TypeError:
            xKey = [xKey]
        try:
            iter(x)
        except TypeError:
            x = [x]

        set_modelParameter(Params, xKey, x)

        ''' Deprecated for above
        for k,Key in enumerate(xKey):
            if(Key not in Params):
                raise ValueError('get_Pixelised_Model_wrapFunction - Key entered (',Key, ') is not contained in model parameter library:', Params.keys())
            Params[Key] = x[k]
        '''
    else:
        print 'get_Pixelised_Model_wrapFunction - x and xKey not passed'

#    image, Params =
    result = user_get_Pixelised_Model(Params, **kwargs)
    image = result[0]; Params = deepcopy(result[1])

    if returnOrder == 1:
        return image
    else:
        return image, Params


def SNR_Mapping(model, var = None, SNR = None):
    """
    Uses GREAT 08 filter-matched version of SNR, consistent with GALSIM definition. Model passed in should be noise-free. 
    If SNR is passed, std of pixel noise is returned. If var is passed (must be variance of pixel noise), SNR is returned. One but not both must be passed.
    """
    if(var is None and SNR is not None):
        return np.sqrt(np.power(model,2.).sum()/(SNR*SNR))
    elif(SNR is None and var is not None):
        return np.sqrt(np.power(model,2.).sum()/var)
    else:
        raise ValueError('SNR_Mapping - Either noise variance or SNR must be entered')

### ---------------------------------------------------------------- Model Production - Lookup Table --------------------------------------------------------------------------------------------------------- ###

def get_Model_Lookup(setParams, pLabel, pRange, dP, **modelFuncArgs):
    """
    Create a lookup table for the model creation - Useful only for the 1D or 2D case, where this corresponds to a significant decrease in run-time 
    (or any case where range/dP < nEval*nGal [nEval - function evaluations to get ML point; nGal - number of ML points] 

    Requires:
    --- setParams - dictionary containign the default value for all other model parameters
    --- pLabel - string labelling the parameter being fit (that which the lookup grid is evaluated
    --- pRange - the range over which the model is evaluated
    --- dP - the step size of the evaluation
    --- modelFuncArgs: input dictionary of arguements required of the model image production routine.

    Returns:
    --- lookup dictionary: Dictionary containing details of the lookup table. Includes:
    ___useLookup: Default True. If true, use the information contained in the lookup construct
    ___Grid: list of grid arrays specifying the parameter values on which the lookup was constructed
    ___Images: [[nGrid]*nPar, [nPix, nPix]] <ndarray> the constructed model images evaluated over set parameter grid
    ___width: list of widths over which parameter grids are constructed (corresponding to image)
    ___nP: number of free parameters over which the lookup is constructed
    ___interp: Sets whether index matching or linear interpolation is used in returning model. Interpolation is only implemented for nP = 1.

    Note: If one cares, this and return Model lookup could be defined in a class.
    -- Code is not set up to use models outside the range specified by the lookup table
    """

    print '\n Constructing Model Lookup Table \n'

    Params = deepcopy(setParams)

    nPar = 1
    if isinstance(pLabel, list):
        nPar = len(pLabel)
    if(nPar > 2):
        raise RuntimeError('get_Model_Lookup - Maximum of 2 parameters are allowed as part of a lookup')

    ##Create lists where appropriate
    if(isinstance(dP, list) == False):
        idP = [dP]
    else:
        idP = dP
    
    if(len(idP) != nPar):
        raise RuntimeError('get_Model_Lookup - dP (parameter width) is not conformal with number of parameters to vary', str(dP), ':', str(nPar))

    ## Create the actual lookup table
    if nPar == 1:
        pGrid = np.arange(pRange[0], pRange[1]+(0.5*idP[0]), idP[0])

        images = [1.]*pGrid.shape[0]
        for pp in range(pGrid.shape[0]):
            set_modelParameter(Params, pLabel[0], pGrid[pp])
            #Deprecated Params[pLabel[0]] = pGrid[pp]
            images[pp] = user_get_Pixelised_Model(Params, **modelFuncArgs)[0]
    elif nPar == 2:
        pGrid = []
        for gg in range(nPar):
            pGrid.append(np.arange(pRange[gg][0], pRange[gg][1]+(0.5*idP[gg]), idP[gg]))
                
        images = [[1.]*pGrid[1].shape[0] for i in range(pGrid[0].shape[0])]
        for pp in range(pGrid[0].shape[0]):
            for qq in range(pGrid[1].shape[0]):
                set_modelParameter(Params, [pLabel[0], pLabel[1]], [pGrid[0][pp], pGrid[1][qq]])
                #Deprecated Params[pLabel[0]] = pGrid[0][pp]; Params[pLabel[1]] = pGrid[1][qq]
            
                images[pp][qq] = user_get_Pixelised_Model(Params, **modelFuncArgs)[0]

    ### Pack up into a dictionary
    return dict(useLookup = True, Grid = pGrid, Images = images, width = idP, nP = nPar, interp = None)


def return_Model_Lookup(lookup, P):
    """
    Returns the lookup model image and integer index corresponding to parameter values P.

    Requires:
    -- lookup: Dictionary defined as in get_Model_Lookup
    -- P <single element list/array>: parameter lookup value
    """

    ##
    if(lookup['interp'] is None):
        if(lookup['nP'] == 1):
            index = int(round((P[0]-lookup['Grid'][0])/lookup['width']))
        elif lookup['nP'] == 2:
            index = [int(round((P[0]-lookup['Grid'][0][0])/lookup['width'][0])), int(round((P[1]-lookup['Grid'][1][0])/lookup['width'][1]))]
        else:
            raise RuntimeError('return_Model_Lookup - Lookup can only support up to 2 parameters')
    elif(lookup['interp'].lower() == 'lin' or lookup['interp'].lower() == 'linear'):
        if(lookup['nP'] == 1):
            index = int((P[0]-lookup['Grid'][0])/lookup['width'])
        elif lookup['nP'] == 2:
            index = [int((P[0]-lookup['Grid'][0][0])/lookup['width'][0]), int((P[1]-lookup['Grid'][1][0])/lookup['width'][1])]
        else:
            raise RuntimeError('return_Model_Lookup - Lookup can only support up to 2 parameters')
        
    if((np.array(index) < 0).sum() > 0):
        raise RuntimeError('return_Model_Lookup - Error with returning model lookup index - negative - Check entered range')
    if((np.array(index) > len(lookup['Images'])).sum() > 0):
        raise RuntimeError('return_Model_Lookup - Error with returning model lookup index - Larger than grid - Check entered range')

    if(lookup['interp'] is None):
        if(lookup['nP'] == 1):
            return lookup['Images'][index], index
        else:
            return lookup['Images'][index[0]][index[1]], index
    elif(lookup['interp'].lower() == 'lin' or lookup['interp'].lower() == 'linear'):
        ## Not tested (15th Sept 2015 cajd)
        if(lookup['nP'] == 1):
            grad = (lookup['Images'][index+1]-lookup['Images'][index])/(lookup['Grid'][index+1]-lookup['Grid'][index])
            rImage = grad*(P[0]-lookup['Grid'][index]) + lookup['Images'][index]
            return rImage, index
        else:
            raise ValueError('lookup - 2D interpolation not yet coded...')
            grad = (lookup['Images'][index+1]-lookup['Images'][index])/(lookup['Grid'][index+1]-lookup['Grid'][index])

### ---------------------------------------------------------------- END Model Production - Lookup Table ----------------------------------------------------------------------------------------------------- ###


def user_get_Pixelised_Model(Params, inputImage = None, Verbose = False, noiseType = None, outputImage = False, sbProfileFunc = None, der = None, **sbFuncArgs):
    """
    Native method of image construction using a pixel response function and PSF model, for specified surface brightness profile (10Jul2015)

    Tests:
    First-order analytic bias on e1 = 0.3 agrees well with A Halls version. Comparison of image and up to second order derivatives compare 
    to < 10% with matched A. Hall version.

    Returns a pixelised image set according to Params and using sbProfileFunc. Use of der allows one to specify whether to return derivatives. 
    Uses an enlargementFactor and fineGridFactor to deal with cases where the PS is smaller than the support of the SB profile, and sub-pixel variations.

    Requires:
    --- Params: dictionary specifying model.
    --- inputImage: image input. If none, image is produced according to Params. Allows one to input a noise free image and add noise on the fly without 
        re-evaluating (e.g. in using multiple noise realisations in sims.
    --- Verbose: If true, more is output to screen. Useful for debugging.
    --- noiseType: Specifies noise model. In none, no noise is added to image. Accepted values:
    ___ `Gaussian`
    --- outputImage: IGNORED.
    --- sbProfileFunc: link to function which specifies the SB profile
    --- der: List of parameters specifying the derivative wrt which the image is produced. Length of der sets the order of the derivative, 
        and each element specifies parameter: e.g. [size, e1] gives d^2(SB)/(dTde1). Derivatives are taken around values specified in Params. 
        Only analytic derivative are coded up at this stage.
    -- sbFuncArgs: Dictionary of argements not specifed otherwise which can be passed to the SB profile function.

    Returns:
    image, iParams: Pixelised model image as [nPix, nPix] array, as defined by input, and dictionary of model parameters including any modifications (e.g. update to noise etc).
    """

    import copy
    iParams = copy.deepcopy(Params)

    if(der is not None):
        SBDer, PSFDer = seperate_Keys_byModel(der,refParam =  iParams)
    else:
        SBDer, PSFDer = [],[]

        
    ##Deal with unphysical/invalid parameters by retunring a default value for the model (set to zero)
    if(iParams['SB']['e1']*iParams['SB']['e1'] + iParams['SB']['e2']*iParams['SB']['e2'] >= 1. or iParams['SB']['size'] <= 0):
        return np.zeros(iParams['stamp_size'])

    if(inputImage is None):   
        ###Get Surface Brightness image on enlarged grid. This is to take into account that the surface brightness profile may be non-zero outside the 
        #Postage Stamp boundaries set.
        ## Ideally, enlargement factor should be set to n*sigma along the major axis of the image. 0.7 accounts for the fact that cos(theta) is at maximum 0.7, 
        #and that enlargement should occur equally in x- and y- direction. Larger enlargement factors wil slow down the process, 
        #and this can be turned off by setting enlargementFactor = 1.
        if(iParams['PSF']['PSF_Type']):
            if(iParams['PSF']['PSF_size'] <= 0.):
                raise ValueError('user_get_Pixelised_Model - PSF Size is invalid (Zero or negative)')
            enlargementFactor = int(5*np.amax([iParams['SB']['size'],iParams['PSF']['PSF_size']])/(np.amin(iParams['stamp_size'])*0.7)+1)
        else:
            enlargementFactor = int(5*iParams['SB']['size']/(np.amin(iParams['stamp_size'])*0.7)+1)
        tempStampSize = enlargementFactor*np.array(iParams['stamp_size'])
        if(Verbose):
            print 'enlargement factor is:', enlargementFactor, tempStampSize
            
            
        ##Evaluate user-defined function on a fine grid to account for sub-Pixel variation
        ## Use only an odd number here. Increasing fineGridFactor imporves accuracy, but limits speed
        fineGridFactor = 5
        xy = [np.arange(1.-int(0.5*(fineGridFactor))/fineGridFactor, 1+tempStampSize[0]+int(0.5*fineGridFactor)/fineGridFactor, 1./fineGridFactor), \
              np.arange(1.-int(0.5*(fineGridFactor))/fineGridFactor, 1+tempStampSize[1]+int(0.5*fineGridFactor)/fineGridFactor, 1./fineGridFactor)]
          
        ##Set the centroid for the image. This instance is a special case, where the centroid is assumed always to be at the centre.
        #cen = [(np.amax(xy[0])+1)/2., (np.amax(xy[1])+1)/2.]
        
        cen = iParams['centroid'].copy()

        ## Adjust centroid so it lies in the same relative region of the enlarged Grid, so that returned image can be produced by isolating central part 
        ## of total image
        ## This could also be done dy readjusting according to distance from centre.

        lOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][0]); rOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][1])
        cen[0] = cen[0] + lOffset
        cen[1] = cen[1] + rOffset

        ## Boundary stores the sub-section of the enlarged PS which contains the input stamp
        boundary = np.array(0.5*(enlargementFactor-1)*np.array(iParams['stamp_size'])).astype(int)
        
        ''' Note: No recovery of final subarray is needed provided that xy is evaluated on the same scale as that of size 
        *i.e using no intervals == (enlargmentFactor*stamp_size), as GALSIM only interpolates on this image '''
        if(sbProfileFunc is None):
            raise RuntimeError('user_get_Pixelised_Model - sbProfileFunc must be passed')


        #print "Calling sb profile function with image params: ", iParams # Commented out as really annoying
        sb = sbProfileFunc(xy, cen, iParams['SB']['size'], iParams['SB']['e1'], iParams['SB']['e2'], iParams['SB']['flux'], der = SBDer, **sbFuncArgs)

        ''' Get the PSF model and convolve (if appropriate) '''
        ## Default PSF parameters: this would eventually be passed in
        ## Future edits to this code would require the PSF model to be passed (or determined by dictionary values)

        if(iParams['PSF']['PSF_Type']):
            import PSF_Models

            if(Verbose):
                print 'Convolving with a PSF'
            
            psfCen = [xy[0][0] + 0.5*(xy[0][-1]-xy[0][0]), xy[1][0] + 0.5*(xy[1][-1]-xy[1][0])]
            if(iParams['PSF']['PSF_Type'] == 1 or str(iParams['PSF']['PSF_Type']).lower() == 'gaussian'):
                ## Use definition of elliptical SB profile, with total_flux == 1. so that integral(PSF) = 1
                #psf = gaussian_SBProfile(xy, psfCen,  iParams['PSF_Parameters'][0],  iParams['PSF_Parameters'][1],  iParams['PSF_Parameters'][2], 1.0)
                from PSF_Models import PSFModel_CXX
                psf = PSFModel_CXX(xy, psfCen, iParams['PSF'], der = PSFDer)
            else:
                raise ValueError('user_get_Pixelised_Model - PSF_Type entered not known:'+str(iParams['PSF_Type']))

            ##Convolve the PSF and SBProfile
            ### Note: Where the PSF model has a well defined fourier transform (as with the Gaussian), this could be sped ip by using the analytic form of the transform
            import scipy.signal
            sb = scipy.signal.fftconvolve(sb, psf, 'same')

        ''' Do the PIXELISATION of the image '''
        
        ## Set up pixel response function
        ##Pixel Response must account for the fineGridFactor, i.e. that in the sotred sb profile, each point is sub-pixel by a factor given by the parameter `fineGridFactor'
        ## Thus: Set pixel response function to encompass f grid points, bounded by single box of zeros on the outside
        PixResponse = np.zeros((fineGridFactor + 2, fineGridFactor + 2))
        PixResponse[1:-1, 1:-1] = 1./(fineGridFactor*fineGridFactor)

        ## Convolve with pixel response function
        import scipy.signal
        Pixelised = scipy.signal.fftconvolve(sb, PixResponse, 'same')
        #Pixelised = scipy.signal.convolve2d(sb, PixResponse, 'same')
        #import astropy.convolution as ast
        #ast.convolve(sb, PixResponse)
    
        ##Isolate the middle value as the central pixel value
        Res = Pixelised[::fineGridFactor, ::fineGridFactor]
        #Res = Pixelised[fineGridFactor/2::fineGridFactor, fineGridFactor/2::fineGridFactor]

        ##Isolate part of postage stamp which corresponds to the 'unenlarged' input array
        Res = Res[boundary[0]:boundary[0]+iParams['stamp_size'][0], \
                  boundary[1]:boundary[1]+iParams['stamp_size'][1]]
        ## Q: Could this be done earlier/quicker?

        #Add a background
        if(iParams['SB']['bg'] is not None):
            Res += iParams['SB']['bg']
            
        '''
        print 'Check'
        print Pixelised[2,:]
        print ' '
        print Res[0,:]
        
        import pylab as pl
        f = pl.figure()
        ax = f.add_subplot(211)
        im = ax.imshow(Pixelised)
        pl.colorbar(im)
        ax = f.add_subplot(212)
        im = ax.imshow(Res)
        pl.colorbar(im)
        print 'SB flux check:', sb.sum()
        pl.show()
        '''
    else:
        Res = np.array(inputImage).copy()

    ### Add noise ####
    if(noiseType is not None):

        if(noiseType.lower() == 'g'):

            ''' #Enable this to confirm that noise is added when appropriate
            print 'Applying Noise to image'
            raw_input('ModPro Check')
            '''
            
            ##Apply Gaussian radnom noise to each pixel using a Guassian model
            ## Get Noise Variance by SNR
            iParams['noise'] = 0.025#SNR_Mapping(Res, SNR = iParams['SNR']) ##This preserves flux
            iParams['noise']
            #print 'User defined noise variance taken to be:', iParams['noise']
            
            ## Apply Noise Variance
            noise = iParams['noise']*np.random.randn(Res.shape[0], Res.shape[1]) #+ mu = 0.
            Res += noise
        else:
            raise ValueError('user_get_Pixelised_Model - noiseType not recognised:', str(noiseType))        

    return Res, iParams


def get_Pixelised_Model(Params, noiseType = None, Verbose = False, outputImage = False, sbProfileFunc = None, **sbFuncArgs):
    import time
    import math
    import os
    """
    DEPRECATED GALSIM VERSION. The user is advise to take care if using this version, as development has not considered this method 
    of production for the majority of the development cycle.
    Routine to return as 2D numpy array the model which will be fitted against the image

    1st incarnation ignores the PSF, but sets up an elliptical Gaussian, defined as a Gaussian with an applied ellipticty defined in Params

    When used as part of the ML estimator method, noise should not be applied

    If sbProfileFunc is passed, then method will attempt to produce an image using the GALSIM interpolate model. sbProfuileFunc must accept as arguments:
    --grid
    --centroid
    --image Size
    --e1
    --e2
    --total flux

    NOTE:
    -- If sbProfileFunc is passed, then in the cases where the galaxy is large/elliptical enough that the profile extends (is non-zero)
       beyond the postage stamp size passed in, then the efective flux assigned using the sbProfileFunc method is different to the GALSIM default. 
       This is because the GALSIM default assigns flux by integrating over the whole model, thus the sum of pixels within the PS will be smaller 
       than the acual flux. In contrast, whilst the sbProfileFunc assigns a total flux in the SB profile function itself, the pixel counts assigned by 
       GALSIM aim to get sum(image) = flux within the PS: thus the latter assumes that the whole SB profile fits within the image.
    -- The use of ``enlargementFactor'' allows the analytic, user-specified SB profile to be evaluated on effectively a larger grid, so that the flux 
       assigned is indeed the total flux, and not the total flux within the postage stamp. The residual to the GALSIM default Gaussian class is verified to 
       <~1% for the circular case, but not for any case with ellipticty (15th June 2015 cajd)

    ISSUES:
    -- Where the sbProfileFunc does not correspond to a traditional surface brightness profile (e.g. when considering derivatives), 
       then there may the the unusual case where the flux defined as the sum over the surface brightness grid is within machin precision of zero. 
       In this case, when trying to define a GALSIM interpolated image object, the assertion "abs(flux - flux_tot) < abs(flux_tot)" will fail and 
       GALSIM will crash. This can be hacked in the following code by adding a constant flux sheet to the SB profile and subtracting this off, 
       however if GALSIM is compiled without assertions then the error will not occur, but nonsense results may be output. This was one of the 
       reasons why an authored `user-defined` pixelised model was written and applied.

    **WARNING**
    It is known that the GALSIM routine defined here produces a Gaussian SB profile which is different than the user-defined models otherwise used. 
    Also, where the SB prfile function is passed in, large differences may also be observed where GALSIM is allowed to draw and pixelate, 
    compared to user (cajd) defined pixelation routines. Care must therefore be taken when comparing these routines.
    
    """
    import galsim

    ##Set up copy of Params to avoid overwriting input. As the copy is output, the original can be overwritten by self-referencing on runtime
    iParams = deepcopy(Params)

    ##additionalConstantFlux is used in certain cases to allow GALSIM to deal with cases where the total flux is close to zero
    #Q: If the flux normalisation is arbirarily shifted, how is the PSF and pixel response function-convolved final image affected.
    additionalConstantFlux = 0
    if(additionalConstantFlux != 0):
        raise ValueError('get_pixelised_model - additionalConstantFlux takes a dis-allowed value at this point of the code. I cannae let you do that captain.')

    ##Set up random deviate, which is used to set noise on image
    if(noiseType is not None):
        seed = int(8241573*time.time()) ##This will need editing to produce a more fully random seed
        rng = galsim.UniformDeviate(seed)

    '''
    Non-Parametric model fitting - user-defined surface profile method. Can be used to define pixleised derivatives of user-defined SB profile
    Alternative method is to specify the model on a fine grid, and set as galsim image, then as GSObject which is an interpolated image. Use this method if evaluating pixelsised version of the derivative of the SB profile
    Function will be evaluated on a finer grid than the final version. This shold be quick since entireley analytic, therefore should not matter how fine the grid is
    '''
    if(sbProfileFunc is not None):
        ## 19th May 2015: Known that this implementation does not work yet. Pixel scale needed for image. Ignored for now.
        #raise RuntimeError('get_Pixelised_Model - sbProfileFunc does not work yet')
        if(Verbose):
            print '\n Constructing GALSIM image using user-defined function \n'
            
        ##Set enlargmentFactor, which sets the size of the grid over which the SB profile to be interpolated is produced. This ensures that if the PS is too small to contain the full SB profile, the flux is set consistently to the total flux of the profile, and not just the flux which falls within the PS. Ideally, enlargement factor should be set to n*sigma along the major axis of the image. 0.7 accounts for the fact that cos(theta) is at maximum 0.7, and that enlargement should occur equally in x- and y- direction. Larger enlargement factors wil slow down the process, and this can be turned off by setting enlargementFactor = 1.
        ## NOTE: it is not recommended to turn off enlargementFactor, since the use of the interpolated image can differ to GALSIM default by a large amount depending on how the renormalisation factor is used when defining the interpolated image
        enlargementFactor = int(5*iParams['size']/(np.amin(iParams['stamp_size'])*0.7)+1)
        tempStampSize = enlargementFactor*np.array(iParams['stamp_size'])
        if(Verbose):
            print 'enlargement factor is:', enlargementFactor, tempStampSize


        ##Evaluate user-defined function on a fine grid
        xy = [np.linspace(1, tempStampSize[0], tempStampSize[0]), \
              np.linspace(1, tempStampSize[1], tempStampSize[1])]

        ##Set the centroid for the image. This instance is a special case, where the centroid is assumed always to be at the centre.
        #cen = [(np.amax(xy[0])+1)/2., (np.amax(xy[1])+1)/2.]
        
        cen = iParams['centroid'].copy()

        ## Adjust centroid so it lies in the same relative region of the enlarged Grid, so that returned image can be produced by isolating central part of total image
        ## This could also be done dy readjusting according to distance from centre.
        lOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][0]); rOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][1])
        cen[0] = cen[0] + lOffset
        cen[1] = cen[1] + rOffset


        ''' Note: No recovery of final subaray is needed provided that xy is evaluated on the same scale as that of size *i.e using no intervals == (enlargmentFactor*stamp_size), as GALSIM only interpolates on this image '''

        sb = sbProfileFunc(xy, cen, iParams['size'], iParams['e1'], iParams['e2'], iParams['flux'], **sbFuncArgs)

        #Use to debug the form of the surface brightness profile
        '''
        import pylab as pl
        f = pl.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(sb, interpolation = 'nearest')
        pl.colorbar(im)
        print 'SB flux check:', sb.sum()
        pl.show()
        '''

        ## Set up as a GALSIM interpolated image. To do this, one must set the flux normalisation value. How this is done is an open question: setting it to sb.sum assumes that the sb is constructed on a pixel scale, and that the full flux of the image is contained in the postage stamp, or it may mean that the flux only reflects that which is contained in the PS and thus is not rescaled to contain the full model flux. Setting it to iParams['flux'] may cause unwanted renormalisation in the image if the PS is too small to contain the entire model.
        ## Note, use of the Guassian model with no ellipticity agrees with GALSIM in both cases to sub-% *if enlargementFactor is used*. If not, then differences (GALSIM_Default - Interpolated)  can be as large as : -20 per pixel, factor of 3 in PS flux using flux =  sb.sum(), and -10:60 per pixel, factor of 1.01 in PS flux using flux = iParams['flux'] (10x10, rs = 6.)

        if(np.absolute(sb.sum()) < 10**(-11)):
            additionalConstantFlux = 0.#(2.*10.**(-11))/np.prod(sb.shape)

            if(Verbose):
                print 'Using additional flux due to machine precision requirements in use of GALSIM'

        print 'Sum of SB profile:', sb.sum()

        #sb += additionalConstantFlux        
        gal = galsim.interpolatedimage.InterpolatedImage(galsim.Image(sb, scale = 1.0), flux = sb.sum())
        #sb -= additionalConstantFlux

    elif(iParams['modelType'].lower() == 'gaussian'):
        
        ##Set up initially circular image
        gal = galsim.Gaussian(flux = iParams['flux'], sigma = iParams['size'])

        ##Shear this profile to get elliptical profile
        gal = gal.shear(e1 = iParams['e1'], e2 = iParams['e2'])
    else:
        raise RuntimeError('get_Pixelised_Model - Invalid Surface Brightness model passed.')

    ### Output to screen model Parameters if Debug == True
    if debug:
        print 'Producing model image using parameters:', iParams.keys(), iParams.values()
        
    ##Create pixel response function as tophat of a given entered (known) scale.
    pix = galsim.Pixel(iParams['pixel_scale'])

    final = galsim.Convolve([gal, pix])

    #Draw image
    try:
        image = final.drawImage(nx = iParams['stamp_size'][0], ny = iParams['stamp_size'][1], scale = iParams['pixel_scale'], method = 'no_pixel')
    except:
        print 'Error drawing image with GALSIM. Model iParams are:'
        for kw in iParams.keys():
            print kw, iParams[kw]
        raise RuntimeError('Failed to run GALSIM')


    if(noiseType is not None):
        if Verbose:
            print 'Adding noise to image: Default',  iParams['noise']
        noise = galsim.GaussianNoise(rng, sigma = iParams['noise'])
        ##Add image noise by setting SNR. If preserve_noise = True, then the SNR is acheived by varying noise_var. SNR defined according to `filter-matched' method of GREAT08: SNR = sqrt(sum{I^2_ij}/noise_var)
        if Verbose:
            print 'Adding noise by SNR value:', iParams['SNR']
        iParams['noise'] = image.addNoiseSNR(noise, snr = iParams['SNR'], preserve_flux = True)
        iParams['noise'] = iParams['noise']**0.5
        if Verbose:
            print  'GALSIM Noise:', iParams['noise']
        
        #image.addNoise(noise) #also image.addNoiseSNR(noise, snr = )...

    ## additionalConstantFlux is subtracted to remove constant sheet of flux applied for certain machine-precision cases
    aimage = image.array# - additionalConstantFlux

    if(sbProfileFunc is not None):
        if(Verbose):
            print 'Model Production: sb sum check:', sb.sum(), aimage.sum()
        #raw_input('Check')
    
    ##If debugging, plot pixelised galaxy image
    if(debug or outputImage):
        ##Write Image out to file
        directory = './debugging_output/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.write(os.path.join(directory, 'GALSIM_Image'))
        print 'Output image to:', os.path.join(directory, 'GALSIM_Image')
        raw_input('<Enter> to continue')
            
    return aimage, iParams


def magnification_Field(inputDict, fittingParams, mag = None):
    """
    This function applies a magnification field to a dictionary

    Requires
    --------

    inputDict: An unlensed dictionary of one galaxy that wants to be lensed. (Dict)
    fittingParams: A tuple containing which galaxy properties should be lensed, i.e. ('size',), ('flux',)
    or ('size','flux',). (Tuple)
    mag: If None then the magnification field applied is taken from the input dictionary. If specific magnificaiton field
    wants to be applied set mag = float. (float)

    Returns
    -------

    galDict: A lensed galaxy dictionary
    """

    galDict = deepcopy(inputDict)
    numbImages = len(galDict)
    mag = np.asscalar(mag)

    ## Selects which part of the dictionary to lens and lenses it 

    if len(fittingParams) ==2: # Lens both size and flux
        for i in range(numbImages):
            for j in range(len(galDict['Realization_'+str(i)])):
                if mag is None:
                    magnification = galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["magnification"]
                elif type(mag) == float or type(mag)==np.float64:
                    magnification = mag
                else:

                    raise TypeError('\'mag\' must be a float or equal to None, type is ' + str(type(mag)))

                galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["flux"] *= magnification
                galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["size"] *= magnification
                

    elif fittingParams[0] == 'size': # lens only size
        for i in range(numbImages):
            for j in range(len(galDict['Realization_'+str(i)])):
                if mag is None:
                    magnification = galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["magnification"]
                elif type(mag) == float or type(mag)==np.float64:
                    magnification = mag
                else:

                    raise TypeError('\'mag\' must be a float or equal to None, type is ' + str(type(mag)))

                galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["size"] *= magnification
                

    elif fittingParams[0] == 'flux': # lens only flux
        for i in range(numbImages):
            for j in range(len(galDict['Realization_'+str(i)])):
                if mag is None:
                    magnification = galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["magnification"]
                elif type(mag) == float or type(mag)==np.float64:
                    magnification = mag
                else:
                    raise TypeError('\'mag\' must be a float or equal to None, type is ' + str(type(mag)))
                galDict['Realization_'+str(i)]['Gal_'+str(j)]['SB']["flux"] *= magnification
               
    else: 
        raise TypeError("The fittingParam variable should be a tuple with either/or /'flux/' or /'size/'")


    return galDict
##---------------------------- Differentiation Methods --------------------------------------------##

def differentiate_Pixelised_Model_Analytic(modelParams, pVal, pLab, n, permute = False):
    import surface_Brightness_Profiles as SBPro
    from generalManipulation import makeIterableList
    """
    Wrapper function to produce an analytic derivatve of the pixelised image, by using the fact that the model production routines can 
    be called defining the surface brightness profile routine, and the arguments that are passed into it.

    Surface Brightness Profiles: An alterantive SB profile implementation can be provided by using sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, 
    however the Weave implementation uses the output of the Sympy routine in C++ through weave (SciPy), is noticably faster, and has been tested 
    to be exact to the default float precision in python. WEAVE replaced by SWIG compiled CXX version

    Requires:
    -- modelParams: Disctionary containing default (fixed) values for all parameters which are not being measured
    -- pVal: List ofparamter values, around which the derivative is taken
    -- pLab: List of strings labelling the measured parameters for which the derivative is taken. Must be the same length as pVal
    -- n: Order to which the derivative is taken. SCALAR IN THIS VERSION
    -- permute: If false, single derivative is output for each order entered. If true, the result is returned in an nParamter^order list covering 
    all permutations of the derivatives, where symmetry is enforced. In this case, the diagonal elements cover the nth derivatve with respect to that parameter. 
    Result is on order of that the parameters are entered. E.g. for parameters a and b entered in that order:
    --- Res = [ ddI/dada, ddI/dadb
                ddI/dbda, ddI/dbdb ]

    Returns:
    Array containing derivatives over entered parameters, and all permutations (if permute == True)
    """

    if( n<1 or n>2 ):
        raise ValueError('differentiate_Pixelised_Model_Analytic - Error - Only derivatives up to second order are supported for now')

    nP = len(pVal)
    nPix = modelParams['stamp_size']
    if permute:
        ##Consider all permutations of entered parameters. Use numpy array
        if n == 1:
            Res = np.zeros((nP, nPix[0], nPix[1]))
            for i in range(nP):
                der = [pLab[i]]
                Res[i,:,:] = get_Pixelised_Model_wrapFunction(pVal, modelParams, pLab,  noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
        
        elif n == 2:
            Res = np.zeros((nP, nP, nPix[0], nPix[1]))
            
            for i in range(nP):
                for j in range(i, nP):
                    der = [pLab[i], pLab[j]]
                    Res[i,j,:,:] = get_Pixelised_Model_wrapFunction(pVal, modelParams, pLab,  noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
                    Res[j,i,:,:] = Res[i,j] #Enforce symmetry

    else:
        ## Consider the derivative to given order for each parameter entered
        Res = np.zeros((nP, nPix[0], nPix[1]))
        for par in range(nP):
            ppLab = makeIterableList(pLab[par])
            if(len(ppLab) == n):
                der = ppLab
            #elif(len(pLab) == 1): ##Disabled for now, asmay allow for bugs to be intorduced. For now, only accept cases where pLab labels the derivatives exactly, or raise exception
            #    der = [pLab]*n
            else:
                raise ValueError('differentiate_Pixelised_Model_Analytic - pLab entered is not acceptable for the order of differentiation entered:'+str(n)+':'+str(pLab))
            Res[par] = get_Pixelised_Model_wrapFunction(pVal, modelParams, pLab,  noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

    return Res


def differentiate_Pixelised_Model_Numerical(modelParams, pVal, pLab, n = [1], order = 3, interval = 0.1, eps = 1.e-3, maxEval = 100):
    from derivatives import finite_difference_derivative
    """
    28/5/15
    Numerically differentiates pixelised model with respect to a given parameter. The model must be produced by a routine which returns a gridded 
    (and/or pixelised) image, and must be accessible using a function of form f(x, *args), where x sets the value of the parameter being differentiated wrt, 
    and args allows this value to be correctly labelled in the input model parameter dictionary. These functions are hard coded in this original version, 
    but may be generalised to a user defined  function in future versions.

    This is useful for the numerical evaluation of ML bias.

    Requires:
    --modelParams: Dictionary of model parameters
    -- pVal: Value of free parameters defining point at which derivative is set
    -- pLab: String labels of model parameters corresponding to pVal
    -- n: Order to which derivative is returned
    --- order: number of function evaluations used to evaluate derivative (named to mimic SciPy definition)
    --- interval: step size used in finite difference method. As defined in finite_difference_derivative()
    --- eps: Tolerance for convergence.  As defined in finite_difference_derivative()
    --- maxEval: Maximum number of derivative evaluations (and step-size intervals) considered in testing for convergence.

    """

    result = finite_difference_derivative(get_Pixelised_Model_wrapFunction, pVal, args = [modelParams, pLab, 1], n = n, order = order, dx = interval, eps = eps, convergenceType = 'sum', maxEval = maxEval)

    return result

##-----------------------------Model SB Profile Definitions (+derivatives)-------------------------##

def gaussian_SBProfile(xy, cen, sigma, e1, e2, Itot):
    from math import pi
    """
    DEPRECATED in favor if SymPy or Weave (C++) defintions. Kept as easier to understand form of profile in comparison to those.
    Returns elliptical Gaussian surface brightness profile on a user-specified 2D Grid (xy), which is a 2-element tuple, with each element a 1D <ndarray>, 
    in order [x,y]

    Requires:
    xy:  2-element tuple which defines grid, with each element a 1D <ndarray>, in order [x,y]
    cen: 2-element tuple which defines centroid, with each element a scalar, in order [x_cen,y_cen]
    sigma: width of guassian
    e1: ellipticity in x direction
    e2: ellipticity in y direction
    Itot: Total integrated flux

    NOTE: If the Postage Stamp is too small (set by xy), then some of the profile will fall outside the PS and in this case integrate(gaussian_SBProfile) != flux.
    
    """

    #delR = np.absolute([xy[0]-cen[0], xy[1]-cen[1]]) #[delX, delY]
    delR = [xy[0]-cen[0], xy[1]-cen[1]] #[delX, delY]

    SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))

    ##Set up deformation matrix
    Q = sigma*sigma*np.array([[1.-e1, e2],[e2,1+e1]])
    detQ = (sigma**4.)*(1. - e1*e1 - e2*e2) 
    QIn = (sigma**2.)*np.array([[1.+e1,-e2],[-e2,1-e1]])/detQ

    ##Can this be edited to remove the loop?
    chi2 = np.zeros((xy[0].shape[0], xy[1].shape[0]))
    for i in range(chi2.shape[0]):
        for j in range(chi2.shape[1]):
            chi2[i,j] = -0.5*(delR[0][i]*QIn[0,0]*delR[0][i] + delR[0][i]*QIn[0,1]*delR[1][j] + \
                              delR[1][j]*QIn[1,0]*delR[0][i] + delR[1][j]*QIn[1,1]*delR[1][j]  )

    norm = Itot/(2.*pi*np.sqrt(detQ))

    SB = norm*np.exp(chi2)
    
    return SB
