'''
Author: cajd
Touch Date: 14th May 2015
Purpose: Contains the code to implement a maximum-likelihood based estimator for image model parameters

To Do:
Output residuals after ML estimate found

Versions:

Surface Brightness Models Implemented:
Gaussian |N|
Sersic |N|

Model Parameters:
-SB Profile:
Radius |N|
Ellipticity (e) |N|
Centroid (x,y) |N|
Total Flux (It) |N|

-Lensing
Convergence/Magnfication |N|
Shear |N| (g)

-PSF |N|
-Background |N|
'''
import galsim
import numpy as np
import os

verbose = False
debug = False

def estimate_Noise(image, maskCentroid = None):
    '''
    Routine which takes in an image and estiamtes the noise on the image, needed to accurately calculate the expected bias on profile measurements

    First iteration only looks for the mean varaince in pixel value, not taking into account image subtraction

    In reality, the noise should be estimated after subtraction of the source, which may also be done by masking out the source centre and taking the std on the background only (assuming constant sky backgroun)

    *** Noise is known to be too large when the postage stamps size is not large enough, so that the model makes up a significant percentage of the image. One may therefore expect the noise to be too large for small PS sizes. ***

    Agrees well with GALSIM noise var on all SNR provided masCentroid is accurately placed on source centre (tested for ellipticity = 0.)

    Requires:
    -- image: Image of source (2-dimensional numpy array)
    -- maskCentroid: center of mask - used to iteritively mask out source to get an accurate estimate of the background noise after removing the source. If None, then the noise is returned as the standard deviation of the image without masking applied. If not None, then the noise is minimum of the difference between successive runs where the mask is increased by one pixel each side of the centre as passed in.
    '''
    if(maskCentroid is not None):
        res = np.zeros(max(maskCentroid[0], maskCentroid[1], abs(image.shape[0]-maskCentroid[0]), abs(image.shape[1]-maskCentroid[1])))
    else:
        res = np.zeros(1)
        
    maskRad = 0; con = 0
    tImage = image.copy()
    while True:
        con += 1

        maskRad = (con-1)*1 #Done in pixels

        if(maskCentroid is not None):
            tImage[maskCentroid[0]-maskRad:maskCentroid[0]+maskRad, maskCentroid[1]-maskRad:maskCentroid[1]+maskRad] = 0.

        res[con-1] = tImage.std()

        if(maskCentroid == None):
            break
        elif(con == res.shape[0]):
            break

    if(maskCentroid is not None):
        return res[np.argmin(np.absolute(np.diff(res)))]
    else:
        return res[0]
    
##-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o---- ML Estimation   ----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##
def find_ML_Estimator(image, fitParams = None, outputHandle = None, setParams = None, modelLookup = None, **iParams):
    import scipy.optimize as opt
    import model_Production as modPro
    '''
    To Do:
    Pass in prior on model parameters

    KNOWN PROBLEMS:
    
    Requires:
    image: 2d array of pixelised image
    fitParams: tuple of strings which satisfy the keywords of iParams:
    ---size {Y}
    ---ellip {Y}
    ---centroid {Y}
    ---flux {Y}

    ---magnification {N}
    ---shear {N}

    GALSIM image declarations (set by image)
    ---stamp_size {Y}
    ---pixel_scale {Y}

    ---modelType: string which defines the surface brightness model which will be fit:
    -----Gaussian [Default] {Y}
    -----Sersic {N}

    outputHandle: file name or file handle of the output. **Result is always appended**. If not passed in, then result is not output

    Model Parameter entry: Model Parameters can be entered using two methods
    setParams: Full Dictionary of initial guess/fixed value for set of parameters. If None, this is set to default set. May not be complete: if not, then model parameters set to default as given in default_ModelParameter_Dictionary()
    iParams: generic input which allows model parameters to be set individually. Keys not set are set to default as given by default_ModelParameter_Dictionary(). Where an iParams key is included in the default dictionary, or setParams, it will be updated to this value (**therefore iParams values have preferrence**). If key not present in default is entered, it is ignored

    Side Effects:

    Returns:
    Parameters: tuple of length equal to fitParams. Gives ML estimator for each fit parameter
    '''

    print '\n'

    ## Exceptions based on input objects
    if(image is None or sum(image.shape) == 0):
        raise RuntimeError('find_ML_Estimator - image supplied is None or uninitialised')

    ## Set up analysis based on input values
    if(fitParams is None):
        print 'find_ML_Estimator - parameters to be fit (/measured) must be specified - using default:'
        fitParams = ['size', 'e1', 'e2']
        print fitParams
        print ' '

        
    if(len(fitParams) > 1 and modelLookup is not None and modelLookup['useLookup']):
        raise RuntimeError('find_ML_Estimator - Model Lookup is not supported for more than single parameter fits')

    ##Set up initial params, which sets the intial guess or fixed value for the parameters which defines the model
    ##This line sets up the keywords that are accepted by the routine
    ## pixle_Scale and size should be in arsec/pixel and arcsec respectively. If pixel_scale = 1., then size can be interpreted as size in pixels
    ## centroid should be set to the center of the image, here assumed to be the middle pixel

    if(setParams is None):
        initialParams = modPro.default_ModelParameter_Dictionary()
    else:
        initialParams = modPro.default_ModelParameter_Dictionary()
        initialParams.update(setParams)

    ## This could be done by initialParams.update(iParams), however theis does not check for unsupported keywords
    failedKeyword = 0
    for kw in iParams.keys():
        if kw not in initialParams:
            print 'find_ML_Estimator - Initial Parameter Keyword:', kw, ' not recognised'
            failedKeyword += 1
        else:
            initialParams[kw] = iParams[kw]
    if(failedKeyword > 0 and verbose):
        ##Remind user of acceptable keywords.
        print '\n Acceptable keywords:'
        print intialParams.keys()
        print ' '

    ## Define dictionary ``Params'', which stores values which are being varied when evaluating the likelihood
    modelParams = initialParams

    ####### Search lnL for minimum
    #Construct initial guess for free parameters by removing them from dictionary
    x0 = modPro.unpack_Dictionary(modelParams, requested_keys = fitParams)

    ##Find minimum chi^2 using scipy optimize routine
    ##version 11+ maxima = opt.minimize(get_logLikelihood, x0, args = (fitParams, image, modelParams))
    maxima = opt.fmin(get_logLikelihood, x0 = x0, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))

    print 'Result:', maxima

    ##Output Result
    if(outputHandle is not None):
        np.savetxt(outputHandle, np.array(maxima).reshape(1,maxima.shape[0]))

    if(debug):
        ##Plot and output residual
        fittedParams = modelParams.copy()
        for i in range(len(fitParams)):
            fittedParams[fitParams[i]] =  maxima[i]
 
        model, disc =  modPro.user_get_Pixelised_Model(fittedParams, sbProfileFunc = modPro.gaussian_SBProfile)
        residual = image-model

        import pylab as pl
        ##Plot image and model
        f = pl.figure()
        ax = f.add_subplot(211)
        ax.set_title('Model')
        im = ax.imshow(model, interpolation = 'nearest')
        pl.colorbar(im)
        ax = f.add_subplot(212)
        ax.set_title('Image')
        im = ax.imshow(image, interpolation = 'nearest')
        pl.colorbar(im)

        pl.show()

        ##Plot Residual
        f = pl.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(residual, interpolation = 'nearest')
        pl.colorbar(im)
        pl.show()

        

    ##Return minimised parameters
    return maxima



def get_logLikelihood(parameters, pLabels, image, setParams, modelLookup = None, returnType = 'sum'):
    import math, sys
    import model_Production as modPro
    '''
    Returns the log-Likelihood for I-Im, where Im is image defined by dictionary ``modelParams'', and I is image being analysed.
    Minimisiation routine should be directed to this function

    Requires:
    parameters: flattened array of parameters to vary (allows for external program to set variation in these params)
    pLabels: tuple of length `parameters`, which is used to identify the parameters being varied. These labels should satisfy the modelParameter dictionary keys using in setting up the model
    image: 2d <ndarray> of pixelised image
    setParams: dictionary of fixed model parameters which sets the model SB profile being fit.
    returnType:
    ---`sum`: Total log-likelihood, summing over all pixels
    ---`pix`: log-likelihood evaluated per pixel. Returns ndarray of the same shape as the input image

    Returns:
    lnL <scalar>: -1*log_likelihood evaulated at entered model parameters

    Known Issue: Splitting input parameters set into model does not work if a particular key of the dictionary corresponds to anything other than a scalar (e.g. tuple ofr array)
    '''

    ##Set up dictionary based on model parameters. Shallow copy so changes do not overwrite the original
    modelParams = setParams.copy()

    if(setParams['stamp_size'] != image.shape):
        raise RuntimeError('get_logLikelihood - stamp size passed does not match image:', str(setParams['stamp_size']), ':', str( image.shape))

    ##Check whether parameters input are iterable and assign to a tuple if not: this allows both `parameters' and `pLabels' to be passed as e.g. a float and string and the method to still be used as it
    if(~(hasattr(parameters, "__iter__") or hasattr(pLabels, "__iter__"))):
       #Both not iterable !!!!EDIT BACK
       parameters = parameters; pLabels = pLabels
       ##parameters = [parameters]; pLabels = [pLabels]
    elif(~hasattr(parameters, "__iter__") or ~hasattr(pLabels, "__iter__")):
       ##Only one not iterable: suggests that one is a form of list whilst the other is not: non-conformal array length
       raise ValueError('get_logLikelihood - parameters and labels entered do not have the same length (iterable test)')
    

    ##Vary parameters which are being varied as input
    for l in range(len(pLabels)):
        if(pLabels[l] not in modelParams):
            raw_input('Error setting model parameters in get_logLikelihood: Parameter not recognised. <Enter> to continue')
        else:
            modelParams[pLabels[l]] = parameters[l]

    #Test reasonable model values - Effectively applying a hard prior
    if(math.sqrt(modelParams['e1']**2. + modelParams['e2']**2.) >= 0.99):
        ##Set log-probability to be as small as possible
        return sys.float_info.max/10 #factor of 10 to avoid any chance of memory issues here
        #raise ValueError('get_logLikelihood - Invalid Ellipticty values set')
    if(modelParams['size'] <= 0.):
        return sys.float_info.max/10

    ''' Get Model'''
    if(modelLookup is not None and modelLookup['useLookup']):
        model = np.array(modelLookup['Images'][modPro.return_Model_Lookup(modelLookup, parameters)])
    else:
        model, disc = modPro.user_get_Pixelised_Model(modelParams, sbProfileFunc = modPro.gaussian_SBProfile)

    ''' Model, lookup comparison '''
    '''
    modelEx, disc = modPro.user_get_Pixelised_Model(modelParams, sbProfileFunc = modPro.gaussian_SBProfile)
    print 'Model, lookup Comparison:', (model-modelEx).sum(), parameters
    import pylab as pl
    f = pl.figure()
    ax = f.add_subplot(211)
    im = ax.imshow(modelEx-model); ax.set_title('model - lookup'); pl.colorbar(im)
    ax = f.add_subplot(212)
    im = ax.imshow(modelEx/model); ax.set_title('model/lookup'); pl.colorbar(im)
    pl.show()
    '''

    if(model.shape != image.shape):
        raise ValueError('get_logLikelihood - model returned is not of the same shape as the input image.')
    
    #Construct log-Likelihood assuming Gaussian noise. As this will be minimised, remove the -1 preceeding
    if(verbose):
        print 'Noise in ln-Like evualtion:', modelParams['noise']
    pixlnL =  (np.power(image-model,2.))
    lnL = pixlnL.sum()
    pixlnL *= 0.5/(modelParams['noise']**2.); lnL *= 0.5/(modelParams['noise']**2.)

    ##Model is noise free, so the noise must be seperately measured and passed in
    ## Answer is independent of noise provided invariant across image
    #lnL2 = 0.5*( (np.power(image-model,2.)).sum()/(modelParams['noise']**2.))
    if(returnType.lower() == 'sum'):
        return lnL
    elif(returnType.lower() == 'pix'):
        return pixlnL
    elif(returnType.lower() == 'all'):
        return [lnL, pixlnL]

##-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o---- Bias Correction ----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##
