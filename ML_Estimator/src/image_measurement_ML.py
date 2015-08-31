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
from copy import deepcopy

verbose = True
vverbose  = False
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


## -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o----- Error Estimation -----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##

def fisher_Error_ML(ML, fitParams, image, setParams, modelLookup):
    '''
    TESTED: Not on any level (31Aug2015)
    
    Calculates the marginalised fisher error on the set of fitParams around maximum-likelihood point ML.

    Note: As the Fisher Matrix assumes that the likelihood is Gaussian around the ML point (in *parameter* space), this estimate is likely to be inaccurate for parameters which are non-linearly related to the observed image value at any point

    Uses the fact that for a Gaussian likelihood (on pixel values, not parameters): ddlnP/(dtheta_i dtheta_j) = 1/sigma^2*sum_pix[delI*model_,ij - model_,i*model_,j]

    Requires:
    ML: Computed ML point, entered as 1D list/tuple/numpy array
    fitParams: list of strings, labelling the parameters to be fit as defined in model dictionary definition
    image: 2D ndarray, containing image postage stamp
    setParams: model disctionary defining all fixed parameters
    modelLookup: modelLookup table as defined in find_ML_Estimator
    '''

    parameters = ML.copy(); pLabels = fitParams.copy()

    ddlnL = differentiate_logLikelihood_Gaussian_Analytic(parameters, pLabels, image, setParams, modelLookup = modelLookup, order = 2, signModifier = 1.)
    ddlnL = -1.*ddlnL ##This is now the Fisher Matrix

    Fin = np.linalg.inv(ddlnL)

    return np.sqrt(np.diag(Fin))
    
##-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o---- ML Estimation   ----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##
def find_ML_Estimator(image, fitParams = None, outputHandle = None, setParams = None, modelLookup = None, searchMethod = 'Powell', preSearchMethod = None, Prior = None, bruteRange = None, biasCorrect = 0, bcoutputHandle = None, **iParams):
    import scipy.optimize as opt
    import model_Production as modPro
    from surface_Brightness_Profiles import gaussian_SBProfile_Weave
    import measure_Bias as mBias
    from generalManipulation import makeIterableList
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

    biasCorrect: int - states what level to correct bias to. Currently accepted value is 0 and 1 [no correction/1st order correction]

    searchMethod: string labelling which method is used to find the minimum chi^2

    preSearchMethod: if not None, then code will run an intial, coarse search over the parameter space to attempt to find the global mimima. By default this is switched off. Where preSearchMethod == grid or brute, the a grid based search is used. Where this is used, a range must either be entered by the user through bruteRange, or it is taken from the entered prior information. NOTE: This still uses a typically coarse grid, therefore if the range is too wide then it is possible that the code may still find a local mimimum if this exists within one grid point interval of the global miminum.

    Side Effects:

    Returns:
    Parameters: tuple of length equal to fitParams. Gives ML estimator for each fit parameter
    '''

    ## Exceptions based on input objects
    if(image is None or sum(image.shape) == 0):
        raise RuntimeError('find_ML_Estimator - image supplied is None or uninitialised')

    ## Set up analysis based on input values
    if(fitParams is None):
        print 'find_ML_Estimator - parameters to be fit (/measured) must be specified - using default:'
        fitParams = ['size', 'e1', 'e2']
        print fitParams
        print ' '

        
    if(len(fitParams) > 2 and modelLookup is not None and modelLookup['useLookup']):
        raise RuntimeError('find_ML_Estimator - Model Lookup is not supported for more than double parameter fits')

    ##Set up initial params, which sets the intial guess or fixed value for the parameters which defines the model
    ##This line sets up the keywords that are accepted by the routine
    ## pixle_Scale and size should be in arsec/pixel and arcsec respectively. If pixel_scale = 1., then size can be interpreted as size in pixels
    ## centroid should be set to the center of the image, here assumed to be the middle pixel

    if(setParams is None):
        initialParams = modPro.default_ModelParameter_Dictionary()
    else:
        initialParams = modPro.default_ModelParameter_Dictionary()
        modPro.update_Dictionary(initialParams, setParams)
        ## Deprecated initialParams.update(setParams)

    modPro.set_modelParameter(initialParams, iParams.keys(), iParams.values())
    ''' Deprecated
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
    '''

    ## Define dictionary ``Params'', which stores values which are being varied when evaluating the likelihood
    
    modelParams = deepcopy(initialParams)

    ####### Search lnL for minimum
    #Construct initial guess for free parameters by removing them from dictionary
    x0 = modPro.unpack_Dictionary(modelParams, requested_keys = fitParams)

    if(preSearchMethod is not None):
        ## Conduct a presearch of the parameter space to set initial guess (usually grid-based or brute-force)
        if(vverbose or debug):
            print '\n Conducting a pre-search of parameter space to idenitfy global minima'
        if(preSearchMethod.lower() == 'grid' or preSearchMethod.lower() == 'brute'):
            ##Brute force method over a range either set as the prior, or the input range.
            if(bruteRange is not None):
                if(vverbose or debug):
                    print '\n Using user-defined parameter range:', bruteRange

                x0 = opt.brute(get_logLikelihood, ranges = bruteRange, args = (fitParams, image, modelParams, modelLookup, 'sum'))

                if(vverbose or debug):
                    print '\n preSearch has found a minimum (on a coarse grid) of:', x0
                
            elif(Prior is not None):
                if(vverbose or debug):
                    print '\n Using prior range'
                raise RuntimeError('find_ML_Estimator - Prior entry has not yet been coded up')

            else:
                raise RuntimeError('find_ML_Estimator - Brute preSearch is active, but prior or range is not set')

    if(debug):
        ##Output Model Dictionary and initial guess information
        print 'Model Dictionary:', modelParams
        print '\n Initial Guess:', x0
        raw_input('Check')

    ##Find minimum chi^2 using scipy optimize routines
    ##version 11+ maxima = opt.minimize(get_logLikelihood, x0, args = (fitParams, image, modelParams))
    if(searchMethod.lower() == 'simplex'):
        maxima = opt.fmin(get_logLikelihood, x0 = x0, xtol = 0.00001, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    elif(searchMethod.lower() == 'brent'):
        maxima = opt.fmin_brent(get_logLikelihood, x0 = x0, xtol = 0.00001, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    elif(searchMethod.lower() == 'powell'):
        maxima = opt.fmin_powell(get_logLikelihood, x0 = x0, xtol = 0.00001, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    elif(searchMethod.lower() == 'cg'):
        ##Not tested (10Aug)
        maxima = opt.fmin_cg(get_logLikelihood, x0 = x0, fprime = differentiate_logLikelihood_Gaussian_Analytic, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug), ftol = 0.000001)
    elif(searchMethod.lower() == 'bfgs'):
        ##Not tested (10Aug)
        maxima = opt.fmin_bfgs(get_logLikelihood, x0 = x0, fprime = differentiate_logLikelihood_Gaussian_Analytic, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    elif(searchMethod.lower() == 'l_bfgs_b'):
        ##Not tested (10Aug)
        maxima = opt.fmin_l_bfgs_b(get_logLikelihood, x0 = x0, fprime = differentiate_logLikelihood_Gaussian_Analytic, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    elif(searchMethod.lower() == 'ncg'):
        ##Not tested (10Aug)
        maxima = opt.fmin_ncg(get_logLikelihood, x0 = x0, fprime = differentiate_logLikelihood_Gaussian_Analytic, args = (fitParams, image, modelParams, modelLookup, 'sum'), disp = (verbose or debug))
    else:
        raise ValueError('find_ML_Estimator - searchMethod entered is not supported:'+str(searchMethod))

    ##Make numpy array (in the case where 1D is used and scalar is returned):
    if(len(fitParams)==1):
        maxima = np.array(makeIterableList(maxima))

    if(vverbose):
        print 'maxima is:', maxima

    if(debug):
        ##Plot and output residual
        fittedParams = deepcopy(modelParams)
        modPro.set_modelParameter(fittedParams, fitParams, maxima)
        ''' Deprecated
        for i in range(len(fitParams)):
            fittedParams[fitParams[i]] =  maxima[i]
        '''
 
        model, disc =  modPro.user_get_Pixelised_Model(fittedParams, sbProfileFunc = gaussian_SBProfile_Weave)
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
        ax.set_title('Image-Model')
        pl.colorbar(im)
        pl.show()

    if(np.isnan(maxima).sum() > 0):
        raise ValueError('get_ML_estimator - FATAL - NaNs found in maxima:', maxima)

    if(verbose):
        print 'Maxima found to be:', maxima

    ##Output Result
    if(outputHandle is not None):
        np.savetxt(outputHandle, np.array(maxima).reshape(1,maxima.shape[0]))

    ## Bias Correct
    if(biasCorrect == 0):
        return maxima
    elif(biasCorrect == 1):
        ana = mBias.analytic_GaussianLikelihood_Bias(maxima, fitParams, modelParams, order = biasCorrect, diffType = 'analytic')
        bc_maxima = maxima-ana

        ##Output Result
        if(bcoutputHandle is not None):
            np.savetxt(bcoutputHandle, np.array(bc_maxima).reshape(1,bc_maxima.shape[0]))

        if(verbose):
            print 'BC Maxima found to be:', bc_maxima

        ##Return minimised parameters
        return maxima, bc_maxima
    else:
        raise ValueError('get_ML_estimator - biasCorrect(ion) value entered is not applicable:'+ str(biasCorrect))



def get_logLikelihood(parameters, pLabels, image, setParams, modelLookup = None, returnType = 'sum'):
    import math, sys
    import model_Production as modPro
    import surface_Brightness_Profiles as SBPro
    import generalManipulation
    '''
    Returns the log-Likelihood for I-Im, where Im is image defined by dictionary ``modelParams'', and I is image being analysed.
    Minimisiation routine should be directed to this function

    Requires:
    parameters: flattened array of parameters to vary (allows for external program to set variation in these params)
    pLabels: tuple of length `parameters`, which is used to identify the parameters being varied. These labels should satisfy the modelParameter dictionary keys using in setting up the model
    image: 2d <ndarray> of pixelised image
    setParams: dictionary of fixed model parameters which sets the model SB profile being fit.
    modelLookup: An instance of the model lookup table, as set in model_Production module
    returnType:
    ---`sum`: Total log-likelihood, summing over all pixels
    ---`pix`: log-likelihood evaluated per pixel. Returns ndarray of the same shape as the input image

    Returns:
    lnL <scalar>: -1*log_likelihood evaulated at entered model parameters

    Known Issue: Splitting input parameters set into model does not work if a particular key of the dictionary corresponds to anything other than a scalar (e.g. tuple ofr array)
    '''

    ##Set up dictionary based on model parameters. Shallow copy so changes do not overwrite the original
    modelParams = deepcopy(setParams)

    if(setParams['stamp_size'] != image.shape):
        raise RuntimeError('get_logLikelihood - stamp size passed does not match image:', str(setParams['stamp_size']), ':', str( image.shape))

    parameters = generalManipulation.makeIterableList(parameters); pLabels = generalManipulation.makeIterableList(pLabels)
    if(len(parameters) != len(pLabels)):
        raise ValueError('get_logLikelihood - parameters and labels entered do not have the same length (iterable test): parameters:', str(parameters), ' labels:', str(pLabels))


    ##Vary parameters which are being varied as input
    modPro.set_modelParameter(modelParams, pLabels, parameters)

    ''' Deprecated for above
    for l in range(len(pLabels)):
        if(pLabels[l] not in modelParams):
            raw_input('Error setting model parameters in get_logLikelihood: Parameter not recognised. <Enter> to continue')
        else:
            modelParams[pLabels[l]] = parameters[l]
    '''

    #Test reasonable model values - Effectively applying a hard prior
    if(math.sqrt(modelParams['SB']['e1']**2. + modelParams['SB']['e2']**2.) >= 0.99):
        ##Set log-probability to be as small as possible
        return sys.float_info.max/10 #factor of 10 to avoid any chance of memory issues here
        #raise ValueError('get_logLikelihood - Invalid Ellipticty values set')
    if(modelParams['SB']['size'] <= 0.):
        return sys.float_info.max/10

    ''' Get Model'''
    if(modelLookup is not None and modelLookup['useLookup']):
        model = np.array(modPro.return_Model_Lookup(modelLookup, parameters)[0]) #First element of this routine is the model image itself
    else:
        model, disc = modPro.user_get_Pixelised_Model(modelParams, sbProfileFunc = SBPro.gaussian_SBProfile_Weave)

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
    if(vverbose):
        print 'Noise in ln-Like evaluation:', modelParams['noise']
    pixlnL =  (np.power(image-model,2.))
    lnL = pixlnL.sum()
    pixlnL *= 0.5/(modelParams['noise']**2.); lnL *= 0.5/(modelParams['noise']**2.)

    if(vverbose):
        print 'lnL:', lnL, [ str(pLabels[i])+':'+str(parameters[i]) for i in range(len(pLabels))]

    ##Model is noise free, so the noise must be seperately measured and passed in
    ## Answer is independent of noise provided invariant across image
    #lnL2 = 0.5*( (np.power(image-model,2.)).sum()/(modelParams['noise']**2.))
    if(returnType.lower() == 'sum'):
        return lnL
    elif(returnType.lower() == 'pix'):
        return pixlnL
    elif(returnType.lower() == 'all'):
        return [lnL, pixlnL]

###---------------- Derivatives of log-Likelihood -----------------------------------------------###

def differentiate_logLikelihood_Gaussian_Analytic(parameters, pLabels, image, setParams, modelLookup = None, returnType = None, order = 1, signModifier = -1.):
    import generalManipulation
    import model_Production as modPro
    ## May need returnType passed in

    '''
    Returns the derivative of the Gaussian log-Likelihood (ignoring parameter-independent prefactor) for parameters labelled by pLabels.
    Uses analytic derivative of the pixelised model as given in differentiate_Pixelised_Model_Analytic routine of model_Production routine.

    Note: `noise` as defined in set params must the noise_std, and must accurately describe the noise properties of the image. 

    Requires:
    parameters: flattened array of parameters to vary (allows for external program to set variation in these params)
    pLabels: tuple of length `parameters`, which is used to identify the parameters being varied. These labels should satisfy the modelParameter dictionary keys using in setting up the model
    image: 2d <ndarray> of pixelised image
    setParams: dictionary of fixed model parameters which sets the model SB profile being fit.
    modelLookup: An instance of the model lookup table, as set in model_Production module
    returnType: Ingored!!!

    Returns:
    [dlnL/dbeta], repeated for all beta in order <1D ndarray>: derivative of -1*log_likelihood evaulated at entered model parameters if order == 1
    [dlnL/dbeta_i dbeta_j], repeated for all beta in order <2D ndarray>: second derivative of -1*log_likelihood evaulated at entered model parameters if order == 1


    '''
    
    ##To be useful as part of a minimisation routine, the arguements passed to this function must be the same as those passed to the ln-Likelihood evalutaion also. This suggest possibly two routines: one, like the model differentiation itself should just return the various derivatives, and a wrapper routine which produces only the relevent derivatives required for mimimisation
    ## Third order is ignored for now, as this wold require an edit to the methdo of calculating model derivatives, and it is unlikely that a third order derivative would ever really be necessary (excpet in the case where an analytic derivative of the model is wanted for the calculation of the bias, where simulations over many images are used: usually, the known statistics of the Gaussian lileihood can be used to remove this necessity anyway).


    ### First derivative only are needed, so for now this will be coded only to deal with first derivatives.
    ### Therefore, n = 1, permute = false by default
    ### Note, that this code is unlikely to speed up any computation provided that the derivative is calculated using SymPY. Therefore this must be addressed.

    ### Set up model parameters as input
    ##Set up dictionary based on model parameters. Shallow copy so changes do not overwrite the original
    modelParams = deepcopy(setParams)

    if(setParams['stamp_size'] != image.shape):
        raise RuntimeError('differentiate_logLikelihood_Gaussian_Analytic - stamp size passed does not match image:', str(setParams['stamp_size']), ':', str( image.shape))

    ##Check whether parameters input are iterable and assign to a tuple if not: this allows both `parameters' and `pLabels' to be passed as e.g. a float and string and the method to still be used as it
    parameters = generalManipulation.makeIterableList(parameters); pLabels = generalManipulation.makeIterableList(pLabels)
    if(len(parameters) != len(pLabels)):
        raise ValueError('get_logLikelihood - parameters and labels entered do not have the same length (iterable test)')

    ##Vary parameters which are being varied as input
    for l in range(len(pLabels)):
        if(pLabels[l] not in modelParams):
            raw_input('Error setting model parameters in get_logLikelihood: Parameter not recognised. <Enter> to continue')
        else:
            modelParams[pLabels[l]] = parameters[l]

    ''' Get Model'''
    if(modelLookup is not None and modelLookup['useLookup']):
        model = np.array(modPro.return_Model_Lookup(modelLookup, parameters)[0]) #First element of this routine is the model image itself
    else:
        model = modPro.user_get_Pixelised_Model(modelParams, sbProfileFunc = modPro.gaussian_SBProfile)[0]


    ''' Get model derivatives '''    
    modDer = modPro.differentiate_Pixelised_Model_Analytic(modelParams, parameters, pLabels, n = 1, permute = False)
    #modDer stores only the n'th derivative of all parameters entered, stored as an nP*nPix*nPix array.

    if(order == 2):
        ##Calculate 2nd derivative also
        modDer2 = modPro.differentiate_Pixelised_Model_Analytic(modelParams, parameters, pLabels, n = 2, permute = True)
            #modDer2 stores the 2nd derivative of all parameters entered, stored as an nP*nP*nPix*nPix array.

    ##Construct the result to be returned. This is a scalar array, with length equal to nP, and where each element corresponds to the gradient in that parameter direction
    nP = len(parameters)
    delI = image - model

    if(order == 1):
        res = np.zeros(nP)
        
        ##Create tdI, which stores dI in the same shape as modDer by adding a first dimension
        tdelI = np.zeros(modDer.shape); tdelI[:] = delI.copy()
        ##Alternatively: tdelI = np.repeat(delI.reshape((1,)+delI.shape), modDer.shape[0], axis = 0)

        ##Set derivative as sum_pix(delI*derI)/sig^2 for all parameters entered
        ## ReturnTypes other than sum could be implemented by removing the sum pats of this relation, however the implementation of fprime in the minimisation routines requires the return to be a 1D array containing the gradient in each direction.
        res = (tdelI*modDer).sum(axis = -1).sum(axis = -1)
        res /= (signModifier/abs(signModifier))*modelParams['noise']*modelParams['noise']
    elif(order == 2):
        ##This could and should be sped-up using two single loops rather than a nested loop, or by defining delI and dIm*dIm in the same dimension as modDer2
        for i in range(nP):
            for j in range(nP):
                res[i,j] = (delI*modDer2[i,j] - modDer[i]*modDer[j]).sum(axis = -1).sum(axis = -1)

        res /= (signModifier/abs(signModifier))*modelParams['noise']*modelParams['noise']



    return res
    
    ## Note: -ve required as code essentially minimises chi^2

##-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o---- Bias Correction ----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##
