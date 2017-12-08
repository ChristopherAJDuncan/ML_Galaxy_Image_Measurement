"""
Author: cajd
Touch Date: 14th May 2015
Purpose: Contains the code to implement a maximum-likelihood based estimator for image model parameters. 
Contains routines to evaluate the log-Likelihood, and it's derivatives, as well as supplementary definitions 
such as mimimisation routines (if applicable), noise estimation from an image, and error estiamtion (e.g. Fisher Matrices).

Update (24th September 2017): Functions have been added to finding log-likelihood for many images with size or magnification as the 
free parameter. Associated optimization routines have been added. Finite differencing methods for 2nd derivatives have been added
to allow for error analysis when caluclating magnification field. (D.Dootson)

"""
from matplotlib.pyplot import *

import numpy as np
import os
from copy import deepcopy

verbose = False
vverbose  = False
debug = False

def estimate_Noise(image, maskCentroid = None):
    """
    Routine which takes in an image and estimates the noise, needed to accurately calculate the expected bias on
     profile measurements. Where a centroid value is passed, the code uses a form of `curve of growth` to estiamte 
     the noise, by increasing the size of a cricular mask steadily by one pixel around that centroid and looking for
      convergence (defined here as the point where the difference between loops is minimised), otherwise the full image is used.

    *** Noise is known to be too large when the postage stamps size is not large enough, so that the model 
    makes up a significant percentage of the image. One may therefore expect the noise to be too large for small PS sizes. ***

    Agrees well with GALSIM noise var on all SNR provided masCentroid is accurately placed on source centre (tested for ellipticity = 0.)

    Requires:
    -- image: Image of source (2-dimensional numpy array)
    -- maskCentroid: center of mask - used to iteritively mask out source to get an accurate estimate 
    of the background noise after removing the source. If None, then the noise is returned as the standard deviation of the image without 
    masking applied. If not None, then the noise is minimum of the difference between successive runs where the mask is increased 
    by one pixel each side of the centre as passed in.

    Returns:
    -- result: Scalar giving the standard deviation of the pixel in the input image, after masking (if appropriate).

    NOTE: UNTESTED for multiple images
    
    """


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
            if(len(tImage.shape) == 3):
                for i in range(tImage.shape[0]):
                    tImage[i][maskCentroid[0]-maskRad:maskCentroid[0]+maskRad, maskCentroid[1]-maskRad:maskCentroid[1]+maskRad] = 0.
            else:
                tImage[maskCentroid[0]-maskRad:maskCentroid[0]+maskRad, maskCentroid[1]-maskRad:maskCentroid[1]+maskRad] = 0.

        res[con-1] = tImage.std()

        if(maskCentroid is None):
            break
        elif(con == res.shape[0]):
            break

    if(maskCentroid is not None):
        return res[np.argmin(np.absolute(np.diff(res)))]
    else:
        return res[0]


## -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o----- Error Estimation -----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##

def fisher_Error_ML(ML, fitParams, image, setParams, modelLookup, useNumericDeriv = False, ml_Eval = None):
    from copy import deepcopy
    """
    Calculates the marginalised fisher error on the set of fitParams (tuple) around maximum-likelihood point ML.
    As the log-Likelihood depends on the image, the images must be supplied, along with a model dictionary giving
    the fixed model parameters (setParams), and the modelLookup (this can be None is no lookup is to be used). 
    The model fit is therefore constructed by setParams+{fitParams:ML}.

    Note: As the Fisher Matrix assumes that the likelihood is Gaussian around the ML point (in *parameter* space), 
    this estimate is likely to be inaccurate for parameters which are non-linearly related to the observed image value at any point.

    Uses the fact that for a Gaussian likelihood (on pixel values, not parameters):
     ddlnP/(dtheta_i dtheta_j) = 1/sigma^2*sum_pix[delI*model_,ij - model_,i*model_,j], where `,i` labels d()/dtheta_i.

    Requires:
    ML: Computed ML point, entered as 1D list/tuple/numpy array
    fitParams: list of strings, labelling the parameters to be fit as defined in model dictionary definition (see default model dictionary definition)
    image: 2D ndarray, containing image postage stamp (image being fit)
    setParams: model dictionary defining all fixed parameters
    modelLookup: modelLookup table as defined in find_ML_Estimator. Can be None if no lookup is used.
    useNumericDeriv: If 'False' then the analytic deriviative is used, if not a finite difference method is used (note so far this only works for magnification measurements and for many realizations).
    ml_Eval: The value of the function evaluated at the ML point, for numerical deriviative only 

    Returns:
    -- err: Tuple containing marginalised Fisher error for all input parameters 
    (in each case all other parameters are considered fixed to ML or input values).

    Tests:
    -- Value of marginalised error is verified to be comparable to the variance over 5x10^5 simulated images for e1, e2 as free parameters without a prior.
    """
    parameters = deepcopy(ML); pLabels = deepcopy(fitParams)
    if (np.shape(image)[1])==1:
        ddlnL = differentiate_logLikelihood_Gaussian_Analytic(parameters, pLabels, image, setParams, modelLookup = modelLookup, order = 2, signModifier = 1.)
        ddlnL = -1.*ddlnL ##This is now the Fisher Matrix
    elif useNumericDeriv == False:
        ddlnL = 0
        for i in range((np.shape(image))[1]):
            image_to_use = np.zeros((np.shape(image))[0])
            for j in range(900): # turns coloumn vector to row vector (could just use .flatten())
                image_to_use[j] = image[j,i]
            ddlnL -= differentiate_logLikelihood_Gaussian_Analytic(parameters, pLabels, image_to_use, setParams['Realization_'+str(i)]['Gal_'+str(0)], modelLookup = modelLookup, order = 2, signModifier = 1.)
    elif useNumericDeriv == True:
        return 1/(finite_Second_Derivative(mag_likelihood, .0001, (ML, image, setParams, fitParams), mlFuncEval = np.asscalar(ml_Eval))) # Check this is the right expression for error with a 1x1 fisher matrix
    else:
        raise TypeError('Error while running fisher_Error_ML')

    Fin = np.linalg.inv(ddlnL) # Inverts Fisher matrix which is the covarience matrix

    return np.sqrt(np.diag(Fin))
    
##-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o---- ML Estimation ----o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-##
def find_ML_Estimator(image,fitParams,outputHandle = None, setParams = None, galaxy_centriod = None, second_gal_param = None, secondFitParams = None,modelLookup = None, searchMethod = 'simplex', preSearchMethod = None, Prior = None, bruteRange = None, biasCorrect = 0, calcNoise = None, bcoutputHandle = None, error = 'Fisher', **iParams):
    import scipy.optimize as opt
    import model_Production as modPro
    from surface_Brightness_Profiles import gaussian_SBProfile_CXX
    import measure_Bias as mBias
    from generalManipulation import makeIterableList
    """
    MAIN ROUTINE FOR THIS MODULE. Takes in an image (at minimum) and a set of values which defines the model 
    parameters (fit and those which are free to vary), and returns the parameter values at which the log-Likelihood 
    is minimised (or Likelihood is maximised). Can correct for first order noise bias (if biasCorrect != 0), and an estimate of the error 
    (if error is equal to a set of pre-defined values [see below]).
    
    Requires:
    -- image: 2d array of pixelised image

    -- fitParams: tuple of strings which define the model parameters which are free to vary (those which will be fit). 
       These must satisfy the definition of model parameters as set out in the default model dictionary. If None, 
       then e1, e2 and T are fit (this could be made stricter by removing the default None initialisation, thereby 
       requiring that a set of parameters to be fit is passed in).

    -- outputHandle: handle of the output file. **Result is always appended**. If not passed in, then result is not output. Output is in ASCII form.

    -- setParams: Default model dictionary containing fixed parameters which describes the model being fixed. 
       One part of a two part approach to setting the full model parameter dictionary, along with iParams. If None, then default model dictionary is taken.
    -- galaxy_centriod: Allows fitted galaxy's centriod not to be at the centre of the postage stamp, takes a 1x2 numpy array as coords. If galaxy_centriod = None
       assumes the centriod is at the centre of the postage stamp.
    -- second_gal_param: Dictionary containing the parameters specifying the secondary galaxies' parameters using the standard format. Galaxy keys should be 'Sec_Gal_1', 
       'Sec_Gal_2' etc.
    -- secondFitParams: tuple of strings that define which parameters want te be fitted for secondary galaxies. The must satisfy the standard dictionary definitions. 
       All strings entered will be fitted for all galaxies. NOTE centriod has to be entered as last in the tuple.
    -- fit_secondary_area: Tells code to fit the area of all secondary galaxies, input are 'y', yes, or 'n', no. 
    -- modelLookup: Dictionary containing lookup table for pixelised model images, as defined in model_Production module.
       If None, no lookup is used, and the model is re-evalauted for each change in model parameters.
    -- searchMethod: String detailing which form of minimisation to use. Accepted values are:
    ___ simplex, brent, powell, cg, bfgs, l_bfgs_b, ncg (as defined in SciPy documentation)

    -- preSearchMethod: String detailing initial search over parameter space to find global Minimium, used as an initial 
       guess for refinement with searchMethod. If None, initial guess is set to default passed in by the combination of
       setParams and iParams. If not None, then code will run an initial, coarse search over the parameter space to attempt
       to find the global mimima. By default this is switched off. Where preSearchMethod == grid or brute, the a grid based search 
       is used. Where this is used, a range must either be entered by the user through bruteRange, or it is taken from the entered 
       prior information. NOTE: This still uses a typically coarse grid, therefore if the range is too wide then it is possible 
       that the code may still find a local mimimum if this exists within one grid point interval of the global miminum.

    -- Prior: NOT USED YET. Skeleton to allow for a parameter prior structure to be passed in

    -- bruteRange: [nPar, 2] sized tuple setting the range in which the initial preSearchMethod is evaluated, if this is done 
       using a grid or brute method (both equivalent), where nPar is the number of free model parameters being fit. 
       THIS DOES NOT CONSTITUTE A PRIOR, as the refinement may still find an ML value outside this range, 
       however where the global maximum occurs outside this range the returned ML value may be expected to be biased.

    -- biasCorrect: integer, states what level of noise bias to correct the estimate to. 
       Only 1st order correction (biasCorrect == 1) is supported. If biasCorrect == 0, the uncorrected estimate 
       (and error if applicable) are output. If biasCorrect > 0, the uncorrected, corrected and error (if applicable) are output. 
       When used, it is important that *the entered model parameter dictionary contains an accurate measure of the pixel noise 
       of appropriate signal--to--noise, as the analytic bias scales according to both*. Noise can be estimate using estimate_Noise() before entry.

    -- bcOutputhandle: As outputHandle, except for the bias corrected estimator.

    -- error: String detailing error estiamte to output. Supported values are:
    ___ fisher: Marginalised fisher error for each parameter around the ML point. See docstring for fisher_Error_ML().
    ___ brute: UNSUPPORTED, however an error defined on the parameter likelihood itself can be derived if the preSearchMethod and 
        bruteRange is defined such that the Likelihood has *compact support*. If not, then this would be inaccurate (underestimated). 
        Therefore coding for this is deferred until the application of a prior is developed, as use of a prior ensures compact support by default.

    -- iParams: set of optional arguments which, together with setParams, defines the intial model dictionary. 
       Allows parameter values to be input individually on call, and is particularly useful for setting initial guesses where preSearchMethod == None.
    
    
    Model Parameter entry: Model Parameters can be entered using two methods
    ___ setParams: Full Dictionary of initial guess/fixed value for set of parameters. If None, this is set to default set. 
        May not be complete: if not, then model parameters set to default as given in default_ModelParameter_Dictionary()
    ___iParams: generic input which allows model parameters to be set individually. Keys not set are set to default as given by 
       default_ModelParameter_Dictionary(). Where an iParams key is included in the default dictionary, or setParams, it will be updated 
        to this value (**therefore iParams values have preferrence**). If key not present in default is entered, it is ignored
    ___ The initial choice of model parameters (including intial guesses for the minimisation routine where preSearchMethod == False) 
        is thus set as setParams+{iParams}



    Returns:
    Returned: tuple of length equal to fitParams. Gives ML estimator for each fit parameter, with bias corrected version 
    (if biasCorrect != 0) and error (if applicable) always in that order. If secondFitParams != None the output is given in order of galaxies with the position of the centoids
    given at the end of the out put in pairs for each galaxy,
    ie fitParams = ('size',), secondFitParams = ('size', 'e1',) and # secondary galaxies =2 then the ML estimator arrary is [size Primary, size Secondary 1,
    e1 Secondary 1, size Secondary 2, e1 Secondary 2]. NOTE there is no error on ML values for secondary galaxies
    """

    ''' Set up defaults '''
    
    #print fitParams

    ##Initialise result variables
    import math, sys
    import model_Production as modPro
    import surface_Brightness_Profiles as SBPro
    import generalManipulation

    Returned = []
    err = None

    ## Exceptions based on input objects
    if(image is None or sum(image.shape) == 0):
        raise RuntimeError('find_ML_Estimator - image supplied is None or uninitialised')
        
    if(len(fitParams) > 2 and modelLookup is not None and modelLookup['useLookup']):
        raise RuntimeError('find_ML_Estimator - Model Lookup is not supported for more than double parameter fits')


    ## Finds number of secondary galaxies
    if second_gal_param == None:
        numbGalaxies = 0
    else:
        numbGalaxies = len(second_gal_param)

    if (verbose or debug):
        print "The number of secondary galaxies is ", numbGalaxies

    ## Check to make sure that if secondary areas are being fitted data the dicts exist
        if secondFitParams is not None and second_gal_param is None:
            raise RuntimeError('No secondary galaxies found')


    ##Set up initial params, which sets the intial guess or fixed value for the parameters which defines the model
    ##This line sets up the keywords that are accepted by the routine
    ## pixle_Scale and size should be in arsec/pixel and arcsec respectively. If pixel_scale = 1., then size can be interpreted as size in pixels
    ## centroid should be set to the center of the image, here assumed to be the middle pixel

    if(setParams is None and galaxy_centriod is None):
        #print "Setting parameters to default"
        initialParams = modPro.default_ModelParameter_Dictionary()
    elif setParams is None and galaxy_centriod is not None: # Need different case depending on if setParams or galxy_centriod is None
        #print "Updating initial parameters with set Params"
        initialParams = modPro.default_ModelParameter_Dictionary()
        initialParams['centroid'] = galaxy_centriod
        modPro.update_Dictionary(initialParams, initialParams)
    elif setParams is not None and galaxy_centriod is None:
        #print "Updating initial parameters with set Params"
        initialParams = modPro.default_ModelParameter_Dictionary()
        modPro.update_Dictionary(initialParams, setParams)        
    else:
        #print "Updating initial parameters with set Params"
        initialParams = modPro.default_ModelParameter_Dictionary()
        initialParams['centroid'] = galaxy_centriod
        modPro.update_Dictionary(initialParams, setParams)
        ## Deprecated initialParams.update(setParams)

    #if iParams is not None:
    #    iParams = iParams.values()   
    #    iParams = iParams[0] 
    modPro.set_modelParameter(initialParams, iParams.keys(), iParams.values())
    
    ## Define modelParams
    modelParams = deepcopy(initialParams)
    
    ## Estimate Noise of Image
    if(calcNoise is not None):
        #Assumes each image is flattened and therefore needs to be reshaped.
        if(len(image.shape) == 2):
            if(image.shape[0] < 2):
                #Use only the first image
                tImage = image[0].reshape(modelParams['stamp_size'])
                maskCentroid = modelParams['centroid']
            else:
                #Use an alternate stack of closest to even (assumes that pixel error is roughly symmetric), (the alternative stack should negate any feature and background, the effect on the noise is uncertain). Can only be used on multiple realisations of the same field
                if(image.shape[0]%2 == 0):
                    finalIndex = image.shape[0]
                else:
                    finalIndex = image.shape[0]-1
                    print "Final Index check (should be even): ", finalIndex
                aStackImage = np.zeros(image[0].shape)
                for i in range(finalIndex):
                    aStackImage += image[i]#*np.power(-1, i)

                print "\nEstimating noise from stack-subtracted image"
                aStackImage /= float(finalIndex)
                tImage = (image[0]-aStackImage).reshape(modelParams['stamp_size'])
            
                #Turn off centroid masking (as feature should be removed), subtract stacked from each realisation, and flatten for noise estimation
                maskCentroid = None
                aStackImage = np.tile(aStackImage, (image.shape[0],1))
                tImage = (image-aStackImage).flatten()

                print "--Done"
                
                #-- Note, this could be improved by removing maskCentroid in this case, thus allowing the flattened array to be used (a larger data vector), and thus reducing the noise on the error estimation
                
                ##Plot
                # import pylab as pl
                # f = pl.figure()
                # ax = f.add_subplot(111)
                # im = ax.imshow(tImage)
                # pl.colorbar(im)
                # pl.show()

        elif(len(image.shape)==1):
            tImage = image.reshape(modelParams['stamp_size'])
            maskCentroid = modelParams['centroid']
        else:
            raise ValueError("find_ML_Estimate: calcNoise: image not of expected shape")
        modelParams['noise'] = calcNoise(tImage, maskCentroid)
    
    ####### Search lnL for minimum
    #Construct initial guess for free parameters by removing them from dictionary
    x0 = modPro.unpack_Dictionary(modelParams, requested_keys = fitParams)
    
    if numbGalaxies !=0 and secondFitParams is not None: ## This could probably be cleared up

        if 'centroid' in secondFitParams:
            for i in range(1,numbGalaxies+1):
                for j in range(len(secondFitParams)-1):
                    x0.append(second_gal_param["Sec_Gal_"+str(i)]["SB"][secondFitParams[j]])  
            for i in range(1,numbGalaxies+1):
                x0.append(second_gal_param["Sec_Gal_"+str(i)]['centroid'][0])  
                x0.append(second_gal_param["Sec_Gal_"+str(i)]['centroid'][1]) 

        else: 
            for i in range(1,numbGalaxies+1):
                for j in range(len(secondFitParams)): 
                    print secondFitParams[j]
                    x0.append(second_gal_param["Sec_Gal_"+str(i)]["SB"][secondFitParams[j]])
    
    ###### Sanity check image dimensions compared to model parameters
    imDim = len(image.shape)
    if(imDim > 2):
        raise ValueError("find_ML_Estimator: Image must not have more than two dimensions. Single postage stamp image must be flattened")
    elif(imDim == 1 and image.shape[0] != np.array(modelParams['stamp_size']).prod()):
        raise ValueError("find_ML_Estimator: Flattened image (1D) length does not correspond to model parameter dimensions")
    elif(imDim == 2 and image.shape[1] != np.array(modelParams['stamp_size']).prod()):
        print 'Image shape: ', image.shape, ' Model shape:' , modelParams['stamp_size']
        raise ValueError("find_ML_Estimator: image shape of second dimension is not consistent with expected model parameter dimension. 2D image array must contain multiple images across first dimension, and (flattened) pixels as a data vector in the second dimension: Have you remembered to flatten the image?")


    if(preSearchMethod is not None):
        ## Conduct a presearch of the parameter space to set initial guess (usually grid-based or brute-force)
        if(vverbose or debug):
            print '\n Conducting a pre-search of parameter space to idenitfy global minima'
        if(preSearchMethod.lower() == 'grid' or preSearchMethod.lower() == 'brute'):
            ##Brute force method over a range either set as the prior, or the input range.
            if(bruteRange is not None):
                if(vverbose or debug):
                    print '\n Using user-defined parameter range:', bruteRange

                print "Using bruteRange: ", bruteRange
                #x0, fval, bruteGrid, bruteVal
                bruteOut = opt.brute(get_logLikelihood, ranges = bruteRange, args = (fitParams, image, modelParams, modelLookup, 'sum'), finish = None, full_output = True)
                x0, fval, bruteGrid, bruteVal = bruteOut
                ## x0 has len(nParam); fval is scalar; bruteGrid has len(nParam), nGrid*nParam; bruteVal has nGrid*nParam

                ###Evaluate error based on brute by integration - this would only work if bruteRange cover the full range where the PDF is non-zero

                if(error is not None and error.lower() == 'brute'):
                    raise RuntimeError('find_ML_Estimator - brute labelled as means of evaluating error. This is possbible, but not coded as limitation in use of bruteRange to cover the whole region where the likelihood is non-zero. When a prior is included, this could be taken to be exact, provided one knows the range where the prior has compact support, and the bruteRange reflects this.')
                ## use scipy.integrate.trapz(bruteVal, x = bruteGrid[i], axis = i) with i looping over all parameters (ensure axis set properly...

                ##Testing of error determination
                # tErr = fisher_Error_ML(x0, fitParams, image, modelParams, modelLookup)
                # from scipy.stats import norm
                # rv = norm(loc = x0, scale = tErr)
                # ##Plot this
                # import pylab as pl
                # f = pl.figure()
                # ax = f.add_subplot(111)
                # import math
                # ax.plot(bruteGrid, np.exp(-1.*(bruteVal-np.amin(bruteVal))), bruteGrid, (np.sqrt(2*math.pi)*tErr)*rv.pdf(bruteGrid))
                # pl.show()
                # raw_input("Check")
                
                if(vverbose or debug):
                    print '\n preSearch has found a minimum (on a coarse grid) of:', x0
                
            elif(Prior is not None):
                if(vverbose or debug):
                    print '\n Using prior range'
                raise RuntimeError('find_ML_Estimator - Prior entry has not yet been coded up')

            else:
                raise RuntimeError('find_ML_Estimator - Brute preSearch is active, but prior or range is not set')

    if(debug or vverbose):
        ##Output Model Dictionary and initial guess information
        print 'Model Dictionary:', modelParams
        print '\n Initial Guess:', x0

    ##Find minimum chi^2 using scipy optimize routines
    ##version 11+ maxima = opt.minimize(get_logLikelihood, x0, args = (fitParams, image, modelParams))
    if(searchMethod.lower() == 'simplex'):
        maxima = opt.fmin(get_logLikelihood, x0 = x0, xtol = 0.00001,  args = (fitParams, image, modelParams, numbGalaxies, second_gal_param, secondFitParams, modelLookup,'sum'), disp = (verbose or debug))

    elif(searchMethod.lower() == "emcee"):
        import emcee

        if(verbose):
            print "\n-Running emcee....."

        #Define MCMC parameters. These should be passed in
        nWalkers = 6
        nRun = 1000
        nBurn = 100

        if(not isinstance(x0, np.ndarray)):
            x0 = np.array(x0)
        nDim = x0.shape[0]

        print "x0: ", x0
        
        #Produce a new x0 for each parameter. For now, take as -1.5x0 to 1.5x0. Better to pass this in, or inform from prior range
        p0 = np.zeros((nWalkers,nDim))
        for i in range(x0.shape[0]):
            p0[:,i] = np.random.uniform(-1.5*x0[i], 1.5*x0[i], nWalkers)

        print "P0:", p0

        sampler = emcee.EnsembleSampler(nWalkers, nDim, get_logLikelihood,  args = (fitParams, image, modelParams, modelLookup, 'sum', -1))

        #Burn-in
        if(verbose):
            print "-Running burn-in....."
        pos, prob, state = sampler.run_mcmc(p0, nBurn)
        sampler.reset()
        if(verbose):
            print "--Finished burn-in."
            print " Position is ", pos
            print "with prob: ", prob
        
        #Run
        if(verbose):
            print "-Sampling....."
        sampler.run_mcmc(pos, nRun)
        if(verbose):
            print "--Finished", nRun, " samples."

        #Get output
        chain = sampler.flatchain
        pChain = sampler.flatlnprobability

        maxIndex = np.argmax(pChain, axis = 0)
        maxima = chain[maxIndex,:]
        err = np.std(chain, axis = 0)

        if(debug):
            import pylab as pl
            f = pl.figure()
            for i in range(1,nDim+1):
                ax = f.add_subplot(nDim, 1, i)
                ax.hist(chain[:,i-1], bins = 100)
                ax.set_title("Par: "+ fitParams[i-1])

            pl.show()
                
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

    #print maxima

    ##Make numpy array (in the case where 1D is used and scalar is returned):
    if(len(fitParams)==1):
        maxima = np.array(makeIterableList(maxima))
        
    if(vverbose):
        print 'maxima is:', maxima

    if(debug):
        ##Plot and output residual
        print 'Plotting residual..'
        
        fittedParams = deepcopy(modelParams)
        modPro.set_modelParameter(fittedParams, fitParams, maxima)
        ''' Deprecated
        for i in range(len(fitParams)):
            fittedParams[fitParams[i]] =  maxima[i]
        '''
 
        model, disc =  modPro.user_get_Pixelised_Model(fittedParams, sbProfileFunc = gaussian_SBProfile_CXX)
        residual = image
        if(len(image.shape) == 2):
            residual -= image
        elif(len(image.shape) == 3):
            for i in range(image.shape[0]):
                residual[i] -= image[i]
        else:
            raise ValueError("Error calculating residual: Image has an unknown rank")

        import pylab as pl
        ##Plot image and model
        f = pl.figure()
        ax = f.add_subplot(211)
        ax.set_title('Model')
        im = ax.imshow(model, interpolation = 'nearest')
        pl.colorbar(im)
        ax = f.add_subplot(212)
        ax.set_title('Image')
        if(len(image.shape) == 3):
            im = ax.imshow(image[0], interpolation = 'nearest')
        else:
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
    #if(outputHandle is not None):
    #   np.savetxt(outputHandle, np.array(maxima).reshape(1,maxima.shape[0])) # Commented out
        
    ## Bias Correct
    if(biasCorrect == 0):
        Returned.append(maxima)
    elif(biasCorrect == 1):
        ana = mBias.analytic_GaussianLikelihood_Bias(maxima, fitParams, modelParams, order = biasCorrect, diffType = 'analytic')
        bc_maxima = maxima-ana

        ##Output Result
        if(bcoutputHandle is not None):
            np.savetxt(bcoutputHandle, np.array(bc_maxima).reshape(1,bc_maxima.shape[0]))

        if(verbose):
            print 'BC Maxima found to be:', bc_maxima

        ##Return minimised parameters
        Returned.append(maxima, bc_maxima)
    else:
        raise ValueError('get_ML_estimator - biasCorrect(ion) value entered is not applicable:'+ str(biasCorrect))


    ## Get Error on measurement. Brute error would have been constructed on the original brute force grid evaluation above.
    # print "the maxima is", maxima
    if(error is not None):
        if(err is not None):
            err = err #Do nothing
        elif(error.lower() == 'fisher'):
            err = fisher_Error_ML(maxima[:len(fitParams)], fitParams, image, modelParams, modelLookup) #Use finalised modelParams here?
        else:
            raise ValueError("get_ML_estimator - failed to return error, error requested, but value not found nor acceptable lable used")
        Returned.append(err)

        
    return Returned





def get_logLikelihood(parameters, pLabels, image, setParams, numbGalaxies=0, other_gal_param=None,
                      secondFitParams=None, modelLookup = None, returnType = 'sum', signModifier = 1, callCount = 0): #  numbGalaxies, other_gal_param,
    import math, sys
    import model_Production as modPro
    import surface_Brightness_Profiles as SBPro
    import generalManipulation
    """
    Returns the (-1.)*log-Likelihood as a Gaussian of lnL propto (I-Im)^2/sigma_n, where Im is image defined by dictionary ``modelParams'', 
    and I is image being analysed, and sigma_n the pixel noise.
    Minimisiation routine should be directed to this function.

    Requires:
    parameters: flattened array of parameter values for free parameters (allows for external program to set variation in these params)
    pLabels: string tuple of length `parameters`, which is used to identify the parameters being varied. These labels should 
    satisfy the modelParameter dictionary keys using in setting up the model.
    image: 2d <ndarray> of pixelised image.
    setParams: dictionary of fixed model parameters which sets the model SB profile being fit.
    numbGalaxies: Number of galaxies on postage stamp. If numbGalaxies = None then one galaxy is used
    other_gal_param: Dictionary containing the parameters of other images on the postage stamp 

    modelLookup: An instance of the model lookup table, as set in model_Production module. If None, the the pixelised model image is 
    re-evaluated for each change in parameters.
    returnType (default sum):
    ---`sum`: Total log-likelihood, summing over all pixels
    ---`pix`: log-likelihood evaluated per pixel. Returns ndarray of the same shape as the input image

    Returns:
    lnL <scalar>: -1*log_likelihood evaulated at entered model parameters

    """
    
    callCount += 1

    ##Set up dictionary based on model parameters. Shallow copy so changes do not overwrite the original
    modelParams = deepcopy(setParams)
    secondModelParams = deepcopy(other_gal_param)

    if secondFitParams is not None:
        for i in range(numbGalaxies):
            if 'centroid' in secondFitParams: # So that the centroid inital guesses aren't deleted yet
                for j in range(len(secondFitParams)-1):
                    secondModelParams["Sec_Gal_"+str(i+1)]["SB"][secondFitParams[j]] = parameters[len(pLabels)]
                    parameters = np.delete(parameters,(len(pLabels)))               
            else:
                for j in range(len(secondFitParams)):
                    secondModelParams["Sec_Gal_"+str(i+1)]["SB"][secondFitParams[j]] = parameters[len(pLabels)]
                    parameters = np.delete(parameters,(len(pLabels))) 

    if secondFitParams is not None and ('centroid' in secondFitParams): # Probably can be placed into above if loop, but I am tired
        for i in range(numbGalaxies):
            secondModelParams["Sec_Gal_"+str(i+1)]['centroid'][0] = parameters[len(pLabels)]
            parameters = np.delete(parameters,(len(pLabels))) 
            secondModelParams["Sec_Gal_"+str(i+1)]['centroid'][0] = parameters[len(pLabels)]
            parameters = np.delete(parameters,(len(pLabels))) 


    if((setParams['stamp_size']-np.array(image.shape)).sum() > 0):
        raise RuntimeError('get_logLikelihood - stamp size passed does not match image:', str(setParams['stamp_size']), ':', str( image.shape))
    
    parameters = generalManipulation.makeIterableList(parameters); pLabels = generalManipulation.makeIterableList(pLabels)
    #if(len(parameters) != len(pLabels)):
    #    raise ValueError('get_logLikelihood - parameters and labels entered do not have the same length (iterable test): parameters:', str(parameters), ' labels:', str(pLabels))
    
    
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

    
    model, disc = modPro.user_get_Pixelised_Model(modelParams, noiseType = None, sbProfileFunc = SBPro.gaussian_SBProfile_CXX)

    if numbGalaxies !=0:
        for i in range(numbGalaxies):
            other_gal_model, disc_to_add = modPro.user_get_Pixelised_Model(secondModelParams["Sec_Gal_"+str(i+1)], noiseType = None, sbProfileFunc = SBPro.gaussian_SBProfile_CXX)
            model += other_gal_model


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

    """ DEPRECATED for multiple models
    if(model.shape != image.shape):
        print "\n\n Model shape: ", model.shape, " :: Image Shape:", image.shape 
        raise ValueError('get_logLikelihood - model returned is not of the same shape as the input image.')
    """
    
    #Flatten model
    model = model.flatten()

    #Construct log-Likelihood assuming Gaussian noise. As this will be minimised, remove the -1 preceeding
    if(vverbose):
        print 'Noise in ln-Like evaluation:', modelParams['noise']
        
    keepPix = returnType.lower() == 'pix' or returnType.lower() == 'all'

    pixlnL = np.array([])
    lnL = 0
    absSign = signModifier/abs(signModifier)

    if(len(image.shape) == len(model.shape)+1):

        if(len(image.shape) == 2 and np.prod(image.shape) == model.shape[-1]):
            print "Model Shape:", model.shape
            print "Image Shape:", image.shape
            raise ValueError("get_logLikelihood: Have you remember to flatten the last two axes of the input image array?")
        
        #print "Considering sum over images", pLabels, parameters

        for i in range(image.shape[0]):
            tpixlnL = absSign*np.power(image[i]-model,2.)
            lnL += tpixlnL.sum()
            if(keepPix):
                pixlnL = np.append(pixlnL, tpixlnL)
    
    else:
        tpixlnL = absSign*np.power(image-model,2.)

        lnL += tpixlnL.sum()
        if(keepPix):
            pixlnL = np.append(pixlnL,tpixlnL)

    pixlnL *= 0.5/(modelParams['noise']**2.); lnL *= 0.5/(modelParams['noise']**2.)

    if(vverbose):
        print 'lnL:', lnL, [ str(pLabels[i])+':'+str(parameters[i]) for i in range(len(pLabels))]

    ##Model is noise free, so the noise must be seperately measured and passed in
    ## Answer is independent of noise provided invariant across image
    #lnL2 = 0.5*( (np.power(image-model,2.)).sum()/(modelParams['noise']**2.))
    if(returnType.lower() == 'sum'):
        print parameters, lnL
        return lnL
    elif(returnType.lower() == 'pix'):
        return pixlnL
    elif(returnType.lower() == 'all'):
        return [lnL, pixlnL]


###-------------- Combined data set analytics ---------------------------------------------------###

def combined_logLikelihood(fittingParameter, images_flattened, galDict, fitting = 'y'):
    import math, sys
    import model_Production as modPro
    import surface_Brightness_Profiles as SBPro
    import generalManipulation
    """
    This function takes a set of images with two galaxies and adds the log likelihood for each image with the size of
    the primary galaxy the only varied parameter.

    Requires
    --------

    fittingParameter: The value of area that the likelihood wants to be evaulated for. (float)
    images_flattened: A numpy array of the flattened images, the second index controls which image. (2d Numpy array)
    galDict: The dictionary used to create the images to be analysed. The key is, galDict['Realizations_x']['Gal_y']
    where x>=0 and y =0 or 1. (Dict)
    fitting: Not being used at the moment.

    Returns
    -------

    lnL: The sum of the Log-likelihood for all the images, NOTE it is really the negative so this function can be passed to an
    optimization routine. 
    """

    numbImages = len(galDict)

    # Creates the models to compare to
    for i in range(numbImages): # Saves the fitting parameter in the galaxies' dictionary. 
        galDict['Realization_'+str(i)]['Gal_0']['SB']["size"] = fittingParameter

    ## Creates the model galaxies using the same gaDict with the primary galaxy size set to fittingParameter. 

    models = np.zeros([galDict['Realization_0']['Gal_0']["stamp_size"][0],galDict['Realization_0']['Gal_0']["stamp_size"][1],numbImages]) #

    model_flattened = np.zeros([galDict['Realization_0']['Gal_0']["stamp_size"][0]*galDict['Realization_0']['Gal_0']["stamp_size"][1], numbImages])
    for i in range(numbImages):
        for j in range(len(galDict['Realization_'+str(i)])): 
        
            model_to_add, disc = modPro.user_get_Pixelised_Model(galDict['Realization_'+str(i)]['Gal_'+str(j)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = None)
            models[:,:,i] += model_to_add 



    model_flattened[:,i] = models[:,:,i].flatten()

    ## Finds the sum of the log-likelihood for all the images. 

    lnL = 0

    for i in range(numbImages):
        tpixlnL = np.power(images_flattened[:,i]-model_flattened[:,i],2.)
        lnL += tpixlnL.sum()*0.5/(galDict['Realization_'+str(i)]['Gal_0']['noise']**2.)
    import math as m

    return lnL
    ##Model is noise free, so the noise must be seperately measured and passed in
    ## Answer is independent of noise provided invariant across image
    #lnL2 = 0.5*( (np.power(image-model,2.)).sum()/(modelParams['noise']**2.))

 #fittingParameter, lnL    


def combinded_min(x0,imagesFlattened, galParams):
    import scipy.optimize as opt
    
    """
    This function calls fmin, a simplex optimization routine, to try and maximise the log-likelihood (it really minimses the -ve of
    the log-likelihood). Note that only the size of the primary galaxy can be varied in combined_logLikelihood therefore we can only 
    fit size in this routine. 

    Requires
    -------

    x0: The initial guess of the primary galaxy size (take from galaxy dictionary). (float)
    imagesFlattened: An array of the flattened images to fit. The second index denotes which galaxy. (2D numpy array)
    galParams: Dictionary of the galaxies used to create the images. The key is galDict['Realizations_x']['Gal_y'] 
    where x>=0 and y =0 or 1. (Dict)

    Returns 
    -------

    mlValue: Maximum likelihood estimator of the size. (float)
    mlError: The error on the ML value of size, calculated with a fisher error. (float)
    """ 
    
    mlValue = opt.fmin(combined_logLikelihood, x0 = x0, xtol=0.0001,args = (imagesFlattened,galParams, 'y')) # Uses simplex optmization

    mlError = fisher_Error_ML(mlValue, ('size',), imagesFlattened, galParams, None) #Error is calculated using a fisher error

    return mlValue, mlError



###--------------- Finds log likelihood given a magnification field ----------------------------###

def mag_likelihood(mu, images_flattened, galDict, fittingParams):

    import math, sys
    import model_Production as modPro
    import surface_Brightness_Profiles as SBPro
    import matplotlib.pyplot as plt
    import generalManipulation

    """
    This function returns the (-ve) log-likelihood of all the inputted images when a given magnification field is applied.

    Requires
    --------

    mu: The value of the magnification field to be applied to the dictionary. (float)
    images_flattened: An array of the flattened images to fit. The second index denotes which galaxy. (2D numpy array)
    galDict: The UNLENSED diction used to describe the galaxies. The key is galDict['Realizations_x']['Gal_y'] 
    where x>=0 and y =0 or 1. (Dict)
    fittingParams: A tuple containing which parameters the magnification field should affect, i.e. ('size',), ('flux',) or
    ('size','flux',). (Tuple)

    Returns
    -------

    lnL: The (-ve) sum of all the log-likelihoods for a given magnification field. The -ve is returned so that the 
    function can be maximsed. (float)
    """
    numbImages = len(galDict)

    ## Updates the magnification value and applies the magnification to the size and flux

    lensedDict = modPro.magnification_Field(galDict, fittingParams, mu) 


    ## Creates the models 

    models = np.zeros([galDict['Realization_0']['Gal_0']["stamp_size"][0],galDict['Realization_0']['Gal_0']["stamp_size"][1],numbImages]) #

    model_flattened = np.zeros([galDict['Realization_0']['Gal_0']["stamp_size"][0]*galDict['Realization_0']['Gal_0']["stamp_size"][1], numbImages])
    

    for i in range(numbImages):
        for j in range(len(galDict['Realization_'+str(i)])):
        
            model_to_add, disc = modPro.user_get_Pixelised_Model(lensedDict['Realization_'+str(i)]['Gal_'+str(j)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = None)
            models[:,:,i] += model_to_add  
            
        model_flattened[:,i] = models[:,:,i].flatten()    



    ## Finds log likelihood and returns the value


    lnL = 0

    for i in range(numbImages):
        tpixlnL = np.power(images_flattened[:,i]-model_flattened[:,i],2.)
        lnL += tpixlnL.sum()*0.5/(galDict['Realization_'+str(i)]['Gal_0']['noise']**2.)


    return lnL



def mag_min(x0,imagesFlattened, unlensedParams, fittingParams):

    import scipy.optimize as opt
    """
    This function calls fmin, a simplex optimization routine, to try and maximise the log-likelihood (it really minimses the -ve of
    the log-likelihood) subject to varitation in the magnification field. Note that only the size of the primary galaxy can be varied 
    in combined_logLikelihood therefore we can only fit size in this routine.    

    Routine
    -------

    x0: Initial guess of the magnificaition field. (float)
    imagesFlattened: An array of the flattened images to fit. The second index denotes which galaxy. (2D numpy array)
    unlensedParams: The UNLENSED diction used to describe the galaxies. The key is galDict['Realizations_x']['Gal_y'] 
    where x>=0 and y =0 or 1. (Dict)
    fittingParams: A tuple containing which parameters the magnification field should affect, i.e. ('size',), ('flux',) or
    ('size','flux',). (Tuple)

    Returns
    -------

    mlValue: Maximum likelihood estimator of the magnification field. (float)
    mlError: The error on the ML value of size, calculated with a fisher error. The derivative is evaluated using a finite difference
    method. This requires further function evaluation and so could be made more efficent if previous evaluations from optimization
    function are used to calculate the error. (float)

    """

    mlValue = opt.fmin(mag_likelihood, x0 = x0, xtol = 0.0001,args = (imagesFlattened, unlensedParams, fittingParams), full_output = 1)
    mlError = fisher_Error_ML(mlValue[0], fittingParams, imagesFlattened, unlensedParams, None, useNumericDeriv = True, ml_Eval = mlValue[1] )


    return mlValue[0], mlError


###---------------- Derivatives of log-Likelihood -----------------------------------------------###

def differentiate_logLikelihood_Gaussian_Analytic(parameters, pLabels, image, setParams, modelLookup = None, returnType = None, order = 1, signModifier = -1.):
    import generalManipulation
    import model_Production as modPro
    from surface_Brightness_Profiles import gaussian_SBProfile_CXX
    '''
    Returns the analytic derivative of the Gaussian log-Likelihood (ignoring parameter-independent prefactor whose derivative is zero) 
    for parameters labelled by pLabels.
    Uses analytic derivative of the pixelised model as given in differentiate_Pixelised_Model_Analytic routine of model_Production routine.

    *** Note: `noise` as defined in set params must the noise_std, and must accurately describe the noise properties of the image. ***

    Requires:
    parameters: flattened array of parameter values to vary (allows for external program to set variation in these params)
    pLabels: tuple of length `parameters`, which is used to identify the parameters being varied. These labels should satisfy the modelParameter 
    dictionary keys using in setting up the model
    image: 2d <ndarray> of pixelised image
    setParams: dictionary of fixed model parameters which sets the model SB profile being fit.
    modelLookup: An instance of the model lookup table, as set in model_Production module
    returnType: IGNORED, but included so that this method mimic the call fingerprint of the log-Likelihood evaluation routine if used as part of 
    a pre-fab minimisation routine.
    order: sets the order to which derivatives are taken. If order == 1, the return is a tuple (ndarray) of length len(parameters), which contains
    the first derivatives of all parameters. If order == 2, the return is a two-dimensional ndarray, where each element i,j gives the second derivative
    wrt parameter i and parameter j. Order >= 3 or <= 0 are not supported.
    signModifier: default -1. Result is multiplied by abs(signModifier)/signModifier, to change the sign of the output. This is required as the lnL routine 
    actually returns -lnL = chi^2 where a minimisation routine is used. Thus, where the minimisation uses first derivatives, the signModifier should be postive, 
    whilst for other applications (such as the fisher error) on requires the derivative of lnL, and so sign modifier must be negative. 
    The absolute value of signModifier is unimportant.

    Returns:
    [dlnL/dbeta], repeated for all beta in order <1D ndarray>: derivative of -1*log_likelihood evaulated at entered model parameters if order == 1
    [[dlnL/dbeta_i dbeta_j]], repeated for all beta in order <2D ndarray>: second derivative of -1*log_likelihood evaulated at entered model 
    parameters if order == 1

    Possible Extensions:
    -- In calculating second order derivatives, a nested loop is used. This is likely to be slow, and as this is used in producing fisher errors 
    (and thus done every run-time), then this could be a bottle-neck on the measurement of the ML point where errors are used

    Tests:
    -- Fisher error agrees well with simulated output for error.
    '''

    #raise ValueError("differentiate_logLikelihood_Gaussian_Analytic: This has been disabled as Weave is not behaving. Further modifications require that model, and derivatives are flattened to mimic that requirement that image is also flattened, and an extension to multiple images (this should occur naturally if model and derivatives are repeated to mimic multiple images")
    
    #if(len(image.shape) > 1):
    #    raise ValueError("differentiate_logLikelihood_Gaussian_Analytic: This routine has not been extended to multiple realisations yet")

    ##To be useful as part of a minimisation routine, the arguements passed to this function must be the same as those passed to the ln-Likelihood evalutaion also. This suggest possibly two routines: one, like the model differentiation itself should just return the various derivatives, and a wrapper routine which produces only the relevent derivatives required for mimimisation
    ## Third order is ignored for now, as this wold require an edit to the methdo of calculating model derivatives, and it is unlikely that a third order derivative would ever really be necessary (excpet in the case where an analytic derivative of the model is wanted for the calculation of the bias, where simulations over many images are used: usually, the known statistics of the Gaussian lileihood can be used to remove this necessity anyway).


    ### First derivative only are needed, so for now this will be coded only to deal with first derivatives.
    ### Therefore, n = 1, permute = false by default

    ### Set up model parameters as input
    ##Set up dictionary based on model parameters. Shallow copy so changes do not overwrite the original


    modelParams = deepcopy(setParams)
    
    ##Check whether parameters input are iterable and assign to a tuple if not: this allows both `parameters' and `pLabels' to be passed as e.g. a float and string and the method to still be used as it
    parameters = generalManipulation.makeIterableList(parameters); pLabels = generalManipulation.makeIterableList(pLabels)
    if(len(parameters) != len(pLabels)):
        raise ValueError('get_logLikelihood - parameters and labels entered do not have the same length (iterable test)')

    ##Vary parameters which are being varied as input

    modPro.set_modelParameter(modelParams, pLabels, parameters)

    ''' Get Model'''
    if(modelLookup is not None and modelLookup['useLookup']):
        model = np.array(modPro.return_Model_Lookup(modelLookup, parameters)[0]) #First element of this routine is the model image itself
    else:
        model = modPro.user_get_Pixelised_Model(modelParams, sbProfileFunc = gaussian_SBProfile_CXX)[0]

    ''' Get model derivatives '''    
    modDer = modPro.differentiate_Pixelised_Model_Analytic(modelParams, parameters, pLabels, n = 1, permute = False)
    #modDer stores only the n'th derivative of all parameters entered, stored as an nP*nPix*nPix array.

    ''' Testing flattening
    print "modDer shape:", modDer.shape()

    #Flatten modDer to mimic flattened image
    modDer = [modDer[i].flatten() for i in range(nP)]
    print "modDer shape:", modDer.shape()
    raw_input()
    '''

    modDer2 = None
    if(order == 2):
        ##Calculate 2nd derivative also
        modDer2 = modPro.differentiate_Pixelised_Model_Analytic(modelParams, parameters, pLabels, n = 2, permute = True)
            #modDer2 stores the 2nd derivative of all parameters entered, stored as an nP*nP*nPix*nPix array.

    #Flatten and reshape model and derivative model images to reflect the form of the input image (which can by multi-realisations)
    model = model.flatten()
    modDer = modDer.reshape((modDer.shape[0], -1))
    if(modDer2 is not None):
        modDer2 = modDer2.reshape((modDer2.shape[0],modDer2.shape[1], -1))
        
    if(len(image.shape) == 2):
        ## Repeat each nReal times
        nRepeat = image.shape[0]
        
        model = np.tile(model, (nRepeat, 1))
        
        modDer = np.array([np.tile(modDer[i],(nRepeat,1)) for i in range(modDer.shape[0])])

        #There's most likely a better way to do this (i.e. quicker)
        modDer2 = np.array([ [np.tile(modDer2[i,j],(nRepeat,1)) for j in range(modDer2.shape[1])] for i in range(modDer2.shape[0])])
        
    # print "Shape check:"
    # print "Image:", image.shape
    # print "Model:", model.shape
    # print "Derivative:", modDer.shape
    # if(modDer2 is not None):
    #     print "2nd Derivative: ", modDer2.shape
    # raw_input("Check")
            
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
        
    elif(order == 2):
        res = np.zeros((nP,nP))
        ##This could and should be sped-up using two single loops rather than a nested loop, or by defining delI and dIm*dIm in the same dimension as modDer2
        ## Alternate speed-up is to implement with CXX
        for i in range(nP):
            for j in range(nP):
                res[i,j] = (delI*modDer2[i,j] - modDer[i]*modDer[j]).sum(axis = -1).sum(axis = -1)

    res /= (signModifier/abs(signModifier))*modelParams['noise']*modelParams['noise']
    
    return res
    

##----- Finite difference 2nd derivative ------##

def finite_Second_Derivative(func, stepPlus, args, stepMinus = None, mlFuncEval = None):

    '''
    This function finds the 2nd derivative using finite differencing only in 1D. Error is proportional to stepPlus**2  

    Paramets
    --------

    func: The function for which you want the 2nd derivative
    args: Tuple of arguments passed to evaluate the function (same order as the argmuents of the function)
    stepPlus: The step size in the positive direction
    stepMinus: The step size in the negative direction, if not entered it is tbe same as the positive step size
    mlFuncEval: To be used if the function has already been evluated at xVal

    Returns
    -------

    secondDeriv: The second derivative evalutated at xVal

    Notes
    -----

    To minimse the error stepPlus & stepMinus should be as close to each other as possible otherwise error is proportional to stepPlus.
    '''

    if stepMinus == None:
        stepMinus = stepPlus


    if mlFuncEval == None:
        return 2*((stepMinus*func((args[0]+stepPlus),*args[1:])+stepPlus*func((args[0]-stepMinus),*args[1:])-(stepPlus+stepMinus)*func(*args))/((stepPlus**2)*stepMinus+(stepMinus**2)*stepPlus))
    elif type(mlFuncEval) ==float:
        return 2*((stepMinus*func((args[0]+stepPlus),*args[1:])+stepPlus*func((args[0]-stepMinus),*args[1:])-(stepPlus+stepMinus)*mlFuncEval)/((stepPlus**2)*stepMinus+(stepMinus**2)*stepPlus))
    else:
        raise TypeError('mlFuncEval should be a float not a ' + str(type(mlFuncEval)))




































