'''
Author: cajd
Date: 29 May 2015
Purpose: This module contains the routines to obtain the bias in ML value from theoretical considerations.

'''

debug = False

import numpy as np

def analytic_GaussianLikelihood_Bias(parameter_value, parameter_label, imageParams, order = 1, diffType = 'numeric'):
    import src.model_Production as modPro
    import src.surface_Brightness_Profiles as SBPro
    from derivatives import finite_difference_derivative
    '''
    Returns the theoretically motivated ML estimator bias due to finite data sample. First instance only calculates the linear bias. This is only applicable to the case where the estimate is taken as the ML point of a Gaussian Likelihood function, or minimising chi^2, and where the noise variance is uniform across the stamp.

    This formalism removes the dependence on the derivative of the lnL on the image by using the simplifying statistics of the image and its noise properties: in the Gaussian case, and in the formalism of `return_numerical_ML_Bias`, K = -3J, <image - model> = 0 and <(image-model)^2> = noise_var

    NOTE:
    --As bias depends on the noise properties of the image (here assumed to uncorrelated Gaussian noise), the `noise` parameter of the imageParams dictionary *must* be correct.
    --where diffType == `num` or `numeric`, finite differences are used to calcalate the derivative, and a rough convergence test is used as stated in the documentation for `finite_difference_derivative`

    To Do:
    Edit to include fully analytic derivative of the image

    Requires:
    parameter_value: value of beta on which to calculate the bias (either intrinsic parameter value, or ML measurment itself): *must be singular in this case*
    parameter_label: labels the parameter being varied. Must be of the form of the default model parameter dictionary.
    imageParams: parameters which define the image. Parameters which are not being varied must be set to default values. `noise` must be accurate.
    order: not implemented yet. Defines to what order the bias is returned. Default is first order.

    Side Effects: None
    
    Returns: bias to stated order.
    '''


    pVal = parameter_value; pLab = parameter_label

    iParams = imageParams.copy()

    ##-- Get the derivatives of the pixelised, noise-free model
    #diffIm = modPro.differentiate_Pixelised_Model_Numerical(imageParams, S0, derLabel, n = [1,2,3], interval = [0.001, 0.001])
    if diffType.lower() == 'numeric' or diffType.lower() == 'num':
        ##Get fully numeric derivative. This takes the derivative of the image as a whole: therefore note that this is likely to be more problematic in ensuring that derivative has converged.
        diffIm = finite_difference_derivative(modPro.get_Pixelised_Model_wrapFunction, pVal, args = [iParams, pLab, 1], n = [1,2], dx = [0.001, 0.001], order = 5, eps = 1.e-3, convergenceType = 'sum', maxEval = 100)
    elif diffType.lower() == 'analytic' or diffType.lower() == 'ana':
        ## Get fully analytic derivative
        diffIm = [modPro.get_Pixelised_Model_wrapFunction(pVal, iParams, pLab,  noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = [pLab]), modPro.get_Pixelised_Model_wrapFunction(pVal, iParams,pLab,  noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = [pLab, pLab])]
        #diffIm = modPro.get_Pixelised_Model(iParams, noiseType = None, outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der) ##Add args to set derivatives
        #raise RuntimeError('analytic_GaussianLikelihood_Bias - I have not yet coded up a fully analytic way of obtaining derivatives of the model image')
        
    else:
        raise RuntimeError('analytic_GaussianLikelihood_Bias - Invalid differential type (diffType) entered:'+diffType)

    ## get prefactor : (sigma^2)/(2)
    preFactor = -1.0*(imageParams['noise']*imageParams['noise'])/2.
    # get bias as prefactor*(sum I' * I'')/ (sum I' ^2)^2
    bias = ( (diffIm[0]*diffIm[1]).sum() )/np.power( np.power(diffIm[0],2.).sum(), 2.);
    #print 'Bias by parts:', bias, preFactor, bias*preFactor
    #raw_input('bias Check')
    
    bias *= preFactor

    return bias

def return_numerical_ML_Bias(parameter_value, parameter_label, imageParams, order = 1, maxEval = 1000):
    import model_Production as modPro
    '''
    Returns the theoretically motivated ML estiamtor bias due to finite data sample. First instance only calculates the linear bias. This is most useful for a `brute force` approach to the ML bias correction, as (in the Gaussian case) the result depends on the image: therefore K, J and L must be calculated over many samples; therefore, this is likely to be computationally expensive compared to the analytic method where the statistics of the image are known.

    To Do:
    Edit so that setParams decrsibes the image, and allow user to define the maximum number (or tolerance) of iterations to get mean values.

    Known Problems:
    -- GALSIM noise parameter is varying with run (around the third run)

    Requires:
    parameter_value: value of beta on which to calculate the bias (either intrinsic parameter value, or ML measurment itself)
    parameter_label: labels the parameter being varied. Must be of the form of the default model parameter dictionary.
    image: 2-dimensional ndarray containing the image to be analysed
    setParams: model parameters set to the default value.
    order: not implemented yet. Defines to what order the bias is returned. Default is first order.

    Side Effects: None
    
    Returns: bias to stated order.
    '''
    from image_measurement_ML import get_logLikelihood
    from derivatives import finite_difference_derivative

    ##Redefine input for ease of notation
    pVal = parameter_value; pLab = parameter_label

    ##Get derivative of log-likelihood wrt the parameter
    #If returnType = pix, then derivative is still kept in pixel form
    ##Note: even though for simple, uncorrelated noise, the ML point does not depend on the noise value, for the derivative it does. Therefore, it is likely that the noise value passed in via setParams, as measured on the image, must be accurate.

    nPix = np.prod(imageParams['stamp_size'])

    K = np.zeros(maxEval); J = np.zeros(maxEval); I = np.zeros(maxEval);

    for ev in range(maxEval):
        image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G', Verbose = debug)

        ## Store imageParams in temporary storage to ensure that dictionary in not overwritten
        iParams = imageParams.copy()
        dpixlnL = finite_difference_derivative(get_logLikelihood, pVal, args = [pLab, image, iParams, 'pix'], n = [1,2,3], dx = [0.0001, 0.0001], maxEval = 1000, eps = 1.e-3)

        K[ev] = dpixlnL[2].sum()/nPix
        
        J[ev] = (dpixlnL[0]*dpixlnL[1]).sum()/nPix
        
        I[ev] = -(dpixlnL[1].sum())/nPix

    K = K.mean()
    J = J.mean()
    I = I.mean()

    print 'Bias components found:'
    print 'K:', K
    print 'J:', J
    print 'I:', I

    bias = (K+(2.*J))/(2.*nPix*I*I)
    
    return bias


