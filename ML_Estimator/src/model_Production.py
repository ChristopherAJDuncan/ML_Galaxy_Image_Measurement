'''
Module that contains the general routines for production of the pixelised (and potnetially noisy) surface brightness models and its derivatives

Author: cajd
Touch Date: 28 May 2015
'''
import numpy as np
import galsim

debug = False

##--------------Model Dictionary Manipulation------------------------##
def default_ModelParameter_Dictionary(**setters):
    '''
    Returns the default parameters used in setting up the model image. Dictionary keyword-value pairs can be passed in to overwrite defaults

    '''

    imgshape = np.array([10, 10])
    dct = dict(size = 3., e1 = 0., e2 = 0., centroid = (np.array(imgshape)+1)/2., flux = 1.e3, magnification = 1., shear = [0., 0.], noise = 1., SNR = 20., stamp_size = imgshape, pixel_scale = 3., modelType = 'gaussian')

    for kw in setters.keys():
        if kw not in dct:
            print 'find_ML_Estimator - Initial Parameter Keyword:', kw, ' not recognised'
        else:
            dct[kw] = setters[kw]

    return dct

def unpack_Dictionary(dict, requested_keys = None):
    '''
    Returns a list of dictionary values corresponding to the list of requested keys input. If no keys are input, the full list of values corresponding to the full dictionary keys list (in stored order) is returned
    '''


    if(requested_keys is None):
        requested_keys = dict.keys()
    elif(~hasattr(requested_keys, "__iter")):
        ##If a single string is passed in (i.e. not a list), make it into a list
        requested_keys = [requested_keys]

    res = [dict[k] for k in requested_keys]
    return res


##-------------------------Model Production-----------------------------------------##
def get_Pixelised_Model_wrapFunction(x, Params, xKey, returnOrder = 1):
    '''
    ##Wrapper function for get Pixelised model, which returns the image according to Params, where parameter with key 'xKey' is set to value 'x'

    Requires:
    --x:
    --Params:
    --xKey:
    --returnOrder:

    Returns:
    model image (always), altered model Params (if returnOrder /= 1)

    Side Effects: None

    To Do: Edit to allow get_Pixelised_Model parameters to be entered as optional argument [will allow extension to general, user-defined function for model production]
    '''

    if(xKey not in Params):
        raise ValueError('get_Pixelised_Model_wrapFunction - Key entered (',xKey, ') is not contained in model parameter library')

    ##Store params value so that original is not overwritten - should this be the case?
    #iParams = Params.copy() ## DEPRECATED FOR NOW as it makes sense that we would want to overwrite this
    
    Params[xKey] = x

    image, Params = get_Pixelised_Model(Params)

    if returnOrder == 1:
        return image
    else:
        return image, Params

def get_Pixelised_Model(Params, noiseType = None, sbProfileFunc = None, Verbose = True, outputImage = False):
    import time
    import math
    import os
    '''
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
    -- If sbProfileFunc is passed, then in the cases where the galaxy is large/elliptical enough that the profile extends (is non-zero) beyond the postage stamp size passed in, then the efective flux assigned using the sbProfileFunc method is different to the GALSIM default. This is because the GALSIM default assigns flux by integrating over the whole model, thus the sum of pixels within the PS will be smaller than the acual flux. In contrast, whilst the sbProfileFunc assigns a total flux in the SB profile function itself, the pixel counts assigned by GALSIM aim to get sum(image) = flux within the PS: thus the latter assumes that the whole SB profile fits within the image.
    -- The use of ``enlargementFactor'' allows the analytic, user-specified SB profile to be evaluated on effectively a larger grid, so that the flux assigned is indeed the total flux, and not the total flux within the postage stamp. The residual to the GALSIM default Gaussian class is verified to <~1% for the circular case, but not for any case with ellipticty (15th June 2015 cajd)

    
    '''
    import galsim

    ##Set up copy of Params to avoid overwriting input. As the copy is output, the original can be overwritten by self-referencing on runtime
    iParams = Params.copy()

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
        enlargementFactor = int(5*iParams['size']/(np.amin(iParams['stamp_size'])*0.7)+1)
        print 'Enlargement Factor is:', enlargementFactor
        tempStampSize = enlargementFactor*np.array(iParams['stamp_size'])

        ##Evaluate user-defined function on a fine grid
        xy = [np.linspace(1, tempStampSize[0], tempStampSize[0]), \
              np.linspace(1, tempStampSize[1], tempStampSize[1])]

        ##Set the centroid for the image. This instance is a special case, where the centroid is assumed always to be at the centre.
        #cen = [(np.amax(xy[0])+1)/2., (np.amax(xy[1])+1)/2.]
        
        cen = iParams['centroid']

        ## Adjust centroid so it lies in the same relative region of the enlarged Grid, so that returned image can be produced by isolating central part of total image
        ## This could also be done dy readjusting according to distance from centre.
        lOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][0]); rOffset = 0.5*((enlargementFactor-1)*iParams['stamp_size'][1])
        cen[0] = cen[0] + lOffset
        cen[1] = cen[1] + rOffset

        ''' Note: No recovery of final subaray is needed provided that xy is evaluated on the same scale as that of size *i.e using no intervals == (enlargmentFactor*stamp_size), as GALSIM only interpolates on this image '''

        sb = sbProfileFunc(xy, cen, iParams['size'], iParams['e1'], iParams['e2'], iParams['flux'])

        ## Should this be pixel_scale?
        gal = galsim.interpolatedimage.InterpolatedImage(galsim.Image(sb, scale = 1.), flux = iParams['flux'], normalization = 'flux')

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
        if Verbose:
            print  'GALSIM Noise:', iParams['noise']
        
        #image.addNoise(noise) #also image.addNoiseSNR(noise, snr = )...

        
    aimage = image.array
    
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

##---------------------------- Differentiation Methods --------------------------------------------##

def differentiate_Pixelised_Model_Numerical(modelParams, dVal, dLab, n = [1], order = 3, interval = None):
    from derivatives import finite_difference_derivative
    '''
    28/5/15
    Numerically differentiates pixelised model with respect to a given parameter. The model must be produced by a routine which returns a gridded (and/or pixelised) image, and must be accessible using a function of form f(x, *args), where x sets the value of the parameter being differentiated wrt, and args allows this value to be correctly labelled in the input model parameter dictionary. These functions are hard coded in this original version, but may be generalised to a user defined  function in future versions.

    This is useful for the numerical evaluation of ML bias.

    Requires:


    Side Effects:

    '''

    ##Set initial finite difference interval
    if interval is None:
        interval = 0.1

    print 'n at modPro level:', n
    result = finite_difference_derivative(get_Pixelised_Model_wrapFunction, dVal, args = [modelParams, dLab, 1], n = n, order = order, dx = interval)

    '''
    result = np.zeros(len(n))
    for nn in range(len(n)):
        ##Get derivative 
        df = der(get_Pixelised_Model_wrapFunction, dVal, n = n[nn], order = order, dx = interval, args = [modelParams, dLab, 1])

    '''

    return result

##-----------------------------Model SB Profile Definitions (+derivatives)-------------------------##

def gaussian_SBProfile(xy, cen, sigma, e1, e2, Itot):
    from math import pi
    '''
    Returns elliptical Gaussian surface brightness profile on a user-specified 2D Grid (xy), which is a 2-element tuple, with each element a 1D <ndarray>, in order [x,y]

    Requires:
    xy:  2-element tuple which defines grid, with each element a 1D <ndarray>, in order [x,y]
    cen: 2-element tuple which defines centroid, with each element a scalar, in order [x_cen,y_cen]
    sigma: width of guassian
    e1: ellipticity in x direction
    e2: ellipticity in y direction
    Itot: Total integrated flux

    NOTE: If the Postage Stamp is too small (set by xy), then some of the profile will fall outside the PS and in this case integrate(gaussian_SBProfile) != flux.
    
    '''

    delR = np.absolute([xy[0]-cen[0], xy[1]-cen[1]]) #[delX, delY]

    SB = np.zeros((xy[0].shape[0], xy[1].shape[0]))

    ##Set up deformation matrix
    Q = sigma*sigma*np.array([[1.-e1, e2],[e2,1+e1]])
    detQ = sigma*sigma*(1. - e1*e1 - e2*e2) 
    QIn = np.array([[1.+e1,-e2],[-e2,1-e1]])/detQ

    ##Can this be edited to remove the loop?
    chi2 = np.zeros((xy[0].shape[0], xy[1].shape[0]))
    for i in range(chi2.shape[0]):
        for j in range(chi2.shape[1]):
            chi2[i,j] = -0.5*(delR[0][i]*QIn[0,0]*delR[0][i] + delR[0][i]*QIn[0,1]*delR[1][j] + \
                              delR[1][j]*QIn[1,0]*delR[0][i] + delR[1][j]*QIn[1,1]*delR[1][j]  )

    norm = Itot/(2.*pi*detQ)

    SB = norm*np.exp(chi2)
    
    return SB

# These need repeated for every model parameter being measured
def dr_gaussian_SBProfile():
    return 1.0

def ddr_gaussian_SBProfile():
    return 1.0
