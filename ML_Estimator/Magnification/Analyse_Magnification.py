import os
import sys
sys.path.insert(0, os.path.abspath("../"))

import mypylib.utils.io as io
import numpy as np
import copy
import os
import collections

import python.model_Production as modPro
import BuildData

saveAll = True

directory = "./storage/"

noiseDict = {"sky":114.,
             "readnoise":5.4,
             "gain":3.5}

fluxBoost = 1.e0
nGalaxy = 100

# Fisher, fitGauss
_errorType = "fitGauss"

# brent, powell, fmin, brent, golden, all (includes timing)
_optMethod = "brent"

# used to set brent bracketing, and guassian error fit guess
_postWidthGuess = 0.1

def log_gaussian_fn(x, mean, sigma, amp):
    result = -1.*0.5*np.power(x-mean, 2)/(np.power(sigma,2)) + amp
    print "Considering: ", mean, sigma, amp , " with result:", result
    return result

def combine_Images(data,catalogue, save = True):
    """
    Combines the data into blended images, and contructs the truth catalogue in the same fashion
    :param data:
    :param catalogue:
    :return:
    """

    ngalim = catalogue["ngalim"]

    totalImages = len(data.keys())
    nImagesUsed = 0

    print "Total images:", totalImages

    combinedImages = collections.OrderedDict()
    combinedCatalogue = collections.OrderedDict()
    counter = -1

    while nImagesUsed < totalImages-1:
        counter += 1

        nGal = 0
        while nGal < 1:
            nGal = np.random.poisson(ngalim)

        # Break when we run out of sources
        if nImagesUsed + nGal > totalImages-1:
            break

        # Create a new blended image
        id = np.arange(nImagesUsed, nImagesUsed+nGal, dtype=np.int)
        combinedCatalogue[str(counter)] = {}
        combinedCatalogue[str(counter)]["ids"] = id
        combinedCatalogue[str(counter)]["nGal"] = nGal
        image = data[str(nImagesUsed)]
        combinedCatalogue[str(counter)][str(0)] = catalogue[str(nImagesUsed)]
        for i in range(1,nGal):
            image += data[str(nImagesUsed+i)]
            combinedCatalogue[str(counter)][str(i)] = catalogue[str(nImagesUsed+i)]

        combinedImages[str(counter)] = image

        nImagesUsed += nGal

    print "Constructed %d combined images using %d individual images "%(counter, nImagesUsed)
    combinedCatalogue['nImage'] = counter

    if save:
        filename = os.path.join(directory,"Combined_Data.h5")
        io.save_dict_to_hdf5(filename, combinedImages)
        print ".. Output combined images to %s"%(filename)

        filename = os.path.join(directory,"Combined_Catalogue.h5")
        io.save_dict_to_hdf5(filename, combinedCatalogue)
        print ".. Output combined catalogue to %s" % (filename)

    print " "

    return combinedImages, combinedCatalogue

def apply_magnification(mu, catalogue):


    for image in range(catalogue["nImage"]):
        for igalaxy in range(catalogue[str(image)]["nGal"]):
            galaxy = catalogue[str(image)][str(igalaxy)]
            #galaxy["SB"]["size"] = np.sqrt(mu)*galaxy["SB"]["size"]
            galaxy["SB"]["flux"] = mu*galaxy["SB"]["flux"]

    return catalogue

def construct_Blended_Model(catalogue):

    model = {}
    for i,image in enumerate(catalogue):
        for g,galaxy in enumerate(image):
            singleModel, disc = modPro.get_Pixelised_Model(catalogue[str(i)], noiseType=None, Verbose=False,
                                                           outputImage=False, sbProfileFunc=None)

            if i == 0:
                model[str(g)] = singleModel
            else:
                model[str(g)] += singleModel

    return model

class iteratorStore(object):
    """
    Class used to allow extraction of iteration information from function, e.g. log-likelihood in optimisation
    """
    def __init__(self):
        self.iteration = []
        self.func = []

    def add(self,x,y):
        self.iteration.append(x)
        self.func.append(y)

    def PRINT(self):
        print "iteration:", self.iteration
        print "function:", self.func

    def get(self):
        return np.array(self.iteration), np.array(self.func)

def logLikelihood(mu, images, icovs, catalogue, signMod = +1, normalisation = 0., itStore = None, asReduced = False):
    """

    :param mu:
    :param images:
    :param icovs:
    :param catalogue:
    :param signMod: If +1, this is a cost. If -1, this is log-likelihood
    :param asReduced: If true, then return reducedChi2 (useful for fitting)
    :return:
    """

    print "Considering mu:", mu

    if isinstance(mu, (list, tuple, np.ndarray)):
        assert len(mu) == 1, "logLikelihood: Only single values of mu can be considered"
        mu = mu[0]

    lensed = apply_magnification(mu, copy.deepcopy(catalogue))

    sign = signMod/abs(signMod)

    # For each source, construct the model and then use to get log-likelihood
    lnL = 0
    for i in range(lensed['nImage']):
        model = np.zeros_like(images[str(i)])
        for g in range(lensed[str(i)]['nGal']):
            galaxy = lensed[str(i)][str(g)]
            singleModel, disc = modPro.get_Pixelised_Model(galaxy, noiseType=None, Verbose=False,
                                                           outputImage=False, sbProfileFunc=None)
            if g == 0:
                model = singleModel
            else:
                model += singleModel

        # Get lnL contribution
        lnL += (np.power(images[str(i)] - model,2)*icovs[i]).sum()

    reduced = lnL/(np.prod(model.shape)*lensed['nImage'])
    print "\n lnL was: ", lnL #Chi^2 here
    print "Expected is roughly: ", np.prod(model.shape)*lensed['nImage']
    print "Reduced is therefore: ", lnL/(np.prod(model.shape)*lensed['nImage'])
    lnL -= normalisation
    print "Normalised: ", lnL

    if itStore is not None:
        itStore.add(mu, -1.*lnL)

    if asReduced:
        assert normalisation == 0., "logLikelihood: asReduced only valid without normalisation"
        return reduced
    else:
        return sign*lnL

def boostFlux(catalogue, images):


    for galaxy in range(catalogue["nGal"]):
        key = str(galaxy)
        catalogue[key]["SB"]["flux"] = catalogue[key]["SB"]["flux"] * fluxBoost
        images[key] = images[key]*fluxBoost

    return catalogue, images

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise RuntimeError("Please enter magnification factor as first argument")

    magnification = float(sys.argv[1])

    print "Running magnification factor: ", magnification

    # --------------------------------------- Generate Data ---------------------------------------------------------- #

    BuildData.buildData(nGalaxy=nGalaxy, magnification=magnification, directory=directory)

    strMu = str(magnification)

    # Denote the input catalogues
    inputDataFile = os.path.join(directory,"GEMS_sampled_data_lensed_"+strMu+".h5")
    inputCatalogueFile = os.path.join(directory,"GEMS_sampled_catalogue_unlensed_"+strMu+".h5")

    # Read in the data
    data = io.load_dict_from_hdf5(inputDataFile)
    print "... Loaded Data"
    catalogue = io.load_dict_from_hdf5(inputCatalogueFile)
    print "... Loaded Catalogue"

    print "Loaded ", catalogue['nGal'], " galaxy images"

    # Boost the flux for magnification measurement
    if fluxBoost > 1.:
        catalogue, data = boostFlux(copy.deepcopy(catalogue),
                                    copy.deepcopy(data))

    # -------------------- Combine data into identical images, based on Poisson Distribution ------------------------- #
    blendedImages, blendedCatalogue = combine_Images(data, catalogue, saveAll)
    print ".. Produced ", len(blendedImages), " blended images"

    # ----------------------------------  Add Noise to the images ---------------------------------------------------- #

    ## -------- For blended images -------

    # Get icovs, which is inverse covariance of the observed source
    import python.noiseDistributions as noiseMod
    import copy
    icovs = []
    for key, image in blendedImages.iteritems():
        icovs.append( 1./noiseMod.estimate_PN_noise(copy.deepcopy(image), **noiseDict) )

    noisyBlendedImages = collections.OrderedDict()
    for key, val in blendedImages.iteritems():
        noisyBlendedImages[key] = noiseMod.add_PN_Noise(copy.deepcopy(val), **noiseDict)
    print ".. Produced ", len(noisyBlendedImages.keys()), " noisy blended images"

    noiseCheck = collections.OrderedDict()
    for key, val in noisyBlendedImages.iteritems():
        noiseCheck[key] = val - blendedImages[key]

    io.output_images_to_MEF(os.path.join(directory,"Noisy_Blended_Images_CHECK_"+strMu+".fits"), noiseCheck.values())

    if saveAll:
        filename = os.path.join(directory,"Noisy_Blended_Images_"+strMu+".fits")
        io.output_images_to_MEF(filename, noisyBlendedImages.values())
        #io.save_dict_to_hdf5(filename, noisyBlendedImages)
        print ".. Output noisy blended images to %s" % (filename)

    ## ------- For unblended images --------

    # Get icovs, which is inverse covariance of the observed source
    import python.noiseDistributions as noiseMod
    import copy
    indIcovs = []
    for key, image in data.iteritems():
        indIcovs.append( 1./noiseMod.estimate_PN_noise(copy.deepcopy(image), **noiseDict) )

    noisyIndImages = collections.OrderedDict()
    for key, val in data.iteritems():
        noisyIndImages[key] = noiseMod.add_PN_Noise(copy.deepcopy(val), **noiseDict)
    print ".. Produced ", len(noisyIndImages.keys()), " noisy individual images"

    if saveAll:
        filename = os.path.join(directory,"Noisy_Individual_Images_"+strMu+".fits")
        io.output_images_to_MEF(filename, noisyIndImages.values())
        #io.save_dict_to_hdf5(filename, noisyBlendedImages)
        print ".. Output noisy individual images to %s" % (filename)

    # --------------------------------------- Deblend ---------------------------------------------------------------- #


    # ------------------------------- Measure the magnification ------------------------------------------------------ #
    iterStore = iteratorStore()
    print "Measuring magnification: "
    import scipy.optimize as opt

    kwargs = collections.OrderedDict()
    kwargs["images"] = noisyBlendedImages
    kwargs["icovs"] = icovs
    kwargs["catalogue"] = blendedCatalogue
    kwargs["signMod"] = +1
    kwargs["normalisation"] = 0.
    kwargs["itStore"] = iterStore
    kwargs["asReduced"] = True
    args = tuple(kwargs.values())

    # Testing: Check lnLikelihood at truth, and at value different to truth
    print "lnL at truth: ", logLikelihood(magnification, *args)
    print "lnL at 2x truth: ", logLikelihood(2.*magnification, *args)

    import time
    print "Fitting..."
    fitMu = None
    if _optMethod.lower() == "fmin" or _optMethod.lower() == "all":
        tf1 = time.time()
        fitargs = opt.fmin(logLikelihood, x0 = magnification, xtol = 1.e-6, args = args)
        tf2 = time.time()
        fitMu = fitargs
        print "Finished fmin"
    if _optMethod.lower() == "powell" or _optMethod.lower() == "all":
        tp1 = time.time()
        fitargs = opt.fmin_powell(logLikelihood, x0=magnification, xtol=1.e-6, args=args)
        tp2 = time.time()
        print "Finished powell"
    #tb1 = time.time()
    #fitargs = opt.fmin_bfgs(logLikelihood, x0=magnification, args=args)
    #tb2 = time.time()
    #print "Finished bfgs"
    if _optMethod.lower() == "brent" or _optMethod.lower() == "all":
        tbr1 = time.time()
        fitargs = opt.brent(logLikelihood, args=args, tol = 1.e-6,
                                 brack = [magnification-_postWidthGuess, magnification, magnification+_postWidthGuess])
        tbr2 = time.time()
        fitMu = fitargs
        print "Finished brent"
    if _optMethod.lower() == "golden" or _optMethod.lower() == "all":
        tg1 = time.time()
        fitargs = opt.golden(logLikelihood, args=args, tol = 1.e-6,
                                 brack = [magnification-_postWidthGuess, magnification, magnification+_postWidthGuess])
        tg2 = time.time()
        fitMu = fitargs
        print "Finished golden"

    if fitMu is None:
        raise ValueError("optMethod (%s) not recognised"%(_optMethod))
    print "... Got Fit", fitMu

    if _optMethod.lower() == "all":
        print "Time Check:"
        print "fmin:", tf2-tf1
        print "powell:", tp2-tp1
        #print "bfgs:", tb2-tb1
        print "brent:", tbr2-tbr1
        print "golden:", tg2 - tg1
        raw_input("Check")

    print "Iteration info"
    print iterStore.get()
    #raw_input("Check")


    # Edit normalisation to reflect the fact the we now know the ML point
    kwargs["normalisation"] = fitMu
    args = tuple(kwargs.values())
    # Append this onto lnLikelihood arguements to minimise numerical error in the following
    #listargs = list(args); listargs.append(fitMu[0]); args = tuple(listargs)

    # Get Fisher Estimate of uncertainty
    if _errorType.lower() == "fisher":
        print "Getting FM...", fitMu
        from mypylib.stats.distributions import fisher_matrix
        # This doesn't look for convergence, which might be a problem where the uncertainty is not known
        FM = -1.*fisher_matrix(logLikelihood, fitMu, args, h = 1.e-5)
        print "...Got FM", FM
        if np.prod(FM.shape) == 1:
            FError = np.sqrt(1./FM[0,0])
        else:
            FError = np.sqrt(np.diag(np.linalg.inv(FM)))
            print "FM shape: ", FM.shape
            raise RuntimeError("No multi-param FM implemented yet")
        print "... FM estimated error is: ", FError
        errorEst = FError
    elif _errorType.lower() == "fitgauss":
        x, y = iterStore.get()
        y = np.array(y) # Convert from cost to log-likelihood

        # Renormalise to allow sensible bound on amplitude to be easily set
        y -= np.max(y)

        p0 = [fitMu, _postWidthGuess/100., 0.]
        bounds = [[fitMu-0.1, 1.e-14, -0.1],[fitMu+0.1,1.,0.1]]

        popt, pcurv = opt.curve_fit(log_gaussian_fn, x, y, p0=p0, bounds=bounds)
        print "Guassian fit with: ", popt
        #print "FIT CHECK:", log_gaussian_fn(x, *popt)
        #raw_input("Check")
        errorEst = popt[1]
    else:
        raise ValueError("No Error Estimate taken")

    kwargs["asReduced"] = False
    args = tuple(kwargs.values())

    # -- Plot it using a brute force resampling -- this is really testing only
    x = np.linspace(fitMu[0]-3.*errorEst, fitMu[0]+3.*errorEst, 100)
    lnL = np.empty_like(x)
    for i,xv in enumerate(x):
        lnL[i] = -1.*logLikelihood(xv, *args)

    import pylab as pl
    ax = pl.subplot()
    ax.plot(x, np.exp(lnL-np.max(lnL)))
    ax.errorbar(fitMu, [0.5], xerr = errorEst, marker = "x")
    pl.show()








