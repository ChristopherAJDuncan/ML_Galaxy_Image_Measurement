#!/Users/Ocelot/anaconda/bin/python


"""
TODO:
-- Add sanity check on deblend routine to ensure no mix up with columns
-- Check GEMS size -> galsim size determination
-- Add size cut as Hoekstra
-- Investigate instances where sources are cut, ensure that distance cut is appropriate
"""

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
# If true, take a brute force evaluation of the posterior and compare to estimate found
mlComparisonPlot = False
_verbose = False

directory = "./storage/"


noiseDict = {"sky":114.,
             "readnoise":5.4,
             "gain":3.5}
fluxBoost = 1.e0
nGalaxy = 200

# Degrees of freedom is determined later
rankDOF = None
DOF = None

# Fisher, fitGauss
_errorType = "fitGauss"

# brent, powell, fmin, brent, golden, all (includes timing)
_optMethod = "brent"

# used to set brent bracketing, and gaussian error fit guess
_postWidthGuess = 0.05

_allowMPI = True
_mpiKeepIterating = True
_mpiUpdatedParameters = True
mycomm = None
myrank = None
mycommsize = None
if _allowMPI:
    from mpi4py import MPI

# If true, time profile lnL eval by multiple iterations and averaging. DO NOT FIT in this case
_timeProfile = False

def log_gaussian_fn(x, mean, sigma):
    result = -1.*0.5*np.power(x-mean, 2)/(np.power(sigma,2)) #+ amp
    print "Considering: ", mean, sigma, " with result:", result
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
        if not str(image) in catalogue:
            continue
        for igalaxy in range(catalogue[str(image)]["nGal"]):
            if not str(igalaxy) in catalogue[str(image)]:
                continue
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


def deblend_and_match(directory = "", imageFile = "", matchCatalogue = None):

    import subprocess
    import shutil

    cfgDir = os.path.join(os.path.abspath(os.getcwd()), "storage/SourceExtractorCfg/")
    cfgFiles = ["default.sex", "default.param", "default.conv"]

    CATALOGUEFILE = "SrcEx_Deblended_Catalogue.cat"
    LOGFILE = "SrcEx.LOG"

    # Copy source extractor files to source directory
    for cfgFile in cfgFiles:
        file = os.path.join(cfgDir, cfgFile)
        assert os.path.isfile(file), "deblend_and_match: Config Files (source extractor) not found (%s)"%(file)
        shutil.copy(file, directory)

    # -------- Run source extractor
    CWD = os.getcwd()

    # Move directory to where sources reside
    os.chdir(directory)

    # Run source extractor
    args = ("sex", imageFile, "-CATALOG_NAME "+CATALOGUEFILE)
    subprocess.call(args, stdout=open(LOGFILE, "w"), stderr=subprocess.STDOUT)

    # Move back to working directory
    os.chdir(CWD)

    # --------- Match Source Extractor output to input catalogue

    # Read in catalogue
    SECat = np.genfromtxt(os.path.join(directory, CATALOGUEFILE))

    print "Successfully read in catalogue, with ", len(SECat), "sources"

    # @@ CHECK ME: This should match SrcEx output
    xColDex = 3
    yColDex = xColDex + 1

    extColDex = 1

    distCut = 3. # Pixels

    from mypylib.utils.numutils import pythLineDistance

    deblendedCat = copy.deepcopy(matchCatalogue)

    # Match based on input catalogue, and limit on flux
    from scipy import spatial
    totalGalInput = 0
    totalKept = 0
    for imageDex in range(matchCatalogue["nImage"]):
        # Isolate only those sources in the extension
        extSources = SECat[SECat[:,extColDex] == imageDex + 1,:]

        if len(extSources) == 0:
            print "All sources cut from ext: ", imageDex, " with ", matchCatalogue[str(imageDex)]["nGal"], " expected"
            del deblendedCat[str(imageDex)]
            continue

        keepCount = 0
        for igal in range(matchCatalogue[str(imageDex)]["nGal"]):
            centroid = matchCatalogue[str(imageDex)][str(igal)]["centroid"]
            dist = pythLineDistance(extSources[:, xColDex] - centroid[0],
                                    extSources[:, yColDex] - centroid[1])
            # Get distance for all points

            if (dist <= distCut).sum() == 0:

                if _verbose:
                    print " "
                    print "Cutting ", str(igal), deblendedCat[str(imageDex)][str(igal)]
                    print "Distance: ", dist
                    print "In SE cat: ", extSources
                    print " "

                del deblendedCat[str(imageDex)][str(igal)]
            else:
                keepCount += 1

        totalGalInput += matchCatalogue[str(imageDex)]["nGal"]
        totalKept += keepCount
        if keepCount < matchCatalogue[str(imageDex)]["nGal"]:
            print matchCatalogue[str(imageDex)]["nGal"]-keepCount, " of ", matchCatalogue[str(imageDex)]["nGal"], \
                   "sources cut from ext: ", imageDex + 1

        assert keepCount <= matchCatalogue[str(imageDex)]["nGal"], "deblend: More sources found on deblending," \
                                                              " this cant happen" \
                                                              "in this application"


    print "Finished deblending: Kept ", totalKept, ' of ', totalGalInput, " input galaxies"

    return deblendedCat

def logLikelihood_MPI(mu, images, icovs, catalogue, signMod = +1, normalisation = 0., itStore = None,
                      asReduced = False):

    #if myrank == 0:
    #    global _mpiKeepIterating
    #    _mpiKeepIterating = False

    print " "

    if not _allowMPI:
        return logLikelihood(mu, images, icovs, catalogue, signMod,normalisation,itStore,asReduced)

    lnL = np.zeros(1)
    summedLnL = np.zeros(1)

    global _mpiKeepIterating, _mpiUpdatedParameters
    if myrank == 0:
        # Make sure to reset these for leader rank
        _mpiKeepIterating = True
        _mpiUpdatedParameters = True

    # Update parameters
    while _mpiKeepIterating:

        _mpiKeepIterating = mycomm.bcast(True, root = 0)
        if not _mpiKeepIterating:
            break

        # Broadcast and receive parameters
        _mpiUpdatedParameters = mycomm.bcast(_mpiUpdatedParameters, root = 0)
        #print "Rank ", myrank, "has updated parameters? ", _mpiUpdatedParameters

        if _mpiUpdatedParameters:

            mu = mycomm.bcast(mu, root = 0)
            print "Rank", myrank, " has parameter value: ", mu

            # Reconstruct the log-likelihood
            lnL[0] = logLikelihood(mu, images, icovs, catalogue, signMod, normalisation=0, itStore = None, asReduced=False)

            #print "Rank ", myrank, " got lnL contribution of:", lnL, " for mag:", mu, " with reduced:", lnL/rankDOF

            if myrank == 0:
                _mpiKeepIterating = False

            mycomm.Reduce(lnL, summedLnL, root=0)

            _mpiUpdatedParameters = False

    summedLnL = summedLnL[0]

    if itStore is not None:
        itStore.add(mu, -1.*signMod*summedLnL)

    if asReduced:
        assert DOF is not None, "logLikelihood_MPI: Reduced requested but DOF not supplied"
        summedLnL /= DOF

    return summedLnL




def logLikelihood(mu, images, icovs, catalogue, signMod = +1, normalisation = 0., itStore = None, asReduced = False,
                  imageBound = None):
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

    if imageBound is None:
        imageBound = [0,lensed['nImage']]

    sign = signMod/abs(signMod)

    # For each source, construct the model and then use to get log-likelihood
    lnL = 0
    for i in range(imageBound[0], imageBound[1],1):

        # If an image has been removed form the catalogue (e.g. in deblending)
        if not str(i) in lensed:
            continue

        model = np.zeros_like(images[str(i)])

        modelCount = -1
        for g in range(lensed[str(i)]['nGal']):

            # If a source has been removed (e.g. due to deblending)
            if not str(g) in lensed[str(i)]:
                continue

            modelCount += 1
            galaxy = lensed[str(i)][str(g)]
            singleModel, disc = modPro.get_Pixelised_Model(galaxy, noiseType=None, Verbose=False,
                                                           outputImage=False, sbProfileFunc=None)
            if modelCount == 0:
                model = singleModel
            else:
                model += singleModel

        # Get lnL contribution
        lnL += (np.power(images[str(i)] - model,2)*icovs[i]).sum()

    #print "\n lnL was: ", lnL #Chi^2 here
    #print "Expected is roughly: ", np.prod(model.shape)*lensed['nImage']
    #print "Reduced is therefore: ", lnL/(np.prod(model.shape)*lensed['nImage'])
    lnL -= normalisation
    #print "Normalised: ", lnL

    if itStore is not None:
        itStore.add(mu, -1.*lnL)

    if asReduced:
        assert normalisation == 0., "logLikelihood: asReduced only valid without normalisation"
        if DOF is not None:
            reduced = lnL/DOF
        else:
            reduced = lnL / (np.prod(model.shape) * lensed['nImage'])
        return reduced
    else:
        return sign*lnL

def boostFlux(catalogue, images):


    for galaxy in range(catalogue["nGal"]):
        key = str(galaxy)
        catalogue[key]["SB"]["flux"] = catalogue[key]["SB"]["flux"] * fluxBoost
        images[key] = images[key]*fluxBoost

    return catalogue, images

def SHUTDOWN():

    global _mpiKeepIterating
    if _allowMPI:
        print "Forced Exit:", myrank
        _mpiKeepIterating = mycomm.bcast(False, root = 0)
    print "GOODBYE"
    exit()

if __name__ == "__main__":

    lnLFunc = logLikelihood_MPI

    if len(sys.argv) != 2:
        raise RuntimeError("Please enter magnification factor as first argument")

    magnification = float(sys.argv[1])

    # Alter directory to reflect input
    directory = os.path.join(directory, "mu_"+str(magnification))

    #global mycomm, myrank, mycommsize
    if _allowMPI:
        mycomm = MPI.COMM_WORLD
        myrank = mycomm.Get_rank()
        mycommsize = mycomm.Get_size()
        # If using MPI, then process all the data separately
        directory = os.path.join(directory, "MPIRank"+str(myrank))

        # Ensure total nGalaxy over *all* threads
        nGalaxy = nGalaxy//mycommsize + 1
        print "Rank ", myrank, " is considering ", nGalaxy, " galaxies"

    else:
        myrank = 0

    io.mkdirs(directory)

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

    """
    noiseCheck = collections.OrderedDict()
    for key, val in noisyBlendedImages.iteritems():
        noiseCheck[key] = val - blendedImages[key]

    io.output_images_to_MEF(os.path.join(directory,"Noisy_Blended_Images_CHECK_"+strMu+".fits"), noiseCheck.values())
    """

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
    deblendedCatalogue = deblend_and_match(directory=directory, imageFile="Noisy_Blended_Images_"+strMu+".fits",
                                           matchCatalogue=blendedCatalogue)

    # ------------------------------- Measure the magnification ------------------------------------------------------ #
    if _allowMPI: mycomm.Barrier()

    iterStore = iteratorStore()
    print "Measuring magnification: "
    import scipy.optimize as opt

    kwargs = collections.OrderedDict()
    kwargs["images"] = noisyBlendedImages
    kwargs["icovs"] = icovs
    kwargs["catalogue"] = deblendedCatalogue
    kwargs["signMod"] = +1
    kwargs["normalisation"] = 0.
    if myrank == 0:
        kwargs["itStore"] = iterStore
        kwargs["asReduced"] = True
    else:
        kwargs["itStore"] = None
        kwargs["asReduced"] = False
    args = tuple(kwargs.values())


    ## Determine the degrees of freedom
    rankDOF = len(deblendedCatalogue)*np.prod(noisyBlendedImages["0"].shape)*np.ones(1)
    if _allowMPI:
        DOF = np.zeros(1)
        mycomm.Reduce(rankDOF, DOF, root = 0)
        DOF = DOF[0]
    else:
        DOF = rankDOF[0]

    print 'nData:', rankDOF, " for rank:", myrank, " with nimage:", len(deblendedCatalogue)

    if myrank == 0:
        print "Degrees of freedom is: ", DOF

    # Start looping lnL update for non-leader rank. For all non-leader ranks, this first call sets up a loop
    # which is used to keep updating the log-likelihood for the parameter value set by the leader.
    # NOTE: This might complicate the interpretation when non-global parameters are fit, e.g. centroiding etc
    disc = lnLFunc(magnification, *args)

    if myrank == 0:
        print "** lnL at truth:", disc

    # Everything after this should be rank 0 only
    if myrank != 0:
        print "Exiting rank:", myrank
        exit()

    # Testing: Check lnLikelihood at truth, and at value different to truth
    # Note, these will be different for n > 1, as other threads contribute to lnL in above
    #print "check lnL:", logLikelihood(magnification, *args)
    if _timeProfile:
        print '\n\n\n TIME PROFILING'
        import time
        nIteration = 20
        rand = np.random.normal(1., 0.01, nIteration)
        t1 = time.time()
        for i in range(nIteration): print "lnL at truth: ", lnLFunc(rand[i]*magnification, *args)
        t2 = time.time()
        print "Time taken for lnL:", t2-t1, (t2-t1)/nIteration
        SHUTDOWN()


    print "lnL at 2x truth: ", lnLFunc(2.*magnification, *args)
    # Note, these will be different for n > 1, as other threads contribute to lnL in above
    #print "check lnL:", logLikelihood(2.*magnification, *args)



    import time
    print "Fitting..."
    fitMu = None
    if _optMethod.lower() == "fmin" or _optMethod.lower() == "all":
        tf1 = time.time()
        fitargs = opt.fmin(lnLFunc, x0 = magnification, xtol = 1.e-6, args = args)
        tf2 = time.time()
        fitMu = fitargs
        print "Finished fmin"
    if _optMethod.lower() == "powell" or _optMethod.lower() == "all":
        tp1 = time.time()
        fitargs = opt.fmin_powell(lnLFunc, x0=magnification, xtol=1.e-6, args=args)
        tp2 = time.time()
        print "Finished powell"
    #tb1 = time.time()
    #fitargs = opt.fmin_bfgs(logLikelihood, x0=magnification, args=args)
    #tb2 = time.time()
    #print "Finished bfgs"
    if _optMethod.lower() == "brent" or _optMethod.lower() == "all":
        tbr1 = time.time()
        fitargs = opt.brent(lnLFunc, args=args, tol = 1.e-6,
                                 brack = [magnification-5.*_postWidthGuess,
                                          magnification,
                                          magnification+5.*_postWidthGuess])
        tbr2 = time.time()
        fitMu = fitargs
        print "Finished brent"
    if _optMethod.lower() == "golden" or _optMethod.lower() == "all":
        tg1 = time.time()
        fitargs = opt.golden(lnLFunc, args=args, tol = 1.e-6,
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
        FM = -1.*fisher_matrix(lnLFunc, fitMu, args, h = 1.e-5)
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

        print "Fitting to", y

        p0 = [fitMu, _postWidthGuess]
        bounds = [[fitMu-_postWidthGuess/1000., 1.e-14],[fitMu+_postWidthGuess/1000.,1.]]

        popt, pcurv = opt.curve_fit(log_gaussian_fn, x, y, p0=p0, bounds=bounds)
        print "Guassian fit with: ", popt
        #print "FIT CHECK:", log_gaussian_fn(x, *popt)
        #raw_input("Check")
        errorEst = popt[1]
    else:
        raise ValueError("No Error Estimate taken")

    kwargs["asReduced"] = False
    args = tuple(kwargs.values())

    if mlComparisonPlot:
        # -- Plot it using a brute force resampling -- this is really testing only
        x = np.linspace(fitMu-3.*errorEst, fitMu+3.*errorEst, 100)
        lnL = np.empty_like(x)
        for i,xv in enumerate(x):
            lnL[i] = -1.*lnLFunc(xv, *args)

        import pylab as pl
        ax = pl.subplot()
        ax.plot(x, np.exp(lnL-np.max(lnL)))
        ax.errorbar(fitMu, [0.5], xerr = errorEst, marker = "x")
        pl.show()

    SHUTDOWN()
    #print "Forced Exit:", myrank
    #_mpiKeepIterating = mycomm.bcast(False, root = 0)
    #print "GOODBYE"
    #exit()






