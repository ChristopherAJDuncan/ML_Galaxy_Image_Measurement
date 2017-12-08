import os
import sys
sys.path.insert(0, os.path.abspath("../"))

import mypylib.utils.io as io
import numpy as np
import copy

import python.model_Production as modPro
import BuildData

saveAll = True

noiseDict = {"sky":114.,
             "readnoise":5.4,
             "gain":3.5}

fluxBoost = 1.e0

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

    combinedImages = {}
    combinedCatalogue = {}
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
        filename = "./Combined_Data.h5"
        io.save_dict_to_hdf5(filename, combinedImages)
        print ".. Output combined images to %s"%(filename)

        filename = "./Combined_Catalogue.h5"
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

def logLikelihood(mu, images, icovs, catalogue, signMod = +1):
    """

    :param mu:
    :param images:
    :param icovs:
    :param catalogue:
    :param signMod: If +1, this is a cost. If -1, this is log-likelihood
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
        model = None
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

    print "lnL was: ", lnL
    print "Expected is roughly: ", np.prod(model.shape)*lensed['nImage']
    print "Reduced is therefore: ", lnL/(np.prod(model.shape)*lensed['nImage'])

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

    BuildData.buildData(nGalaxy=100, magnification=magnification)

    strMu = str(magnification)

    # Denote the input catalogues
    inputDataFile = "./GEMS_sampled_data_lensed_"+strMu+".h5"
    inputCatalogueFile = "./GEMS_sampled_catalogue_unlensed_"+strMu+".h5"

    # Read in the data
    data = io.load_dict_from_hdf5(inputDataFile)
    print "... Loaded Data"
    catalogue = io.load_dict_from_hdf5(inputCatalogueFile)
    print "... Loaded Catalogue"

    print "Loaded ", catalogue['nGal'], " galaxy images"

    # Boost the flux for magnification measurement - this must boost unlensed flux value
    if fluxBoost > 1.:
        catalogue, data = boostFlux(copy.deepcopy(catalogue),
                                    copy.deepcopy(data))

    # Combine data into identical images, based on Poisson Distribution
    blendedImages, blendedCatalogue = combine_Images(data, catalogue, saveAll)

    # Get icovs, which is inverse covariance of the observed source
    import python.noiseDistributions as noiseMod
    import copy
    icovs = []
    for key, image in blendedImages.iteritems():
        icovs.append( 1./noiseMod.estimate_PN_noise(copy.deepcopy(image), **noiseDict) )

    # Add Noise to the images
    noisyBlendedImages = {}
    for key, val in blendedImages.iteritems():
        noisyBlendedImages[key] = noiseMod.add_PN_Noise(copy.deepcopy(val), **noiseDict)
    print "Produced ", len(noisyBlendedImages.keys()), " noisy blended images"

    if saveAll:
        filename = "./Noisy_Blended_Images_"+strMu+".h5"
        io.save_dict_to_hdf5(filename, noisyBlendedImages)
        print ".. Output noisy blended images to %s" % (filename)

    # Measure the magnification
    print "Measuring magnification: "
    import scipy.optimize as opt
    args = (noisyBlendedImages, icovs, blendedCatalogue, +1)

    # Testing: Check lnLikelihood at truth, and at value different to truth
    print "lnL at truth: ", logLikelihood(magnification, *args)
    print "lnL at 2x truth: ", logLikelihood(2.*magnification, *args)

    print "Fitting..."
    #fitMu = opt.fmin(logLikelihood, x0 = magnification, xtol = 1.e-6, args = args)

    x = np.linspace(1.19, 1.21, 100)
    lnL = np.empty_like(x)
    for i,xv in enumerate(x):
        lnL[i] = -1.*logLikelihood(xv, *args)

    import pylab as pl
    ax = pl.subplot()
    ax.plot(x, np.exp(lnL-np.max(lnL)))
    pl.show()








