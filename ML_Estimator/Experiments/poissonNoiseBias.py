import init_Build
init_Build.init_build()

import os
import sys
import numpy as np
import copy
import model_Production as modPro
import IO

verbose = True

output = os.path.abspath(os.path.join(os.getcwd(), "../Output/Gaussian/"))

minimiseMethod = 'emcee'
errorType = 'Fisher'

loopParam = "e1"

fitParams = (loopParam, "flux")
initialGuess = dict(e1 = 0.11, flux = 9500.)

biasCorrect = 0

# DISABLED FOR DEBUG nReal = 1000000
nReal = 10000

#Set up the ground truth
SB = dict(size = 1.41, e1 = 0.0, e2 = 0.0, magnification = 1., shear = [0.,0.], flux = 100, modelType = 'gaussian', bg = 0.)
PSF = dict(PSF_Type = 0, PSF_size = 0.05, PSF_Gauss_e1 = 0., PSF_Guass_e2 = 0.0)
imageShape = (10,10)
groundParams = modPro.default_ModelParameter_Dictionary(SB = SB, PSF = PSF, centroid = (np.array(imageShape)+1)/2., noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1.)


#Debugging
plotImage = False
    
def run(imageParams = None, verbose = False):
    """
    The work horse
    """
    import image_measurement_ML as ML
    import surface_Brightness_Profiles as SBPro
    import produce_Data as proDat
    import noiseDistributions as noiseDist
    import IO

    if(imageParams is None):
        print "\n Using Default Dictionary (as in run)"
        
        #Set up a reasonable output
        SB = dict(size = 1.41, e1 = 0.1, e2 = 0.0, magnification = 1., shear = [0.,0.], flux = 1000, modelType = 'gaussian')
        PSF = dict(PSF_Type = 0, PSF_size = 0.001, PSF_Gauss_e1 = 0., PSF_Guass_e2 = 0.0)
    
        imageShape = (10,10)
        imageParams = modPro.default_ModelParameter_Dictionary(SB = SB, PSF = PSF, centroid = (np.array(imageShape)+1)/2., noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1.)

    if(verbose):
        print "\nGround Truth:"
        modPro.print_ModelParameters(imageParams)
    
    groundTruth = copy.deepcopy(imageParams)

    #Produce Realisations
    #ccdSpecs = dict(readout = 4., ADUf = 1.) #Gaussian
    #ccdSpecs = dict(qe = 0.9, charge = 0.001, readout = 1., ADUf = 1) #PGN

    ccdSpecs = dict(qe = 0.9, charge = 0.0001, readout = 1., ADUf = 1) #PGN simplified

    #Alternate is to use noiseDist.PN_Likelihood
    data = proDat.produce_Realisations(imageParams, nReal, ccdSpecs, noiseDist.PN_Likelihood, os.path.join(output, "Realisations.dat"), suppressOutput = True)

    #Get covariance
    # import scipy
    # cova = scipy.cov(data[1], rowvar = False)
    # print 'Cov shape:', cova.shape, data[1].shape
    
    if(verbose):
        print 'Constructed data with shape:'
        print data[1].shape

    if(plotImage):
        import pylab as pl
        f = pl.figure()
        ax = f.add_subplot(111)
        
        import mypylib.plot.images as myim
        myim.imageShow(ax, data[0][:].reshape(imageParams['stamp_size']))
        #myim.imageShow(ax, data[0][0,:].reshape(imageParams['stamp_size']))
        

        #ax = f.add_subplot(212)
        #ax.imshow(cova)
        
        pl.show()
    
    #Maximise Likelihood for data
    MaxL = ML.find_ML_Estimator(data[1], fitParams = fitParams, error = errorType, calcNoise = ML.estimate_Noise, setParams = imageParams.copy(), searchMethod = minimiseMethod, **initialGuess)

    bias = (MaxL[0]-modPro.unpack_Dictionary(groundTruth, requested_keys = fitParams))[0]

    if(verbose):
        print "Maximum Likelihood found to be: " , MaxL[0][0], " +- ", MaxL[1][0]
        print "Bias is :", bias, " +- ", MaxL[1][0]
        print "With significance ", abs(bias)/MaxL[1][0]

    
    #Maximise Likelihood for data
    MaxL = ML.find_ML_Estimator(data[1], fitParams = fitParams, error = None, calcNoise = ML.estimate_Noise, setParams = imageParams.copy(), searchMethod = 'simplex', **initialGuess)

    bias = (MaxL[0]-modPro.unpack_Dictionary(groundTruth, requested_keys = fitParams))[0]

    if(verbose):
        print "Maximum Likelihood found to be: " , MaxL[0][0], " +- ", MaxL[1][0]
        print "Bias is :", bias, " +- ", MaxL[1][0]
        print "With significance ", abs(bias)/MaxL[1][0]

    raw_input("Check difference")
        
    return MaxL[0][0], bias, MaxL[1][0]

def run_loopParam(imageParams = None, parLab = None, parVals = None, verbose = False):
    
    if(imageParams is None):
        raise RuntimeError("run_loopParam: Image Parameter dictionary must be passed for this to work")
    
    if(parLab is None):
        raise RuntimeError("run_loopParam: Parameter label must be passed for this to work")

    if(parVals is None):
        raise RuntimeError("run_loopParam: Parameter values must be passed for this to work")
    
    ML, bias, err = [],[],[]
    for val in parVals:
        print "\n\nConsidering ", parLab , " = ", val
        
        imageParams['SB'][parLab] = val

        result = run(imageParams, verbose)

        ML.append(result[0])
        bias.append(result[1])
        err.append(result[2])

    print "Finished loop. Outputting"

    ML = np.array(ML)
    bias = np.array(bias)
    err = np.array(err)
    
    import h5py as h5
    f = h5.File(os.path.join(output, "BiasLoop.h5"), 'w')
    dset = f.create_dataset("ParLabel", data = parLab)
    dset = f.create_dataset("ParVals", data = parVals)
    dset = f.create_dataset("ML", data = ML)
    dset = f.create_dataset("bias", data = bias)
    dset = f.create_dataset("err", data = err)
    f.close()

    ##Plot
    import pylab as pl
    f = pl.figure()
    ax = f.add_subplot(111)

    x = np.array(parVals)
    ax.plot(parVals,bias)#, label = str(imageParams['SB']['flux']))
    ax.plot(parVals,bias+err, color = 'cyan', linestyle = '--')
    ax.plot(parVals,bias-err, color = 'cyan', linestyle = '--')
    ax.fill_between(parVals, bias-err, bias+err, color = 'cyan', alpha = 0.5)
    
    pl.show()

    
if __name__ == "__main__":
    IO.initialise_Directory(output)
    
    run_loopParam(groundParams, loopParam, np.arange(-0.5, -0.4, 0.1), verbose)
    
    #run(groundParams, verbose)
