# Produces and outputs nSample realisations of an underlying galaxy surface brightness profile
import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import pylab as pl
import python.IO as IO
import os

#Define the number of samples to use
nReal = 1000
Output = "./Realisations/"

#Define Image to be simulated
imageShape = (10, 10)
imageParams = modPro.default_ModelParameter_Dictionary(SB = dict(size = 1.41, e1 = 0.0, e2 = 0.0, magnification = 1., shear = [0., 0.], flux = 1000, modelType = 'gaussian'),\
                                                       centroid = (np.array(imageShape)+1)/2., noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1.,\
                                                       PSF = dict(PSF_Type = 0, PSF_size = 0.05, PSF_Gauss_e1 = 0., PSF_Gauss_e2 = 0.0))

def produce_Realisations(imageParams, nReal = 1000, ccdSpecs = None, noiseFunc = nDist.PN_Likelihood, outputPrefix = os.path.join(Output,"Realisations.dat")): 
#Single Run - Derivative
    print 'Running Data Production'
    
##Surface Brightness profile routine
    image, imageParams = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Weave)
    
#Start outputting. First line is always the noise free model
    handle = IO.initialise_Output(outputPrefix, mode = "a")
    handle.write("# Flux:"+str(imageParams['SB']['flux'])+" e1:"+str(imageParams['SB']['e1'])+" e2:"+str(imageParams['SB']['e2'])+
                 " size:"+str(imageParams['SB']['size'])+" stamp_size:"+str(imageParams['stamp_size'])+" centroid:"+str(imageParams['centroid']) +"\n")
    handle.write("# First line is noise-free image \n")
    
#Produce noise realisations by sampling from the correct distribution
    
#--Flatten image
    fImage = image.flatten()
    
    np.savetxt(handle, fImage.reshape(1,fImage.shape[0]))
    
#-Define CCD specs
    if(ccdSpecs is None):
        ccdSpecs = dict(qe = 0.9, charge = 0.001, readout = 1., ADUf = 1)
    
    nImage = np.zeros((nReal,fImage.shape[0]))
    for i in range(fImage.shape[0]):
        counts, pdf = noiseFunc(fImage[i], ccdSpecs)
        
        try:
            nImage[:,i] = nDist.inverse_Sample(counts, pdf, nReal)
        except ValueError as e:
            print "Sampling failed for likelihood shown::"
            print "Exception raised: ", e.args[0]
            f = pl.figure(); ax = f.add_subplot(111)
            ax.plot(counts, pdf)
            pl.show()
            exit()
            
        np.savetxt(handle, nImage[i].reshape(1,nImage.shape[1]))

    print "Finished sampling"
    handle.close()

    return fImage, nImage


if __name__ == "__main__":

    fImage, nImage = produce_Realisations(imageParams, nReal)

#Output to screen
    f = pl.figure()
    
    ax = f.add_subplot(121)
    im = ax.imshow(fImage.reshape(imageParams['stamp_size']), interpolation = 'nearest')
    pl.colorbar(im)
    
    ax = f.add_subplot(122)
    print fImage.shape, nImage[:,0].shape
    ax.imshow(nImage[:,0].reshape(imageParams['stamp_size']), interpolation = 'nearest')
    
    
    pl.show()
