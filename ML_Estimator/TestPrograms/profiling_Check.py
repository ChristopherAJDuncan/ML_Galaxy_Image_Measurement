import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist

#Single Run - Derivative
print 'Running'

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 20.
imageParams["SB"]['e1'] = 0.
imageParams["SB"]['e2'] = 0.
imageParams["SB"]['size'] = 2.0
imageParams["SB"]['flux'] = 4.524
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams['stamp_size'])+1)/2.

print "Image Params:", imageParams

#der = ['e1', 'e2']
der = None


##Surface Brightness profile routine
image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

print "Produced CXX image"

#----------------------------- Test for model variation
# imageParams["SB"]['size'] = imageParams["SB"]['size']*2
# wrongImage, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

# imageParams["SB"]['size'] = imageParams["SB"]['size']/2
# rightImage, disc  = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

# print "Testing model parameter variation in image construction"
# print "Wrong:"
# diff =  wrongImage-image
# print "Diff:", diff.sum(), diff.max(), diff.min()
# print "Right:"
# diff =  rightImage-image
# print "Diff:", diff.sum(), diff.max(), diff.min()

# print "image Parameters check:" , imageParams

# exit()
#------------------------------------------------------

### Test log-likelihood recovery and profiling
nIter = 10000
import time
t1 = time.time()
for i in range(nIter):
    wrongLL = imMeas.get_logLikelihood([1.],['size'], image.flatten(), imageParams)
t2 = time.time()
rightLL  = imMeas.get_logLikelihood([2.], ['size'], image.flatten(), imageParams)


print "Checking logLikelihood: "
print "Right:", rightLL
print "wrong:", wrongLL

if(nIter > 1):
    print "Time check:", t2-t1, (t2-t1)/nIter

exit()
