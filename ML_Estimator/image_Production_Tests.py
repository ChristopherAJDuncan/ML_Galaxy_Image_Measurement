import src.model_Production as modPro
import src.surface_Brightness_Profiles as SBPro
import src.image_measurement_ML as imMeas
import numpy as np

#Single Run - Derivative
print 'Running'

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 35.
imageParams['e1'] = 0.6
imageParams['e2'] = 0.2
imageParams['size'] = 8.0
imageParams['flux'] = 4.524
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams['stamp_size'])+1)/2.


###Get image using GALSIM default models
#image, disc = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, Verbose = True, sbProfileFunc = modPro.gaussian_SBProfile)
#image, imageParams = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)


##Surface Brightness profile routine
image, imageParams = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Weave, der = ['e1', 'e1'])

imageSB, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = ['e1','e1'])

### User-defined model with Guassian noise
#image = np.genfromtxt('./TestPrograms/Hall_Models/fid_image.dat')
#image = image.T



#print 'image Noise Estimated as:', imMeas.estimate_Noise(image, maskCentroid = imageParams['centroid'])

#print 'imageSB Noise Estimated as:', imMeas.estimate_Noise(imageSB, maskCentroid = imageParams['centroid'])

##A Halls version

#image = np.genfromtxt('/home/cajd/Downloads/dfid_image.dat')
#image = image.T
#print 'Got Original'

#print 'Halls:', np.power(image,2.).sum()

###Get image using user-specified surface brightness model
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)


#imageSB, imageParams = modPro.user_get_Pixelised_Model(imageParams, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = ['e1', 'e1'])

print 'Final Check:: sumImage, sumImageSB, ratioSum(image/SB), flux:', image.sum(), imageSB.sum(), image.sum()/imageSB.sum(), imageParams['flux']

import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)
ax = f.add_subplot(212)
im = ax.imshow(imageSB, interpolation = 'nearest')
pl.colorbar(im)
pl.show()


### Plot Residual
import pylab as pl

pl.imshow((imageSB-image))
#pl.set_title('GALSIM - User-Specified')
pl.colorbar()

pl.show()
