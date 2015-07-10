import src.model_Production as modPro
import src.surface_Brightness_Profiles as SBPro
import numpy as np

#Single Run - Derivative
print 'Running'

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 200.
imageParams['e2'] = 0.
imageParams['size'] = 6.
imageParams['stamp_size'] = [10,10]
imageParams['centroid'] = (np.array(imageParams['stamp_size'])+1)/2.


###Get image using GALSIM default models
#image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True)
#image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)

image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)
###Get image using user-specified surface brightness model
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)
imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)


print 'Ratio of Sum of Image:', image.sum(), imageSB.sum(), image.sum()/imageSB.sum(), imageParams['flux']

### Plot Residual
import pylab as pl

#pl.imshow((imageSB/image))
pl.imshow((image))
#pl.set_title('GALSIM - User-Specified')
pl.colorbar()

#pl.imshow((imageSB-image))
#pl.set_title('GALSIM - User-Specified')
#pl.colorbar()

pl.show()
