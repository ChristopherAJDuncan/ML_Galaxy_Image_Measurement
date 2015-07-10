import src.model_Production as modPro
import src.surface_Brightness_Profiles as SBPro
import numpy as np

#Single Run - Derivative
print 'Running'

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 35.
imageParams['e1'] = 0.3
imageParams['e2'] = 0.
imageParams['size'] = 0.84853
imageParams['flux'] = 4.524
imageParams['stamp_size'] = [10,10]
imageParams['centroid'] = (np.array(imageParams['stamp_size'])+1)/2.


###Get image using GALSIM default models
#image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True)
#image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)

##A Halls version
#image = np.genfromtxt('/home/cajd/Downloads/fid_image.dat')
image = np.genfromtxt('/home/cajd/Downloads/dfid_image.dat')
image = image.T
print 'Got Original'

print 'Halls:', np.power(image,2.).sum()

###Get image using user-specified surface brightness model
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)


#imageSB, imageParams = modPro.user_get_Pixelised_Model(imageParams, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = ['e1', 'e1'])

from src.derivatives import finite_difference_derivative
imageSB = finite_difference_derivative(modPro.get_Pixelised_Model_wrapFunction, 0.3, args = [imageParams, 'e1', 1], n = [1], dx = [0.001, 0.001], order = 5, eps = 1.e-3, convergenceType = 'sum', maxEval = 100)

print 'Mine:', np.power(imageSB,2.).sum()

print 'Ratio of Sum of Image:', image.sum(), imageSB.sum(), image.sum()/imageSB.sum(), imageParams['flux']

import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image)
pl.colorbar(im)
ax = f.add_subplot(212)
im = ax.imshow(imageSB)
pl.colorbar(im)
pl.show()


### Plot Residual
import pylab as pl

pl.imshow((imageSB-image))
#pl.set_title('GALSIM - User-Specified')
pl.colorbar()

pl.show()
