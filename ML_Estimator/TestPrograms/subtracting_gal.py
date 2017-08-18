import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = np.array([15,15])

image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = np.array([23,23])

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der) 
 
image = image+image_temp

import pylab as pl
f = pl.figure()
pl.title("Both galaxies")
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()

image = image - image_temp

f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()

image_sub, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der) 

image = image- image_sub+image_temp


f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()
