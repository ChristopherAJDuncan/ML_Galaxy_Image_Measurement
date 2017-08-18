import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
from matplotlib.pyplot import *

#Single Run - Derivative
rc('text', usetex=True)
rc('font', family='serif')

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 30
imageParams['SB']['e1'] = .12
imageParams["SB"]['e2'] = .05
imageParams["SB"]['size'] = 2.5
imageParams["SB"]['flux'] = 14
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)/2

#der = ['e1', 'e2']
der = None

image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

##Surface Brightness profile routine

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 30
imageParams['SB']['e1'] = -.40
imageParams["SB"]['e2'] = .20
imageParams["SB"]['size'] = 1.8
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array([10,18]))

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

image += image_temp

noise =  0.035*np.random.randn(30,30)
image += noise

f = figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
colorbar(im)
title(r'Input image')
show()


# Under predicting size

size_estimates = [2.0,2.5,3.0]

for i in size_estimates:

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] = 30
	imageParams['SB']['e1'] = -.40
	imageParams["SB"]['e2'] = .20
	imageParams["SB"]['size'] = 1.8
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array([10,18]))

	image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
	res = image - image_temp

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] = 30
	imageParams['SB']['e1'] = .12
	imageParams["SB"]['e2'] = .05
	imageParams["SB"]['size'] = i
	imageParams["SB"]['flux'] = 14
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)/2

	image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
	res -= image_temp

	f = figure()
	ax = f.add_subplot(211)
	im = ax.imshow(res, interpolation = 'nearest')
	colorbar(im)
	title(r'Residuals')
	show()
