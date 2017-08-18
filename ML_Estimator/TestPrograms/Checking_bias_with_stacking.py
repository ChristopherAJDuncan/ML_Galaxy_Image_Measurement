import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

numb_iter = 1000000

der = None

area_value = np.array([1,1.5,2,2.5,3,3.5,4])
average_deviation = np.zeros(len(area_value))
standard_deviation = np.zeros(len(area_value))

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre

for i in range(len(area_value)):
	image  = np.zeros([30,30])
	imageParams['SNR'] = 20

	for j in range(numb_iter):
		imageParams["SB"]['size'] = area_value[i]
		im_to_add, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image += im_to_add

	imageParams['SNR'] = 20.0*m.sqrt(numb_iter)
	imageParams["SB"]['flux'] = 10*numb_iter
	image_flattened = image.flatten()
	result = imMeas.find_ML_Estimator(image_flattened, ('size','flux'), setParams = imageParams)
	average_deviation[i] = result[0][0] -area_value[i]
	standard_deviation[i] = result[1][0]
	'''
	import pylab as pl
	f = pl.figure()
	ax = f.add_subplot(211)
	im = ax.imshow(image, interpolation = 'nearest')
	pl.colorbar(im)

	pl.show()
	'''

import matplotlib.pyplot as plt 

plt.errorbar(area_value,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')

plt.xlabel("Area")
plt.xlim([0,4.5])
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for a single galaxy")
plt.show()
