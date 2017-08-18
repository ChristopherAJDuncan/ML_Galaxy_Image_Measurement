import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None
flux_frac = np.array([0.25,0.35,0.45,.55,.75,1,1.25,1.5,1.75,2.0])
numb_iter = 300

# Create off centre image

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 1.5
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.6# Randomly selects centre

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)


average_deviation_close = np.zeros(len(flux_frac))
standard_deviation_close = np.zeros(len(flux_frac))

for frac in range(len(flux_frac)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2.0
	imageParams["SB"]['flux'] = 10*flux_frac[frac]
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre

	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_error[i] = 1/result[1][0]

	average_deviation_close[frac] = (np.mean(iter_result)-2)
	standard_deviation_close[frac] = 1/(np.sum(iter_error))

# Medium


imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 1.5
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.70# Randomly selects centre

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)


average_deviation_med = np.zeros(len(flux_frac))
standard_deviation_med = np.zeros(len(flux_frac))

for frac in range(len(flux_frac)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2.0
	imageParams["SB"]['flux'] = 10*flux_frac[frac]
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre

	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_error[i] = 1/result[1][0]

	average_deviation_close[frac] = (np.mean(iter_result)-2)
	standard_deviation_close[frac] = 1/(np.sum(iter_error))


# Far

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 1.5
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.80# Randomly selects centre

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)


average_deviation_far = np.zeros(len(flux_frac))
standard_deviation_far = np.zeros(len(flux_frac))

for frac in range(len(flux_frac)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2.0
	imageParams["SB"]['flux'] = 10*flux_frac[frac]
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre

	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_error[i] = 1/result[1][0]

	average_deviation_close[frac] = (np.mean(iter_result)-2)
	standard_deviation_close[frac] = 1/(np.sum(iter_error))





from matplotlib.pyplot import *

#plt.errorbar(area_values_plot,average_deviation,yerr=standard_deviation, fmt = 'x', label = 'Not fitting Eccentricity ')
errorbar(flux_frac, average_deviation_close ,yerr=standard_deviation_close, fmt = 'x', label = 'Close')
errorbar(flux_frac, average_deviation_med ,yerr=standard_deviation_med, fmt = 'x', label = 'Medium')
errorbar(flux_frac, average_deviation_far ,yerr=standard_deviation_far, fmt = 'x', label = 'Far')
xlabel("Relative flux")
legend(loc = 'best')
axhline(0, color = 'k')
xlim([0,2.2])
ylabel("Mean deviation from true value")
title("Bias in area fitting algorithm with second galaxy varying relative flux")
show()









		