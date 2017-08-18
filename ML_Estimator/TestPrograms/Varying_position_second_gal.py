import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None

radial_position = [0.5,0.51,0.52,.53,.54,.55,.56,.57,.58,.59,.60,.61,.62,.63,.64,.65,.665,.680,0.70,0.75,0.8,0.85]
numb_iter = 1000

average_deviation = np.zeros(len(radial_position))
standard_deviation = np.zeros(len(radial_position))


# Create off centre image

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = .2
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.5# Randomly selects centre

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)


for position in range(len(radial_position)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array([30,30])+1)*radial_position[position] # Randomly selects centre

	der = None

	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_error[i] = (1/result[1][0])**2

	average_deviation[position] = (np.mean(iter_result)-2)
	standard_deviation[position] = m.sqrt(1/(np.sum(iter_error)))

print average_deviation
print standard_deviation
from matplotlib.pyplot import *

errorbar(radial_position,average_deviation,yerr=standard_deviation, fmt = 'x', label = 'Not fitting Eccentricity ')

xlabel("Radial position as fraction of stamp size (stamp size is 30x30 pixels)")
ylabel("Mean deviation of area from true value")
title("Bias in area fitting algorithm with second galaxy at different radial position, 1000 realization per point & no noise")
xlim([0.45,.95])
axhline(0, color = 'k')
show()



		