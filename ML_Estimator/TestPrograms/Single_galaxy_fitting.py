import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

numb_iter = 50

area_values_plot = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
average_deviation = np.zeros(len(area_values_plot))
standard_deviation = np.zeros(len(area_values_plot))

for area_value in range(len(area_values_plot)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = area_values_plot[area_value]
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre

	der = None

	iter_result = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]


	print iter_result


	average_deviation[area_value] = (np.mean(iter_result)-area_values_plot[area_value])
	standard_deviation[area_value] = (np.std(iter_result))/numb_iter

print average_deviation
print standard_deviation

import matplotlib.pyplot as plt

plt.errorbar(area_values_plot,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')
plt.xlim([0,5.5])
plt.xlabel("Galaxy area")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm")
plt.show()