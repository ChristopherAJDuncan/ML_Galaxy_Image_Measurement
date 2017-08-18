import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None

# Create off centre image

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 1.5
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.83# Randomly selects centre

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)



numb_iter = 10000 # Fitting e1 and e2 ran 10000 


area_values_plot = np.array([.5,1.0,1.5,2.0,2.5,3,3.5,4.,4.5])#.5,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,4.25,4.5,4.75,5.0]) #


# Rest of fitting script same as for one image


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
	iter_likelihood = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_likelihood[i] = 1/result[1][0]

	average_deviation[area_value] = (np.mean(iter_result)-area_values_plot[area_value])
	standard_deviation[area_value] = 1/(np.sum(iter_likelihood))

print average_deviation
print standard_deviation


# Allowing e1 and e2 to fit too
'''
average_deviation_ecen = np.zeros(len(area_values_plot))
standard_deviation_ecen = np.zeros(len(area_values_plot))

average_ecen_1 = np.zeros(len(area_values_plot))
standard_deviation_ecen_1 = np.zeros(len(area_values_plot))

average_ecen_2 = np.zeros(len(area_values_plot))
standard_deviation_ecen_2 = np.zeros(len(area_values_plot))

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
	iter_result_ecen_1 = np.zeros(numb_iter)
	iter_result_ecen_2 = np.zeros(numb_iter)

	iter_result_error = np.zeros(numb_iter)
	iter_result_ecen_1_error = np.zeros(numb_iter)
	iter_result_ecen_2_error = np.zeros(numb_iter)

	for i in range(numb_iter):
		print area_values_plot[area_value]
		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size','e1','e2'))
		iter_result[i] = result[0][0]
		iter_result_ecen_1[i] = result[0][1]
		iter_result_ecen_2[i] = result[0][2]

		iter_result_error[i] = 1/result[1][0]
		iter_result_ecen_1_error[i] = 1/result[1][1]
		iter_result_ecen_2_error[i] = 1/result[1][2]

	average_deviation_ecen[area_value] = (np.mean(iter_result)-area_values_plot[area_value])
	standard_deviation_ecen[area_value] = 1/(np.sum(iter_result_error))

	average_ecen_1[area_value] = (np.mean(iter_result_ecen_1))
	standard_deviation_ecen_1[area_value] = 1/(np.sum(iter_result_ecen_1_error))/numb_iter

	average_ecen_2[area_value] = (np.mean(iter_result_ecen_2))
	standard_deviation_ecen_2[area_value] = 1/(np.sum(iter_result_ecen_2_error))/numb_iter


print average_ecen_1
print average_ecen_2

from matplotlib.pyplot import *

subplot(2,1,1)
title("Bias in area fitting algorithm with second galaxy allowing e1 and e2 to fit")
errorbar(area_values_plot,average_deviation_ecen,yerr=standard_deviation_ecen, fmt = 'x', label = 'Fitting Eccentricity ')
axhline(0, color ='k')
ylabel("Mean deviation from true value")
#ax1.xlim([0,5.5])

subplot(2,1,2)
errorbar(area_values_plot,average_ecen_1,yerr=standard_deviation_ecen_1, fmt='x', label = 'e1 with eccentricity fitting')
errorbar(area_values_plot,average_ecen_2,yerr=standard_deviation_ecen_2, fmt='x', label = 'e2 with eccentricity fitting')
axhline(0, color ='k')
legend(loc = "best")
xlabel('Galaxy size')
ylabel('Eccentricity')
#ax2.xlim([0,5.5])

show()

'''
import matplotlib.pyplot as plt

plt.errorbar(area_values_plot,average_deviation,yerr=standard_deviation, fmt = 'x', label = 'Not fitting Eccentricity ')
#plt.errorbar(area_values_plot,average_deviation_ecen,yerr=standard_deviation_ecen, fmt = 'x', label = 'Fitting Eccentricity ')

plt.xlabel("Galaxy area")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm with second galaxy varying area of the central galaxy (not fitting secondary galaxy")
plt.axhline(0, color ='k')
plt.show()







		