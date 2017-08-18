import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None
angular_position = np.zeros([8,2])
angle_to_x = [0,45,90,135,180,225,270,315]
for i in range(len(angular_position)):
	angular_position[i,0] = m.sin(i*(m.pi)/4) 
	angular_position[i,1] = m.cos(i*(m.pi)/4) 


numb_iter = 1000

# Create on centre image

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = .2
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*.5

image_temp, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

'''
# Close in galaxies

average_deviation_close = np.zeros(8)
standard_deviation_close = np.zeros(8)


for position in range(len(angular_position)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = 15*0.5*angular_position[position,:] +15 # Randomly selects centre

	der = None

	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]
		iter_error[i] = 1/result[1][0]

	average_deviation_close[position] = (np.mean(iter_result)-2)
	standard_deviation_close[position] = 1/(np.sum(iter_error))

# Medium 

average_deviation_medium = np.zeros(8)
standard_deviation_medium = np.zeros(8)

for position in range(len(angular_position)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = 15*0.65*angular_position[position,:] +15 # Randomly selects centre

	der = None

	iter_result = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]

	average_deviation_medium[position] = (np.mean(iter_result)-2)
	standard_deviation_medium[position] = (np.std(iter_result))/numb_iter
	'''

# Far	

average_deviation_far = np.zeros(8)
standard_deviation_far = np.zeros(8)

for position in range(len(angular_position)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = 2
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = 15*0.8*angular_position[position,:] + 15 # Randomly selects centre

	der = None

	iter_result = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',))
		iter_result[i] = result[0][0]

	average_deviation_far[position] = (np.mean(iter_result)-2)
	standard_deviation_far[position] = (np.std(iter_result))/numb_iter







from matplotlib.pyplot import *



#errorbar(angle_to_x, average_deviation_close,yerr=standard_deviation_close, fmt = 'x', label = 'Close')
#errorbar(angle_to_x, average_deviation_medium,yerr=standard_deviation_medium, fmt = 'x', label = 'Medium')
errorbar(angle_to_x, average_deviation_far,yerr=standard_deviation_far, fmt = 'x', label = 'Far')
xlabel("Angular postion of the second galaxy")
ylabel("Mean deviation of area from true value")
title("Bias in area fitting algorithm with second galaxy at different angular position")
legend(loc='best')
axhline(0, color = 'k')
show()


		