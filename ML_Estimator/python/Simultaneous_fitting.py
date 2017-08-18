import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None 
numb_iter = 100

area_value = np.array([.5,.75,0.80,.85,.9,1.,1.15,1.3,1.45,1.6,1.75,2.,2.25,2.5,3.5,4]) # .5,1.0,1.5,2.0,2.5,3,3.5,4.,4.5,5.0
average_deviation = np.zeros(len(area_value))
standard_deviation = np.zeros(len(area_value))


fittingParams = modPro.default_ModelParameter_Dictionary()
fittingParams['SNR'] =20 # Has to equal a constan
fittingParams['SB']['e1'] = 0.0
fittingParams["SB"]['e2'] = 0.0
fittingParams["SB"]['flux'] = 10
fittingParams['stamp_size'] = [50,50]
fittingParams['centroid'] = (np.array(fittingParams['stamp_size'])+1)*0.5

imageParams = modPro.default_ModelParameter_Dictionary()

# Creates extra galaxy (the one not being fitted)

imageParams = modPro.default_ModelParameter_Dictionary()

print get_Image_params(imageParams)
print "hello"
for j in range(len(area_value)):

	fittingParams["SB"]['size'] = area_value[j]
	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):
		print i, j
		
		extra_gal, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image, disc = modPro.user_get_Pixelised_Model(fittingParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX,der = der)
		image = image + extra_gal
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)

		iter_result[i] = result[0]
		iter_error[i] = (1/result[1][0])

	average_deviation[j] = (np.mean(iter_result)-area_value[j])
	standard_deviation[j] = (1/(np.sum(iter_error)))

print average_deviation
print standard_deviation

import matplotlib.pyplot as plt

plt.errorbar(area_value,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')
plt.xlim([0,5])
plt.xlabel("Fitted galaxy area")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, randomly varying 2nd galaxy (stamp size is 50x50)")
plt.show()