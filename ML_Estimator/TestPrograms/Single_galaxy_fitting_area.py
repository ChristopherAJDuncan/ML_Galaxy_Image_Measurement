import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None


numb_iter = 1000

area_value = np.array([.5,1,1.5,2.,2.5,3.0,3.5,4.0,4.5])
average_deviation = np.zeros(len(area_value))
standard_deviation = np.zeros(len(area_value))

for j in range(len(area_value)):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =10 # Has to equal a constan
	imageParams['SB']['e1'] = 0.0
	imageParams["SB"]['e2'] = 0.0
	imageParams["SB"]['size'] = area_value[j]
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*0.5 # Randomly selects centre


	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',), setParams = imageParams)
		iter_result[i] = result[0][0]
		iter_error[i] = (1/result[1][0])**2

	average_deviation[j] = (np.mean(iter_result)-imageParams["SB"]['size'])
	standard_deviation[j] = m.sqrt(1/(np.sum(iter_error)))

print average_deviation
print standard_deviation

import matplotlib.pyplot as plt

plt.errorbar(area_value,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')
plt.xlim([0,5])
plt.xlabel("Area of galaxy")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for a single galaxy, SNR = 10")
plt.show()


'''
imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['centroid'] = np.array([15,15])
imageParams['SB']["e1"] = 0.3
imageParams['SB']["e2"] = 0.2

image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
image_flattened = image.flatten()
result = imMeas.find_ML_Estimator(image_flattened, ('size',), galaxy_centroid = np.array([10,10])) #iParams = {"e1":0.3, "e2":.2}) #  ,
print result
import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()
'''



