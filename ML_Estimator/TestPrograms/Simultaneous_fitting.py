import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

der = None 
numb_iter = 1

area_value = np.array([2]) # .5,1.0,1.5,2.0,2.5,3,3.5,4.,4.5,5.0
average_deviation = np.zeros(len(area_value))
standard_deviation = np.zeros(len(area_value))


Sec_Gal_1 = modPro.default_ModelParameter_Dictionary()
Sec_Gal_1['SNR'] =20 # Has to equal a constan
Sec_Gal_1['SB']['e1'] = .2
Sec_Gal_1["SB"]['e2'] = 0.0
Sec_Gal_1["SB"]['flux'] = 10
Sec_Gal_1['stamp_size'] = [30,30]
Sec_Gal_1['centroid'] = (np.array(Sec_Gal_1['stamp_size'])+1)*0.8
Sec_Gal_1["SB"]['size'] = 2



Sec_Gal_2 = modPro.default_ModelParameter_Dictionary()
Sec_Gal_2['centroid'] = (np.array(Sec_Gal_1['stamp_size'])+1)*0.3


secondGal = {'Sec_Gal_1':Sec_Gal_1, 'Sec_Gal_2':Sec_Gal_2}


primaryGal = modPro.default_ModelParameter_Dictionary()
primaryGal['centriod'] = (np.array(primaryGal['stamp_size']))*0.5



for j in range(len(area_value)):

	primaryGal["SB"]['size'] = area_value[j]
	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):
		#print i, j
		
		extra_gal, disc = modPro.user_get_Pixelised_Model(secondGal['Sec_Gal_1'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image, disc = modPro.user_get_Pixelised_Model(primaryGal, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX,der = der)
		image = image + extra_gal
		extra_gal, disc = modPro.user_get_Pixelised_Model(secondGal['Sec_Gal_2'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image +extra_gal
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',), numbGalaxies = 2, second_gal_param = secondGal, secondFitParams = ('size','e1','centroid',))

		iter_result[i] = result[0][0]
		iter_error[i] = (1/result[1][0])

	average_deviation[j] = (np.mean(iter_result)-area_value[j])
	standard_deviation[j] = (1/(np.sum(iter_error)))

print average_deviation
print standard_deviation
'''
import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()


import matplotlib.pyplot as plt

plt.errorbar(area_value,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')
plt.xlim([0,5])
plt.xlabel("Fitted galaxy area")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, randomly varying 2nd galaxy (stamp size is 50x50)")
plt.show()
'''