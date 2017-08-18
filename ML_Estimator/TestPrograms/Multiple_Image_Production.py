import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m

Numb_Galaxies = 30 # Number of galxies wanted on the postage stamp

# Randomly creates galaxy parameters
print 'Running' 
def get_Image_params():
	stamp_Size = 100
	Gal_size = np.random.normal(2,1)

	while Gal_size<2:
		Gal_size = np.random.normal(3,.25)
	else:
		pass

	Flux = np.random.normal(2,.25)

	while Flux<2:
		Flux = np.random.normal(5,1)
	else:
		pass

	ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]

	while m.sqrt(ellipticity[0]**2+ellipticity[1]**2)>.71:
		ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]
	else:
		pass

	centroid = np.random.rand(2)*stamp_Size 

	while (centroid[0]<1 or centroid[0]>(stamp_Size-1) or centroid[1]<1 or centroid[1]>(stamp_Size-1)):
		ellipticity = np.random.rand(2) * stamp_Size
	else:
		pass


	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = ellipticity[0]
	imageParams["SB"]['e2'] = ellipticity[1]
	imageParams["SB"]['size'] = Gal_size
	imageParams["SB"]['flux'] = Flux
	imageParams['stamp_size'] = [100,100]
	imageParams['centroid'] = centroid # Randomly selects centre
	return imageParams

def overlapping_functions(overall_image, adding_image, alpha):

	shape = np.shape(overall_image)
	for i in range(shape[0]): # two for loop iterate over image matric
		for j in range(shape[1]):
			if overall_image[i,j]<10**(-4): # If no galaxy there nothing happens
				pass
			elif (adding_image[i,j]<=alpha*overall_image[i,j]): # If the fore ground too dense (bright) no image added
				adding_image[i,j] = 0
	return adding_image


der = None




image = np.zeros((100,100))

for galaxy in range(Numb_Galaxies-1):
	image_temp, disc = modPro.user_get_Pixelised_Model(get_Image_params(), noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
	image += overlapping_functions(image,image_temp,0.8) 

# Adding noise

image_temp, disc = modPro.user_get_Pixelised_Model(get_Image_params(), noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
image += image_temp


import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()
