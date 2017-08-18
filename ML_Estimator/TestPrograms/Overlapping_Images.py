import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m



# Randomly creates galaxy parameters
print 'Running'
def changing_centriod(frac):

	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] =20 # Has to equal a constan
	imageParams['SB']['e1'] = 0
	imageParams["SB"]['e2'] = 0.5
	imageParams["SB"]['size'] = 3
	imageParams["SB"]['flux'] = 5
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)*frac # Randomly selects centre
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


#der = ['e1', 'e2']
der = None

import pylab as pl

fig, AX = pl.subplots(nrows = 6, ncols = 5, sharey = True, sharex = True)
alpha = 0


for row in AX:
	i = 0.1
	 # indexes the coloumn
	for ax in row:
		image = np.zeros((30,30))

		image_temp, disc = modPro.user_get_Pixelised_Model(changing_centriod(i), noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image +=overlapping_functions(image,image_temp,alpha)

		image_temp, disc = modPro.user_get_Pixelised_Model(changing_centriod((1.0-i)), noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image += overlapping_functions(image,image_temp,alpha)


		fig.add_subplot(ax,row)
		
		im = ax.imshow(image, interpolation = 'nearest')
		pl.colorbar(im)
		#ax.set_title(str(alpha))
		i +=0.1
		print ax, row
	alpha +=.2

pl.suptitle(r"Modelling Blending of Galaxies (alpha increase down the page)", fontsize = 28)


pl.show()


