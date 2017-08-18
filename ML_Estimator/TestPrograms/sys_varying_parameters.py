import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist

#Single Run - Derivative
#print 'Running'
def get_Image_param(x,y):
	
	imageParams = modPro.default_ModelParameter_Dictionary()
	imageParams['SNR'] = 20
	imageParams['SB']['e1'] = float(x)
	imageParams["SB"]['e2'] = float(y)
	imageParams["SB"]['size'] = 4.0
	imageParams["SB"]['flux'] = 10
	imageParams['stamp_size'] = [30,30]
	imageParams['centroid'] = (np.array(imageParams["stamp_size"])+1)/2
	return imageParams
#der = ['e1', 'e2']
der = None

###Get image using GALSIM default models
#image, disc = modPro.get_Pixelised_Model(imageParams, noiseType = 'Gaussian', outputImage = True, Verbose = True, sbProfileFunc = modPro.gaussian_SBProfile)
#image, imageParams = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)

##Surface Brightness profile routine
#image, disc = modPro.user_get_Pixelised_Model(get_Image_param(.7,0), noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

### User-defined model with Guassian noise
#image = np.genfromtxt('./TestPrograms/Hall_Models/fid_image.dat')
#image = image.T

#print "Produced CXX image"


#print 'image Noise Estimated as:', imMeas.estimate_Noise(image, maskCentroid = imageParams['centroid'])

#print 'imageSB Noise Estimated as:', imMeas.estimate_Noise(imageSB, maskCentroid = imageParams['centroid'])

##A Halls version

#image = np.genfromtxt('/home/cajd/Downloads/dfid_image.dat')
#image = image.T
#print 'Got Original'

#print 'Halls:', np.power(image,2.).sum()

###Get image using user-specified surface brightness model
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)
#imageSB, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)


#imageSB, imageParams = modPro.user_get_Pixelised_Model(imageParams, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy, der = ['e1', 'e1'])

#print 'Final Check:: sumImage, sumImageSB, ratioSum(image/SB), flux:', image.sum(), imageSB.sum(), image.sum()/imageSB.sum(), imageParams['flux']

import pylab as pl

fig, AX = pl.subplots(nrows = 5, ncols = 5, sharey = True, sharex = True)



j=-4 # indexes row

for row in AX:
	i = -4 # indexes the coloumn
	for ax in row:
		fig.add_subplot(ax,row)
		image, disc = modPro.user_get_Pixelised_Model(get_Image_param(0.1*i,0.1*j), noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		im = ax.imshow(image, interpolation = 'nearest')
		#ax[i+5,j+5].imshow(image,interpolation = 'nearest')
		pl.colorbar(im)
		ax.set_title("[%.1f, %.1f]" %(0.1*i,0.1*j))
		i +=2
	j +=2

# ax = f.add_subplot(212)
# im = ax.imshow(imageSB, interpolation = 'nearest')
# pl.colorbar(im)
pl.rc('text', usetex = True)
pl.rc('font', family = 'serif')
pl.suptitle(r"$[e_{1},e_{2}]", fontsize = 28)

pl.show()


### Plot Residual
#import pylab as pl

#pl.imshow((imageSB-image))
#pl.set_title('GALSIM - User-Specified')
#pl.colorbar()

#pl.show()
