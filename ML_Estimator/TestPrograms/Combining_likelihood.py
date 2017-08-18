import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m
import scipy.optimize as opt
der = None

def get_Image_params(dict, size, centered = 'n'): # randomly creates images parameters
	stamp_Size = 30

	if size != None:
		dict['SB']["size"] = size


	else:
		Gal_size = np.random.normal(2,1)
		while Gal_size<1.5:
			Gal_size = np.random.normal(3,.25)
		else:
			dict['SB']["size"] = Gal_size
		

	Flux = np.random.normal(2,.25)

	while Flux<2:
		Flux = np.random.normal(5,1)
	else:
		dict["SB"]['flux']

	ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]

	while ellipticity[0]**2+ellipticity[1]**2>(.71)**2:
		ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]
	else:
		dict["SB"]['e1'] = ellipticity[0]
		dict["SB"]['e2'] = ellipticity[1]
	if centered == 'y':
		dict['centroid'] = np.array([0.5,0.5])*stamp_Size
	else:
		centroid = np.random.rand(2)*30

		while (centroid[0]<1 or centroid[0]>(stamp_Size-1) or centroid[1]<1 or centroid[1]>(stamp_Size-1) or (centroid[0]-15)**2+(centroid[1]-15)**2>((15)**2)*2):
			centroid = np.random.rand(2) * stamp_Size
		else:
			dict['centroid'] = centroid

	return dict	

''
area_values = np.array([3]) #np.linspace(1.5,4,10)
deviation = []
error = []

for j in range(len(area_values)):

	# First create the numpy array to hold the images, the number of sheets is number of realizations

	pixel_size = 30
	numb_real = 1
	images = np.zeros([pixel_size,pixel_size,numb_real])
	images_flattened = np.zeros([ np.size(images)/numb_real,numb_real])  

	# Create noise sheets by taking away the galaxy from the postage stamp
	for i in range(numb_real):
		
		#imageParams = modPro.default_ModelParameter_Dictionary()
		#image_with_noise, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		#image_without_noise, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		images[:,:,i] = (0.025/10000)*np.random.randint(0, high = 10000, size = (30,30))

	# Creates the second galaxy parameters

	Gal_Params = {}

	for i in range(numb_real):
		Gal_Params['Realization_'+str(i)] = {'Gal_0':get_Image_params(modPro.default_ModelParameter_Dictionary(), area_values[j], centered = 'y'),'Gal_1':get_Image_params(modPro.default_ModelParameter_Dictionary(), None)}
		#Gal_Params['Realization_'+str(i)]['Gal_0']["SB"]['e2'] = -.1 
		for k in range(len(Gal_Params['Realization_'+str(i)])):
			temp_image, disc = modPro.user_get_Pixelised_Model(Gal_Params['Realization_'+str(i)]['Gal_'+str(k)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
			images[:,:,i] +=temp_image	

        images_flattened[:,i] = images[:,:,i].flatten()

	
	deviation_to_add, error_to_add = (imMeas.combinded_min((area_values[j]),images_flattened, Gal_Params))#deviation.append(np.asscalar((imMeas.combinded_min((area_values[j]),images_flattened, Gal_Params)-area_values[j])))
	deviation.append(np.asscalar(deviation_to_add)-area_values[j])
	error.append(np.asscalar(error_to_add))

import matplotlib.pyplot as plt

'''
plt.errorbar(area_values,deviation, yerr = error, fmt = 'x', label = 'Deviation of primary galaxy area')

plt.axhline(0, color ='k')
plt.xlim([1,4.2])

plt.xlabel("Size of galaxy")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, combing the data set of 10000 realizations")
plt.show()
'''

import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(images[:,:0], interpolation = 'nearest')
pl.colorbar(im)

pl.show()



'''
for k in range(len(Gal_Params['Realization_0'])):

	temp_image, disc = modPro.user_get_Pixelised_Model(Gal_Params['Realization_0']['Gal_'+str(k)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
	model[:,:,0] +=temp_image	 



res = model - images

import pylab as pl

fig, (ax1,ax2,ax3) = pl.subplots(nrows=1, ncols=3)




im = ax1.imshow(res[:,:,0], interpolation = 'nearest')
ax1.set_title('Residuals')


ax2.imshow(images[:,:,0], interpolation = 'nearest')
ax2.set_title('Image')

ax3.imshow(model[:,:,0], interpolation = 'nearest')
ax3.set_title('Model')
pl.show()



''

area_values = np.linspace(1,5.5,1000)
likelihood = np.zeros_like(area_values)

# First create the numpy array to hold the images, the number of sheets is number of realizations

pixel_size = 30
numb_real = 10
images = np.zeros([pixel_size,pixel_size,numb_real])
images_flattened = np.zeros([np.size(images)/numb_real,numb_real])  
noise = np.zeros([pixel_size,pixel_size,numb_real])

# Create noise sheets by taking away the galaxy from the postage stamp



for j in range(len(area_values)):
	for i in range(numb_real):
		
		#imageParams = modPro.default_ModelParameter_Dictionary()
		#image_with_noise, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		#image_without_noise, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		images[:,:,i] = (0.025/10000)*np.random.randint(0, high = 10000, size = (30,30))

	# Creates the second galaxy parameters

	Gal_Params = {}

	for i in range(numb_real):
		Gal_Params['Realization_'+str(i)] = {'Gal_0':get_Image_params(modPro.default_ModelParameter_Dictionary(), area_values[j], centered = 'y'),'Gal_1':get_Image_params(modPro.default_ModelParameter_Dictionary(), None)}
		#Gal_Params['Realization_'+str(i)]['Gal_0']["SB"]['e2'] = -.1 
		for k in range(len(Gal_Params['Realization_'+str(i)])):
			temp_image, disc = modPro.user_get_Pixelised_Model(Gal_Params['Realization_'+str(i)]['Gal_'+str(k)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
			images[:,:,i] +=temp_image	

        images_flattened[:,i] = images[:,:,i].flatten()
	likelihood[j] = imMeas.combined_logLikelihood(area_values[j], images_flattened, Gal_Params, fitting = 'y')

import math

x = (likelihood[np.argmin(likelihood)])
print likelihood
likelihood =np.exp(x-likelihood)

print likelihood
from matplotlib.pyplot import *

plot(area_values,likelihood, color = 'k')
title('Likelihood against area value')
xlabel('Fitted area value')	
ylabel('Likelihood')


show()

def guassian(parameters,xdata,ydata):
	parameters[0]=const
	parameters[1]=mu
	parameters[2]=sigma
	parameters[3]=zero_error
	return np.sum(((const*np.exp((xdata-mu)**2/(2*sigma**2)) +zero_error - ydata))**2)

print area_values[np.argmax(likelihood)]


'''










