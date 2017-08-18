import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m


''

def get_Image_params(dict, size):
	stamp_Size = 30
	if size == 'no':
		Gal_size = np.random.normal(2,1)

		while Gal_size<2:
			Gal_size = np.random.normal(3,.25)
		else:
			dict['SB']["size"] = Gal_size
		

		Flux = np.random.normal(2,.25)

		while Flux<2:
			Flux = np.random.normal(5,1)
		else:
			dict["SB"]['flux']

		ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]

		while m.sqrt(ellipticity[0]**2+ellipticity[1]**2)>.71:
			ellipticity = [np.random.normal(0,.5),np.random.normal(0,.5)]
		else:
			dict["SB"]['e1'] = ellipticity[0]
			dict["SB"]['e2']

		centroid = np.random.rand(2)*30

		while (centroid[0]<1 or centroid[0]>(stamp_Size-1) or centroid[1]<1 or centroid[1]>(stamp_Size-1) or m.sqrt((centroid[0]-15)**2+(centroid[1]-15)**2)>15*m.sqrt(2)):
			centroid = np.random.rand(2) * stamp_Size
		else:
			dict['centroid'] = centroid

		return dict	
	else:
		Gal_size = np.random.normal(2,1)

		while Gal_size<2:
			Gal_size = np.random.normal(3,.25)
		else:
			dict['SB']["size"] = Gal_size
			return dict	


der = None 
numb_iter = 100

SNR = np.array([7,10,12.5,15,17.5,19,20,21,25,30,35,40,45]) # .5,1.0,1.5,2.0,2.5,3,3.5,4.,4.5,5.0
average_deviation = np.zeros(len(SNR))
extra_average_deviation = np.zeros(len(SNR))
standard_deviation = np.zeros(len(SNR))


PrimaryParams = modPro.default_ModelParameter_Dictionary()
PrimaryParams["SB"]['size'] = 2 # Has to equal a constan
PrimaryParams['SB']['e1'] = 0.0
PrimaryParams["SB"]['e2'] = 0.0
PrimaryParams["SB"]['flux'] = 10
PrimaryParams['stamp_size'] = [30,30]
PrimaryParams['centroid'] = (np.array(PrimaryParams['stamp_size'])+1)*0.5

# Creates extra galaxy (the one not being fitted)

imageParams = modPro.default_ModelParameter_Dictionary()

print "hello"
for j in range(len(SNR)):

	PrimaryParams["SB"]['size'] = SNR[j]
	iter_result = np.zeros(numb_iter)
	extra_iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):
		print i, j
		secondGal = {'Sec_Gal_1': get_Image_params(imageParams, 'no')}
		PrimaryParams = get_Image_params(PrimaryParams, None)
		extra_gal, disc = modPro.user_get_Pixelised_Model(secondGal['Sec_Gal_1'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image, disc = modPro.user_get_Pixelised_Model(PrimaryParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX,der = der)
		image = image + extra_gal
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size',), numbGalaxies = 2, second_gal_param = secondGal, secondFitParams = None)
		iter_result[i] = result[0][0]-PrimaryParams['SB']['size']
		#extra_iter_result[i] = result[0][1]-secondGal["Sec_Gal_1"]["SB"]['size']
		iter_error[i] = (1/result[1][0])

	average_deviation[j] = (np.mean(iter_result))
	#extra_average_deviation[j] = (np.mean(extra_iter_result))
	standard_deviation[j] = (1/(np.sum(iter_error)))


print average_deviation
print extra_average_deviation
print standard_deviation


import matplotlib.pyplot as plt

plt.errorbar(SNR,average_deviation, yerr = standard_deviation, fmt = 'x', label = 'Deviation of primary galaxy area')
#plt.scatter(SNR,extra_average_deviation, marker = 'x', label = 'Deviation of secondary galaxy area', color = 'k')
plt.axhline(0, color ='k')
plt.xlim([0,51])
plt.legend(loc = 'best')
plt.xlabel("SNR")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, second galaxy is randomly varied")
plt.show()
''
'''
#### Varying the radial positions of the two galaxies
der = None 
numb_iter = 10000
Off_centre_centriod = np.array([25,25])
position = [0.5,0.51,0.52,.53,.54,.55,.56,.57,.58,.59,.60,.61,.62,.63,.64,.65,.665,.680,0.70,0.75,0.8,0.85]
average_deviation = np.zeros(len(position))
standard_deviation = np.zeros(len(position))


fittingParams = modPro.default_ModelParameter_Dictionary()
fittingParams['SNR'] =20 # Has to equal a constan
fittingParams['SB']['e1'] = 0.0
fittingParams["SB"]['e2'] = 0.0
fittingParams["SB"]['size'] = 2
fittingParams["SB"]['flux'] = 10
fittingParams['stamp_size'] = [30,30]
fittingParams['centroid'] = (np.array(fittingParams["stamp_size"])+1)*0.5

image_temp, disc = modPro.user_get_Pixelised_Model(fittingParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

# Creates extra galaxy (the one not being fitted)

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]





for j in range(len(position)):

	imageParams['centroid'] = np.array([30*(position[j]),30*(position[j])])
	print position[j]
	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):
		extra_gal, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		
		image = image_temp + extra_gal
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)

		iter_result[i] = result[0][0]
		iter_error[i] = (1/result[1][0])

	average_deviation[j] = (np.mean(iter_result)-2)
	standard_deviation[j] = (1/(np.sum(iter_error)))

print average_deviation
print standard_deviation

import matplotlib.pyplot as plt

plt.errorbar(position,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.axhline(0, color ='k')
plt.xlim([.4,.9])
plt.xlabel("2nd galaxy position from centre")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, fitting second galaxy")
plt.show()

'''
'''
### Varying angular position of the second galaxy 

der = None
angular_position = np.zeros([8,2])
angle_to_x = [0,45,90,135,180,225,270,315]
for i in range(len(angular_position)):
	angular_position[i,0] = m.sin(i*(m.pi)/4) 
	angular_position[i,1] = m.cos(i*(m.pi)/4) 


numb_iter = 100

# Create on centre image

centre_gal = modPro.default_ModelParameter_Dictionary()
centre_gal['SNR'] =20 # Has to equal a constan
centre_gal['SB']['e1'] = .2
centre_gal["SB"]['e2'] = 0.0
centre_gal["SB"]['size'] = 2
centre_gal["SB"]['flux'] = 10
centre_gal['stamp_size'] = [30,30]
centre_gal['centroid'] = (np.array(centre_gal["stamp_size"])+1)*.5

image_temp, disc = modPro.user_get_Pixelised_Model(centre_gal, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

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

		image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image_temp, disc = modPro.user_get_Pixelised_Model(centre_gal, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image+image_temp
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)
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
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)
		iter_result[i] = result[0][0]

	average_deviation_medium[position] = (np.mean(iter_result)-2)
	standard_deviation_medium[position] = (np.std(iter_result))/numb_iter


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
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)
		iter_result[i] = result[0][0]

	average_deviation_far[position] = (np.mean(iter_result)-2)
	standard_deviation_far[position] = (np.std(iter_result))/numb_iter







from matplotlib.pyplot import *



errorbar(angle_to_x, average_deviation_close,yerr=standard_deviation_close, fmt = 'x', label = 'Close')
#errorbar(angle_to_x, average_deviation_medium,yerr=standard_deviation_medium, fmt = 'x', label = 'Medium')
#errorbar(angle_to_x, average_deviation_far,yerr=standard_deviation_far, fmt = 'x', label = 'Far')
xlabel("Angular postion of the second galaxy")
ylabel("Mean deviation of area from true value")
title("Bias in area fitting algorithm with second galaxy at different angular position")
legend(loc='best')
axhline(0, color = 'k')
show()
'''
'''
### Varying SNR

der = None 
numb_iter = 100000
Off_centre_centriod = np.array([25,25])
SNR = np.array([7,10,12.5,15,17.5,19,20,21,25,30,35,40,50]) #.5,1,1.5,
average_deviation = np.zeros(len(SNR))
standard_deviation = np.zeros(len(SNR))


fittingParams = modPro.default_ModelParameter_Dictionary()
fittingParams['SNR'] =20 # Has to equal a constan
fittingParams['SB']['e1'] = 0.0
fittingParams["SB"]['e2'] = 0.0
fittingParams["SB"]['flux'] = 10
fittingParams['stamp_size'] = [30,30]
fittingParams['centroid'] = (np.array(fittingParams["stamp_size"])+1)*0.5

# Creates extra galaxy (the one not being fitted)

imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] =20 # Has to equal a constan
imageParams['SB']['e1'] = 0.0
imageParams["SB"]['e2'] = 0.0
imageParams["SB"]['size'] = 2
imageParams["SB"]['flux'] = 10
imageParams['stamp_size'] = [30,30]
imageParams['centroid'] = Off_centre_centriod 

extra_gal, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)


for j in range(len(SNR)):

	fittingParams['SNR'] = SNR[j]
	print fittingParams['SNR']
	iter_result = np.zeros(numb_iter)
	iter_error = np.zeros(numb_iter)
	for i in range(numb_iter):
		image_temp, disc = modPro.user_get_Pixelised_Model(fittingParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
		image = image_temp + extra_gal
		image_flattened = image.flatten()
		result = imMeas.find_ML_Estimator(image_flattened, ('size'), numbGalaxies = 2, second_gal_param = imageParams)

		iter_result[i] = result[0][0]
		iter_error[i] = (1/result[1][0])**2

	average_deviation[j] = (np.mean(iter_result)-2)
	standard_deviation[j] = m.sqrt(1/(np.sum(iter_error)))

print average_deviation
print standard_deviation

import matplotlib.pyplot as plt

plt.errorbar(SNR,average_deviation,yerr=standard_deviation, fmt = 'x')
plt.xlabel("SNR")
plt.axhline(0, color = 'k')
plt.xlim([0,55])
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies (fitting both) with 100,000 iterations")
plt.show()
'''

### Unflatterening an image

'''
der = None

imageParams = modPro.default_ModelParameter_Dictionary()

image, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'g', outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)

import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()


image_flattened = image.flatten()

image = np.zeros([imageParams['stamp_size'][0], imageParams['stamp_size'][1]])


for i in range(imageParams['stamp_size'][0]):
	for j in range(imageParams['stamp_size'][1]):
		image[i,j] = image_flattened[j +i*imageParams['stamp_size'][0]]

import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(image, interpolation = 'nearest')
pl.colorbar(im)

pl.show()
'''


