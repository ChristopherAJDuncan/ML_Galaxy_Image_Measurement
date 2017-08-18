import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m
import scipy.optimize as opt
import multiprocessing as mp
import csv 
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





Gal_Params['Realization_0'] = {'Gal_0':get_Image_params(modPro.default_ModelParameter_Dictionary())}

image, other_thing =modPro.user_get_Pixelised_Model(Gal_Params['Realization_0']['Gal_0'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)




import pylab as pl
f = pl.figure()
ax = f.add_subplot(211)
im = ax.imshow(images, interpolation = 'nearest')
pl.colorbar(im)

pl.show()



areas = np.linspace(1.5,4,10)

'''

def finding_ML_estimator(area_values, coreNumber, numbCores):
	pass

	pixel_size = 30
	numb_real = 100000

	## Opens up csv file and gets it all ready

	file = "Data_"+str(coreNumber)+".csv"
	out = open(file, 'w')

	mywriter = csv.writer(out)
	mywriter.writerow(['Size','Deviation', 'Deviation error'])

	for j in range(coreNumber,(len(area_values)), numbCores):

		print "Core " +str(coreNumber) +" is doing area " +str(area_values[j])

		# First create the numpy array to hold the images, the number of sheets is number of realizations

		images = np.zeros([pixel_size,pixel_size,numb_real])
		images_flattened = np.zeros([ np.size(images)/numb_real,numb_real])  


		for i in range(numb_real):
			images[:,:,i] = 0.025*np.random.randn(30,30)

		# Creates the second galaxy parameters

		Gal_Params = {}

		for i in range(numb_real):
			Gal_Params['Realization_'+str(i)] = {'Gal_0':get_Image_params(modPro.default_ModelParameter_Dictionary(), area_values[j], centered = 'y'),'Gal_1':get_Image_params(modPro.default_ModelParameter_Dictionary(), None)}

			for k in range(len(Gal_Params['Realization_'+str(i)])):
				temp_image, disc = modPro.user_get_Pixelised_Model(Gal_Params['Realization_'+str(i)]['Gal_'+str(k)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
				images[:,:,i] +=temp_image	

	        images_flattened[:,i] = images[:,:,i].flatten()

		deviation_to_add, error_to_add = (imMeas.combinded_min((area_values[j]),images_flattened, Gal_Params))
		deviation_to_add = np.asscalar(deviation_to_add - area_values[j])
		error_to_add = np.asscalar(error_to_add)


		mywriter.writerow([area_values[j],deviation_to_add,error_to_add])

	
	out.close()



numb_cores = 3

Process_0 = mp.Process(target = finding_ML_estimator, args = (areas, 0, numb_cores,))
Process_1 = mp.Process(target = finding_ML_estimator, args = (areas, 1, numb_cores,))
Process_2 = mp.Process(target = finding_ML_estimator, args = (areas, 2, numb_cores,))
#Process_3 = mp.Process(target = finding_ML_estimator, args = (areas, 3, numb_cores,))


Process_0.start()
Process_1.start()
Process_2.start()
#Process_3.start()

print 'done'

## Plotting stuff

import matplotlib.pyplot as plt


plt.errorbar(areas,y_axis, yerr = yerrbar, fmt = 'x', label = 'Deviation of primary galaxy area')

plt.axhline(0, color ='k')
plt.xlim([1,4.2])

plt.xlabel("Size of galaxy")
plt.ylabel("Mean deviation from true value")
plt.title("Bias in area fitting algorithm for two galaxies, combing the data set of 10000 realizations")
plt.show()
'''