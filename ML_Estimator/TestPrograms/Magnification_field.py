"""
Author: D.D.
Touch Date: 26th September 2017
Purpose: To find bias in magnification measurements of primary galaxy when secondary galaxy is completely accounted for. The galaxies are sampled
from the GEMS catalogue. 
"""

import init_Build
init_Build.init_build()
import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import colors
import numpy as np
import random
import csv 
import math
import scipy.optimize as opt
import multiprocessing as mp
from scipy.stats import norm
from matplotlib.pyplot import *

### Imports the data from GEMS & applies cuts ###

## reads in the gems data CSV file

file = "Gems_data.csv"
out = open(file, 'rU')
Gems_data = csv.reader(out)
Gems_data = [row for row in Gems_data]
out.close()

# Puts selection critira on Gems data

for x in Gems_data[1:]: # Applies cuts to the gems catalogue

        # Qaulity cut, sigma_R/R cut (why?), Size cut upper, Size cut lower (sigma cuts...?) 
	if (x[11] == '0') or (float(x[4])/float(x[3])>0.1) or (math.log(0.03*float(x[3]))>(-1.077+1.9*0.641) or math.log(0.03*float(x[3]))<(-1.077-1.9*0.641)):
		Gems_data.remove(x)

## Explaintion of variable 'Gems_data' ##
'''
Gems_data is a list of list. The first list indexes the galaxy the second list gives the value, the order is as follows. The 1st row are headings

Gems Name [0] | F850 - GALFIT F850 band magnitude [1] | e_ - error in mag [2] |Re - GALFIT half light radius (pixels) [3] | e_ - error in radius (pixel) [4]| 
n - GALFIT Sersic index [5] | e_ - error in sersic index [6] |b/a - GALFIT ration of axis length [7]| e_ - error in b/a ratio [8] |PA - position angle [9] | 
f1 - flag constraint for fitting [10] | f2 - flag for science sample [11] | RAJ2000 - RA [12] | DEJ2000 [13]
NOTE in the variable Gems_data all the inputs are strings and the first row is coloumn titles. Eccentricity measures need to be changed to give shear values for model

'''

### Function that randomly selects galaxy from the Gems Cat ###



def getRandData(catData, centered = 'n'):

	'''
	This function takes a catalogue and returns a dictionary using the parameters of galaxy in the entered catalogue. We apply a selection bias like that in the
	Hoekstra paper, i.e. only galaxies with 20<m<24.5 are used. 

	Requires
	--------

	catData: the catalogue from which to take the data with the same headings as defined above and with headings in the 1st row. (np array)
	centered: if 'n' then the centriod is randomly assigned on the postage stamp (takes value from default dict). If 'y' the the centriod is on the centre of the postage stamp.
	If type(size) == scalar then the radial position is fixed and the angle is randomly varied. If centered == np array then this is passed as the centriod position
	
	Returns
	-------

	galDict: A dictionary in the standard form with all parameters randomly sampled from the GEMS catalogue
	'''
	gemsPixelSize = 0.03 # arcseconds, used to rescale the pixel size to make the image 'Euclid like'
	euclidPixelSize = 0.1 # arcseconds
	galDict = modPro.default_ModelParameter_Dictionary()

	galData = random.choice(catData[1:]) # Selects a random galaxy from the catalogue

	
	galDict['SB']["flux"] = float(galData[1]) # saves the data making sure that is in the write sample	


	galDict['SB']["size"] = float(galData[3])*gemsPixelSize/euclidPixelSize # Rescales the size so that it is 'Euclid-like'



	## First we use the to get ellipticity we use the  Rayleigh dist
	sigma = 0.25 # std of the distribution 

	galDict['SB']["e1"] = random.gauss(0,sigma)
	galDict['SB']["e2"] = random.gauss(0,sigma)

        #Resample for ellipticity which is too large
	while (galDict['SB']["e1"]**2+galDict['SB']["e2"]**2)>0.9**2:
		galDict['SB']["e1"] = random.gauss(0,sigma)
		galDict['SB']["e2"] = random.gauss(0,sigma)



	## We set the centriod of the galaxy
	if centered == 'y':
		pass # for default dict the centriod is centred on the centre of the postage stamp
	elif centered == 'n':
		galDict['centroid'][0] = random.random()*galDict['stamp_size'][0] # assumes uniform dist

                # Resample if they fall into a pixel boundary around the outside
		while galDict['centroid'][0]<1 or galDict['centroid'][0]>(galDict['stamp_size'][0]-1):
			galDict['centroid'][0] = random.random()*galDict['stamp_size'][0]	
		
		galDict['centroid'][1] = random.random()*galDict['stamp_size'][1]

                # Resample
		while galDict['centroid'][1]<1 or galDict['centroid'][1]>(galDict['stamp_size'][1]-1):
			galDict['centroid'][1] = random.random()*galDict['stamp_size'][1]

        # If centered is a float, then treat it as a radial separation
	elif type(centered) == float or type(centered) == np.float64:
		
		if centered>(0.5*galDict['stamp_size'][0]):
			raise ValueError('The entered radial position is off the stamp')

		angle = random.random()*(math.pi)*2
		galDict['centroid'] += centered*np.array([math.cos(angle), math.sin(angle)])

	elif type(centered) == np.ndarray:
		galDict['centroid'] = centered
	else:
		raise ValueError('The entered value for centered = ' +str(centered)+ ' is invalid')

	return galDict



### Function that sets the radial position of the second galaxy. ###	



def radial_Change(secondDict, radialPosition):
	"""
	A function that takes the dictionary of the secondary galaxy and places it at a set radial displacment and random angular position.

	Requires
	--------

	secondDict: The dictionary of the secondary galaxy to have it's radial position changed. (dict)
	radialPosition: The final radial displace in px from the centre of the postage stamp. (float)

	Returns:

	position_Second_Gal: The (x,y) coords of the secondary galaxy at a random angle and set radial displacement.

	"""
	## Check the radial position is on the postage stamp

	if radialPosition>secondDict['stamp_size'][0] or radialPosition<0 or radialPosition>secondDict['stamp_size'][1]:
		raise ValueError('The radial position is of the postage stamp')

	angle = random.random()*(math.pi)*2 # Angle is randomly selected

	return np.array([0.5*secondDict['stamp_size'][0] +radialPosition*math.cos(angle),0.5*secondDict['stamp_size'][1] +radialPosition*math.sin(angle)])

### Data anylsis function ###

def finding_ML_estimator(magnification, unLensedDict, radialPosition, fittingParams,coreNumber, numbCores, numb_real):
	"""
	This is the function that is target in the multipocessing routine. It splits the work up by each core doing a different radial
	displacement. It can also be changed so that each core does a set of the magnifcaition values. The output from each core is sent 
	to Data_coreNumber and can then be combinded using the 'CSV_reading.py' file.

	Requires
	--------

	magnification: This is a numpy array of all the magnification fields that want to be applied. (numpy array)
	unLesnsedDict: The diction giving the parameters of all the galaxies before the galaxy has been lensed. (dict)
	radialPosition: The radial distance between the centriod of the primary and secondary galaxy in pixels. (numpy array)
	fittingParams: A tuple containing either ('size',), ('flux',) or ('size','flux',). It contains the properties that 
	the magnification field should change. (tuple)
	coreNumber: An interger from 0 up. This names the core so the work can be split up. (int) 
	numbCores: The number of cores that want to be used, should be the same as the max coreNumber used. (int)
	numb_real: The number of realizations used for each magnification field and radial displacement. (int)

	Returns
	-------

	CSV files starting 'Data_' followed by a number in the dir. These should be analysed with 'CSV_reading.py'
	"""

	pixel_size = 30 # defines the pixel size

	## Opens up all the CSV files for each core to write to.

	file = "Data_"+str(coreNumber)+".csv"
	out = open(file, 'w')

	mywriter = csv.writer(out)
	mywriter.writerow(['Magnification','Measured Magnification', 'Deviation error'])

	## Data analysis part of the code

	for position in range(coreNumber,len(radialPosition), numbCores): # Each core does an element in position array, this for splits up the cores
			
		for secondary_Gal in range(len(unLensedDict)): # This core sets the radial postion of the second galaxy
			unLensedDict['Realization_'+str(secondary_Gal)]['Gal_1']['centroid'] = radial_Change(unLensedDict['Realization_'+str(secondary_Gal)]['Gal_1'], radialPosition[position])

		for j in range(len(magnification)): # This is the bit where each core will start to do its own magnification


			lensedDict = modPro.magnification_Field(unLensedDict,  fittingParams, mag = magnification[j]) # Magnificiation field is applied to the dict

			Image = np.zeros([Pixel_size,Pixel_size, Numb_real]) # 3D numpy array to hold all the images. Each sheet (3rd index) is a new image
			Images_flattened = np.zeros([Pixel_size**2, Numb_real]) # Creates a flattened array for the flattened images

			for i in range(Numb_real): # for loop creates and flattens the images
			    for k in range(len(lensedDict['Realization_'+str(i)])):

                                # Does this add multiple objects?
			        model_to_add, disc = modPro.user_get_Pixelised_Model(lensedDict['Realization_'+str(i)]['Gal_'+str(k)], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = None)
			        Image[:,:,i] += model_to_add 

			    Image[:,:,i] += Noise[:,:,i] 
			    Images_flattened[:,i] = Image[:,:,i].flatten()


			print "Core " +str(coreNumber) +" is doing magnification " +str(Magnification[j]) # Prints where the code is at


                        ## @@ What is this?
			deviation_to_add, error_to_add = (imMeas.mag_min(magnification[j],Images_flattened, unLensedDict, fittingParams)) # The images are analysed
			deviation_to_add = np.asscalar(deviation_to_add)-magnification[j] 
			

			mywriter.writerow([magnification[j],deviation_to_add,error_to_add]) # Results are saved to the CSV corresponding to the core

	out.close()

### Defining variables for all the core ###

## First generate the dicts and noise

Unlensed_dict = {}

for i in range(Numb_real): # This bit samples the number of galaxies on the postage stamp from a poisson distribution. 
	NumbGal = int(np.random.poisson(2,1))

	while NumbGal ==0:
		NumbGal = int(np.random.poisson(2,1))

	for j in range(NumbGal):
		Unlensed_dict['Realization_'+str(i)]['Gal_'+str(j)] = getRandData(Gems_data)

#for i in range(Numb_real):
#	Unlensed_dict['Realization_'+str(i)] = {'Gal_0':getRandData(Gems_data, centered = 'y'), 'Gal_1':getRandData(Gems_data,)}


Noise = np.zeros([Pixel_size,Pixel_size,Numb_real])

#Strange noise profile -> this should be a guassian
for i in range(Numb_real): # noise is a random variable
	Noise[:,:,i] = 0.8*np.random.randn(Pixel_size,Pixel_size)



Magnification = np.linspace(0.95,1.05,10) # Creates a numpy array of all the magnification values
Numb_real = 2 # number of realizations per magnificiation field
Pixel_size = 30 # postage stamp size in pixel (assumed square)
der = None 
Fitting_Params = ('size','flux',) # Tuple of parameters that the mag field will affect (could also be either 'flux' or 'size')
Radial_Position = np.array([3,6,10,14]) # The radial position between the primary and secondary galaxy in pixels


### Defines the process that the multiproccesing modulue will use ###


Numb_cores = 4 # Number of cores that are wanted to be used

## Calling the multiprocessing functions 

Process_0 = mp.Process(target = finding_ML_estimator, args = (Magnification, Unlensed_dict, Radial_Position, Fitting_Params, 0, Numb_cores, Numb_real,))
Process_1 = mp.Process(target = finding_ML_estimator, args = (Magnification, Unlensed_dict, Radial_Position, Fitting_Params, 1, Numb_cores, Numb_real,))
Process_2 = mp.Process(target = finding_ML_estimator, args = (Magnification, Unlensed_dict, Radial_Position, Fitting_Params, 2, Numb_cores, Numb_real,))
Process_3 = mp.Process(target = finding_ML_estimator, args = (Magnification, Unlensed_dict, Radial_Position, Fitting_Params, 3, Numb_cores, Numb_real,))


Process_0.start()
Process_1.start()
Process_2.start()
Process_3.start()

print 'done'
