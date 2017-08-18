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

## Import gems data

file = "Gems_data.csv"
out = open(file, 'rU')
Gems_data = csv.reader(out)
Gems_data = [row for row in Gems_data]
out.close()

# Puts selection critira on Gems data

to_del = []

for x in Gems_data[1:]:

	if (x[11] == '0') or (float(x[4])/float(x[3])>0.1) or (math.log(0.03*float(x[3]))>(-1.077+1.9*0.641) or math.log(0.03*float(x[3]))<(-1.077-1.9*0.641)):
		Gems_data.remove(x)

## Explaintion of variable 'Gems_data' ##
'''
Gems_data is a list of list. The first list indexes the galaxy the second list gives the value, the order is as follows

Gems Name [0] | F850 - GALFIT F850 band magnitude [1] | e_ - error in mag [2] |Re - GALFIT half light radius (pixels) [3] | e_ - error in radius (pixel) [4]| 
n - GALFIT Sersic index [5] | e_ - error in sersic index [6] |b/a - GALFIT ration of axis length [7]| e_ - error in b/a ratio [8] |PA - position angle [9] | 
f1 - flag constraint for fitting [10] | f2 - flag for science sample [11] | RAJ2000 - RA [12] | DEJ2000 [13]
NOTE in the variable Gems_data all the inputs are strings and the first row is coloumn titles. Eccentricity measures need to be changed to give shear values for model

'''

# This bit of code ranomly selects a galaxy from the Gems data and saves the data to a dictionary

def getRandData(catData, centered = 'n'):

	'''
	This function takes a catalogue and returns a dictionary using the parameters of galaxy in the entered catalogue. We apply a selection bias like that in the
	Hoekstra paper, i.e. only galaxies with 20<m<24.5 are used. 

	Requires:

	catData: the catalogue from which to take the data
	
	centered: if 'n' then the centriod is randomly assigned on the postage stamp (takes value from default dict). If 'y' the the centriod is on the centre of the postage stamp.
	If type(size) == scalar then the radial position is fixed and the angle is randomly varied. If centered == np array then this is passed as the centriod position

	'''
	gemsPixelSize = 0.03 # arcseconds
	euclidPixelSize = 0.1 #''
	galDict = modPro.default_ModelParameter_Dictionary()

	galData = random.choice(catData[1:]) # Don't want to select the titles

	
	galDict['SB']["flux"] = float(galData[1]) # saves the data making sure that is in the write sample	


	galDict['SB']["size"] = float(galData[3])*gemsPixelSize/euclidPixelSize # CHECK THAT WE CAN USE THIS AS A MEASURE OF SIZE



	# First we use the to get ellipticity we use the  Rayleigh dist
	sigma = 0.25

	galDict['SB']["e1"] = random.gauss(0,sigma)
	galDict['SB']["e2"] = random.gauss(0,sigma)

	while (galDict['SB']["e1"]**2+galDict['SB']["e2"]**2)>0.9**2:
		galDict['SB']["e1"] = random.gauss(0,sigma)
		galDict['SB']["e2"] = random.gauss(0,sigma)



	# We set the centriod of the galaxy

	if centered == 'y':
		pass # for default dict the centriod is centred on the centre of the postage stamp
	elif centered == 'n':
		galDict['centroid'][0] = random.random()*galDict['stamp_size'][0] # assumes uniform dist

		while galDict['centroid'][0]<1 or galDict['centroid'][0]>(galDict['stamp_size'][0]-1):
			galDict['centroid'][0] = random.random()*galDict['stamp_size'][0]	
		
		galDict['centroid'][1] = random.random()*galDict['stamp_size'][1]

		while galDict['centroid'][1]<1 or galDict['centroid'][1]>(galDict['stamp_size'][1]-1):
			galDict['centroid'][1] = random.random()*galDict['stamp_size'][1]

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




# Same as above, but only the size of target galaxy changes between iteration and each out put file is a different radial bin



Magnification = np.linspace(.94,1.05,12) #np.array([1.00,1.001,1.002,1.003])
MagParams = ('flux','size')
der = None



# First generate the dicts and noise

Numb_real = 2


Unlensed_dict = {} # Structure of GalDict is GalDict['Mag_'+str(Rad_position index)]['Realization_'+str(realization index)]['Gal_'+str(gal index)]

for j in range(Numb_real):
	Unlensed_dict['Realization_'+str(j)] = {'Gal_0':getRandData(Gems_data, centered = 'y'), 'Gal_1':getRandData(Gems_data,)}
print 'unlensed dict is', Unlensed_dict

GalDict = {}

for j in range(len(Magnification)):
	GalDict['Mag_'+str(j)] = Unlensed_dict



	
	GalDict['Mag_'+str(j)]['Realization_0']['Gal_0']['SB']["flux"] *= Magnification[j]
	GalDict['Mag_'+str(j)]['Realization_0']['Gal_1']['SB']["size"] *= Magnification[j]
	
	GalDict['Mag_'+str(j)]['Realization_1']['Gal_0']['SB']["flux"] *= Magnification[j]
	GalDict['Mag_'+str(j)]['Realization_1']['Gal_1']['SB']["size"] *= Magnification[j]		




# We create the noise data

Pixel_size = 30
Noise = np.zeros([Pixel_size,Pixel_size,Numb_real])

for i in range(Numb_real):
	Noise[:,:,i] = 0*0.025*np.random.randn(Pixel_size,Pixel_size)

# We create the input images

pixel_size = 30

Flattened_Images = {}

for i in range(len(Magnification)):
	Images = np.zeros([pixel_size,pixel_size,Numb_real])
	Images_flattened = np.zeros([ np.size(Images)/Numb_real,Numb_real]) 

	for j in range(Numb_real):
		for k in range(2):

			temp_image, disc = modPro.user_get_Pixelised_Model(GalDict['Mag_'+str(i)]['Realization_'+str(j)]['Gal_'+str(k)],noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = None)
			Images[:,:,j] +=temp_image

		Images[:,:,j] +=Noise[:,:,j]
		Images_flattened[:,j] = Images[:,:,j].flatten()






def finding_ML_estimator(magnificiation, imagesFlattened, unlensed_dict,noise, coreNumber, numbCores, numb_real,lensedDict):

	pixel_size = 30

	## Opens up csv file and gets it all ready

	file = "Data_"+str(coreNumber)+".csv"
	out = open(file, 'w')

	mywriter = csv.writer(out)
	mywriter.writerow(['Magnification','Measured Magnification', 'Deviation error'])


	for j in range(coreNumber,len(magnificiation), numbCores):


		print "Core " +str(coreNumber) +" is doing magnificiation " +str(Magnification[j])
		#print lensedDict['Mag_'+str(j)]
		#deviation_to_add = (imMeas.mag_min((Magnification[j]),imagesFlattened, unlensed_dict, ('flux','size')))
		#deviation_to_add = np.asscalar(deviation_to_add)-Magnification[j]
		#error_to_add = 0 
		print Unlensed_dict
		print imMeas.mag_likelihood(Magnification[j], ('flux','size',),imagesFlattened, Unlensed_dict)

		#mywriter.writerow([Magnification[j],deviation_to_add,error_to_add])

	
	out.close()

# Data writing code, each core saves the output to a different CSV file

Numb_cores = 1

Process_0 = mp.Process(target = finding_ML_estimator, args = (Magnification, Images_flattened, Unlensed_dict,Noise, 0, Numb_cores, Numb_real, GalDict))
#Process_1 = mp.Process(target = finding_ML_estimator, args = (Magnification, GalDict, Noise, 1, Numb_cores, Numb_real,))
#Process_2 = mp.Process(target = finding_ML_estimator, args = (Magnification, GalDict, Noise, 2, Numb_cores, Numb_real,))
#Process_3 = mp.Process(target = finding_ML_estimator, args = (Magnification, GalDict, Noise, 3, Numb_cores, Numb_real, ))


Process_0.start()
#Process_1.start()
#Process_2.start()
#Process_3.start()

print 'done'
