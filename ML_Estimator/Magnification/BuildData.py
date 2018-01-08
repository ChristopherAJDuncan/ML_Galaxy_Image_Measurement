"""
Author: D.D.
Touch Date: 26th September 2017
Purpose: To find bias is size measurements of primary galaxy when secondary galaxy is completely accounted for. The galaxies are sampled from
the GEMS catalogue. 
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
import os
import scipy.optimize as opt
import multiprocessing as mp
from scipy.stats import norm

import copy

Gems_data = None

# On 0.1'' pixel, 30 gal/arcmin, we need 110^2 to get a mean of 1 gal
stamp_size = np.array([110, 110])
# Used to keep most of galaxy in image - galaxy is always added within the stamp_
# size area
stamp_pad = np.array([40, 40])

euclidPixelSize = 0.1  # ''
ngal = 30  # per sq arcmin

# Puts selection critira on Gems data
to_del = []

## Explanation of variable 'Gems_data' ##
"""
Gems_data is a list of list. The first list indexes the galaxy the second list gives the value, the order is as follows

Gems Name [0] | F850 - GALFIT F850 band magnitude [1] | e_ - error in mag [2] |Re - GALFIT half light radius (pixels) [3] | e_ - error in radius (pixel) [4]| 
n - GALFIT Sersic index [5] | e_ - error in sersic index [6] |b/a - GALFIT ration of axis length [7]| e_ - error in b/a ratio [8] |PA - position angle [9] | 
f1 - flag constraint for fitting [10] | f2 - flag for science sample [11] | RAJ2000 - RA [12] | DEJ2000 [13]
NOTE in the variable Gems_data all the inputs are strings and the first row is coloumn titles. Eccentricity measures need to be changed to give shear values for model

"""
### Code allows for anaylsis of the GEMS data set

'''
# Some basic analysis of the gems data set	

eccen = []
size = []
x = np.linspace(-4,2)
for i in range(1,len(Gems_data)): # Starts from 1 as first row is titles
	eccen.append(float(Gems_data[i][1])) # entries in Gems_data are strings 0
	size.append(0.3*float(Gems_data[i][3]))


print len(size)
plt.hist2d(eccen, size,bins = 200, norm=colors.LogNorm())
plt.colorbar() 

#n, bins, patches = plt.hist(size, bins =100, normed = True,log = False, facecolor='green', alpha=0.75)
# = mlab.normpdf( bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#plt.xlim([-4,2])
plt.xlabel(r'Absolute magnitude', fontsize = 30)
plt.ylabel(r'Half light radius size (GALFIT) [px]}', fontsize = 30)
plt.title(r'Galaxy parameters showing cuts}', fontsize = 30)
plt.ylim(0,50)
plt.axvline(x=20, color = 'k', linewidth = 3.5, linestyle = 'dashed', label = r'\LARGE{Magnitude cut}')
plt.axvline(x=25, color = 'k', linewidth = 3.5, linestyle = 'dashed')
plt.axhline(y=1, color = 'g', linewidth = 3.5, linestyle = 'dashed')
plt.axhline(y=11.51, color = 'g', linewidth = 3.5, linestyle = 'dashed',  label = r'\LARGE{Size cut}')
plt.tick_params(labelsize=25)
plt.legend(loc= 'best', fontsize = 35)



plt.show()
'''


### Defines function that randomly selects data from the GEMS catalogue. ###

def getRandData(catData, size=None, centered='n'):
    """
	This function takes a catalogue and returns a dictionary using the parameters of galaxy in the entered catalogue. We apply a selection bias like that in the
	Hoekstra paper, i.e. only galaxies with 20<m<24.5 are used. 

	Requires
	--------

	catData: the catalogue from which to take the data with the same headings as defined above and with headings in the 1st row. (np array)
	size: Can set the size of the galaxy indep of the catalogue is set to a float. (float)
	centered: if 'n' then the centriod is randomly assigned on the postage stamp (takes value from default dict). If 'y' the the centriod is on the centre of the postage stamp.
	If type(size) == scalar then the radial position is fixed and the angle is randomly varied. If centered == np array then this is passed as the centriod position
	
	Returns
        A dictionary of the type used in the ML estimator implementation describing a randomly sampled galaxy
	-------

	galDict: A dictionary in the standard form with all parameters randomly sampled from the GEMS catalogue
	"""
    gemsPixelSize = 0.03  # arcseconds, this is to rescale the dist to be 'Euclid-like'.

    galDict = modPro.default_ModelParameter_Dictionary()
    galDict["PSF"]["PSF_Type"] = 2  # 2: Airy
    galDict["PSF"]["PSF_Airy_lambda"] = 8.e-7
    galDict["PSF"]["PSF_Airy_diameter"] = 1.2
    galDict["PSF"]["PSF_Airy_obscuration"] = 0.3
    galDict["SB"]["modelType"] = "disk"

    galDict["stamp_size"] = stamp_size + stamp_pad

    galData = random.choice(catData[1:])  # Randomly selects a row from the catalogue, ignores the headings.

    while (float(galData[1]) < 20 and float(galData[1]) > 24.5):  # Puts in selection critira on images used
        # Keep sampling until you select a source with the right properties
        galData = random.choice(catData[1:])

    # Set flux by sample of galaxy magnitude
    mag = float(galData[1])
    galDict["SB"]["flux"] = 10 ** (12.881 - (0.4 * mag))

    # Sample sizes
    if size == None:
        # If size is not set by the user
        galDict['SB']["size"] = float(galData[3]) * gemsPixelSize / euclidPixelSize
    else:
        # If size if set by the user
        if type(size) != float:
            raise TypeError('the galaxy size must be a float, it is ' + str(type(size)))
        else:
            galDict['SB']["size"] = size  ## save value in the dictionary

    ## We use the to get ellipticity we use the  Rayleigh dist. Sigma here is the *per-component* ellipticity
    sigma = 0.25

    galDict['SB']["e1"] = random.gauss(0, sigma)
    galDict['SB']["e2"] = random.gauss(0, sigma)

    while (galDict['SB']["e1"] ** 2 + galDict['SB']["e2"] ** 2) > 0.9 ** 2:
        galDict['SB']["e1"] = random.gauss(0, sigma)
        galDict['SB']["e2"] = random.gauss(0, sigma)

    ## We set the centriod of the galaxy

    if centered == 'y':
        pass  # for default dict the centriod is centred on the centre of the postage stamp
    elif centered == 'n':
        rand = random.random()
        galDict['centroid'][0] = stamp_pad[0] / 2 + rand * stamp_size[0]  # assumes uniform dist

        if (galDict['centroid'][0] > stamp_size[0]+stamp_pad[0]):
            print "Galaxy went too far over in x:", rand, stamp_pad[0] / 2, stamp_size[0], galDict['centroid'][0]

        rand = random.random()
        galDict['centroid'][1] = stamp_pad[1] / 2 + rand * stamp_size[1]

        if (galDict['centroid'][1] > stamp_size[1]+stamp_pad[1]):
            print "Galaxy went too far over in x:", rand, stamp_pad[1] / 2, stamp_size[1], galDict['centroid'][1]


    # Set by radial distance
    elif type(centered) == float or type(centered) == np.float64:

        if centered > (0.5 * galDict['stamp_size'][0]):
            raise ValueError('The entered radial position is off the stamp')

        angle = random.random() * (math.pi) * 2
        galDict['centroid'] += centered * np.array([math.cos(angle), math.sin(angle)])

    elif type(centered) == np.ndarray:
        galDict['centroid'] = centered
    else:
        raise ValueError('The entered value for centered = ' + str(centered) + ' is invalid')

    return galDict


### Defines the function that is targeted by the multipocessing function ###

def finding_ML_estimator(area_values, galDict, noise, coreNumber, numbCores, numb_real):
    """
	This function is targeted by the multipocessing routine. It splits up the work by having each radial bin done by a different core.
	The out put is sent to a csv file called 'Data_'+str(corenumber), where the core number is the core id. Analysis should be done with
	'CSV_reading.py'

	Requires 
	--------

	area_values: The set of different sizes of the primary galaxy. (numpy array)
	galDict: The dictionary of all the galaxies in the form defined above. (Dict)
	noise: The common noise array for all the different set-ups. (3D numpy array)
	coreNumber: An interger from 0 up. This names the core so the work can be split up. (int) 
	numbCores: The number of cores that want to be used, should be the same as the max coreNumber used. (int)
	numb_real: The number of realizations used for each magnification field and radial displacement. (int)

	Returns
	-------

	numbCores number of CSV files that should be analysed with 'CSV_reading.py' (note to select the right number of cores that were used).
	"""

    pixel_size = 30  # Postage stamp size in px, assumed the stamp is square.

    ## Opens up csv file and gets it all ready

    file = "Data_" + str(coreNumber) + ".csv"
    out = open(file, 'w')

    mywriter = csv.writer(out)
    mywriter.writerow(['Size', 'Measured Size', 'Deviation error'])

    for j in range(len(area_values)):  # loops through all the area values

        print "Core " + str(coreNumber) + " is doing area " + str(area_values[j])  # Prints progress update

        ## First create the numpy array to hold the images, the number of sheets is number of realizations

        images = np.zeros([pixel_size, pixel_size, numb_real])
        images_flattened = np.zeros([np.size(images) / numb_real, numb_real])

        ## Changes the size of the primary galaxy

        for i in range(numb_real):
            galDict['Radial_position_' + str(coreNumber)]['Realization_' + str(i)]['Gal_0']['SB']["size"] = area_values[
                j]

            for k in range(len(galDict['Radial_position_' + str(coreNumber)][
                                           'Realization_' + str(i)])):  # This is the bit where each core seperates
                temp_image, disc = modPro.user_get_Pixelised_Model(
                    galDict['Radial_position_' + str(coreNumber)]['Realization_' + str(i)]['Gal_' + str(k)],
                    noiseType=None, outputImage=True, sbProfileFunc=SBPro.gaussian_SBProfile_CXX, der=der)
                images[:, :, i] += temp_image

            images[:, :, i] += noise[:, :, i]
        images_flattened[:, i] = images[:, :, i].flatten()

        ## Image analysis using imMeas script.
        deviation_to_add, error_to_add = (
            imMeas.combinded_min((area_values[j]), images_flattened, galDict['Radial_position_' + str(coreNumber)]))

        deviation_to_add = np.asscalar(deviation_to_add) - area_values[j]
        error_to_add = np.asscalar(error_to_add)

        mywriter.writerow([area_values[j], deviation_to_add, error_to_add])  # Saves the analysis to the CSV file

    out.close()


def magnify_Catalogue(mu, catalogue):

    #for galaxy in catalogue:
    galaxy = catalogue["SB"]
    #galaxy["size"] = np.sqrt(mu)*galaxy["size"]
    galaxy["flux"] = mu*galaxy["flux"]

    return catalogue

def buildData(nGalaxy = 10000, magnification = 1., directory = ""):
    ### Defines the common variables to all cores ###
    import mypylib.utils.io as io
    from collections import OrderedDict

    ngalim = np.prod(stamp_size * euclidPixelSize / 60.) * ngal
    print "Average number per image is :", ngalim

    Numb_real = nGalaxy

    ## Import gems data
    file = "Gems_data.csv"
    out = open(file, 'rU')
    global Gems_data
    Gems_data = csv.reader(out)
    Gems_data = [row for row in Gems_data]
    out.close()

    for x in Gems_data[1:]:
        if (x[11] == '0'):  # or (float(x[4])/float(x[3])>0.1): #or (math.log(0.03*float(x[3]))>(-1.077+1.9*0.641) or math.log(0.03*float(x[3]))<(-1.077-1.9*0.641)):
            Gems_data.remove(x)

    catalogue = {"ngalim":ngalim}
    data = OrderedDict()
    lensed = OrderedDict()
    iterOutput = 100000

    strMu = str(magnification)

    catFilename = os.path.join(directory,"GEMS_sampled_catalogue_unlensed_"+strMu+".h5")
    lensedFilename = os.path.join(directory,"GEMS_sampled_catalogue_lensed_" + strMu + ".h5")
    dataFilename = os.path.join(directory,"GEMS_sampled_data_lensed_" + strMu + ".h5")

    for i in range(Numb_real):
        catalogue[str(i)] = getRandData(Gems_data)
        lensed[str(i)] = magnify_Catalogue(magnification, copy.deepcopy(catalogue[str(i)]))
        # Produce the image
        data[str(i)], disc = modPro.get_Pixelised_Model(lensed[str(i)], noiseType=None, Verbose=False,
                                                        outputImage=False, sbProfileFunc=None)

        ## Output as a function of iteration
        if i == Numb_real-1 or i%iterOutput == 0:
            catalogue["nGal"] = i
            print "Saving to: ", catFilename
            io.save_dict_to_hdf5(catFilename, catalogue)
            print "Saved successfully"

            #data["nGal"] = i #This screws up MEF output, required for h5 readin of data (could use catalogue info)
            io.save_dict_to_hdf5(dataFilename, copy.deepcopy(data))



            lensed["nGal"] = i
            io.save_dict_to_hdf5(lensedFilename, lensed)


    # Final output of MEF data version: not sure why this doesn't work with interim: could it be pointer
    # issues with data dictionary?
    dataFilename = os.path.splitext(dataFilename)[0] + ".fits"
    print "Calling MEF2", dataFilename
    io.output_images_to_MEF(dataFilename
                             , data.values())
    print "Called MEF2"

    print "Finished Building Data. Output files to: "
    print catFilename
    print dataFilename
    print lensedFilename
    print " "
    #raw_input("Check")

def CFHTLens_SizeMagnitude_Alsing2014(mag, lnsize):
    """
    Returns the Alsing 2014 fit to the CFHTLenS data, without a selection function.
    :param mag:
    :param lnsize:
    :return:
    """

    from mypylib.stats.distributions import normal_distribution, schecter_distribution

    # Parameterisation.
    a1 = -0.368
    a2 = 6.96
    b1 = 0.0513
    b2 = -0.8
    alpha = -1.174
    mstar = 22.511

    # Construct the mag-dependent lnSize mean and variance
    mLam = a1*mag + a2
    sigLam = b1*mag + b2

    return (normal_distribution(lnsize, mean = mLam, sigma = sigLam)*
            schecter_distribution(mag, xstar = mstar, alpha = alpha))


"""
exit()


der = None
areas = np.array([2, 2.71, 4.48, 8.00, 11.47])  # Different areas of the primary galaxy
Rad_position = [3.0, 6.0, 10.0, 14.0]  # Different radial seperations of the primary and secondary galaxy


# Everything after this point should be analysis:
# I.e. read in data, combine in a specific way, analyse

Pixel_size = 30
Noise = np.zeros([Pixel_size, Pixel_size, Numb_real])

for i in range(Numb_real):
    Noise[:, :, i] = 0.025 * np.random.randn(Pixel_size, Pixel_size)

### Section that starts the multiprocessing and defines the processes ###

Numb_cores = 4

Process_0 = mp.Process(target=finding_ML_estimator, args=(areas, GalDict, Noise, 0, Numb_cores, Numb_real,))
Process_1 = mp.Process(target=finding_ML_estimator, args=(areas, GalDict, Noise, 1, Numb_cores, Numb_real,))
Process_2 = mp.Process(target=finding_ML_estimator, args=(areas, GalDict, Noise, 2, Numb_cores, Numb_real,))
Process_3 = mp.Process(target=finding_ML_estimator, args=(areas, GalDict, Noise, 3, Numb_cores, Numb_real,))

Process_0.start()
Process_1.start()
Process_2.start()
Process_3.start()

print 'done'
"""