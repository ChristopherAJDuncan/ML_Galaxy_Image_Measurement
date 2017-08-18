import init_Build
init_Build.init_build()

import python.model_Production as modPro
import python.surface_Brightness_Profiles as SBPro
import python.image_measurement_ML as imMeas
import numpy as np
import python.noiseDistributions as nDist
import math as m


der = None 



Sec_Gal_1 = modPro.default_ModelParameter_Dictionary()
Sec_Gal_1['centroid'] = (np.array(Sec_Gal_1['stamp_size'])+1)*0.5
Sec_Gal_1["SB"]['size'] = 4

Sec_Gal_2 = modPro.default_ModelParameter_Dictionary()
Sec_Gal_2['centroid'] = np.array([5,25])


secondGal = {'Sec_Gal_1':Sec_Gal_1, 'Sec_Gal_2':Sec_Gal_2}







		
image, disc = modPro.user_get_Pixelised_Model(secondGal['Sec_Gal_1'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
extra_gal, disc = modPro.user_get_Pixelised_Model(secondGal['Sec_Gal_2'], noiseType = None, outputImage = True, sbProfileFunc = SBPro.gaussian_SBProfile_CXX, der = der)
#image = image +extra_gal

image_flattened = image.flatten()

imMeas.postage_stamp_fitting(image_flattened, 2)
print Sec_Gal_1["SB"]['size']

