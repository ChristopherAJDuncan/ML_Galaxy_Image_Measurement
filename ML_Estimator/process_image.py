'''
Program to process an image according to the ML Estiamtor script. In its first incarnation, this will produce the image as well as process it
'''

import numpy as np
import src.image_measurement_ML as ML


def intialise_Output(filename, mode = 'w', verbose = True):
    import os
    '''
    Checks for directory existence and opens file for output.
    Modes are python default:
    --r : read-only (should most likely not be used with this routine
    --a : append
    --w : write

    verbose : If true, will output filename to screen
    '''

    
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    handle = open(filename, mode)

    if(verbose):
        print 'File will be output to: ',filename

    return handle
            


if __name__ == "__main__":

    print 'Running'

    handle = intialise_Output('./ML_Output/SNR_500./Test.dat', mode = 'a')

    for i in range(10000000):
        print 'Doing:', i

        imageShape = (50., 50.)
        imageParams = dict(size = 1.2, e1 = 0.3, e2 = 0., centroid = np.array(imageShape)/2, flux = 1.e5, \
                           magnification = 1., shear = [0., 0.], noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1.,\
                           modelType = 'gaussian')
        image, imageParams = ML.get_Pixelised_Model(imageParams, noiseType = 'G')

        print 'Finding ML'
        ML.find_ML_Estimator(image, fitParams = ['e1'],  outputHandle = handle, size = 1.2, e2 = 0., centroid = np.array(imageShape)/2, flux = 1.e5, magnification = 1., shear = [0., 0.], noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1., modelType = 'gaussian')
    
    print 'Finished Normally'

