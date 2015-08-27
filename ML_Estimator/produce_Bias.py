#!/usr/bin/python 
'''
Program to process (create and analysis) multiple noise realisations of an image according to the ML Estiamtor script. 
'''

import numpy as np
import src.image_measurement_ML as ML
import src.model_Production as modPro
import src.surface_Brightness_Profiles as SBPro

Output = './ML_Output/SNRBias/10Aug2015/1DTests/Powell/e1/15x15/NOLookup/HighSNR/'
#'./ML_Output/SNRBias/9Jul2015/e1/15x15/HighRes/Lookup/ZeroInitialGuess/Powell/'
filePrefix = 'e10p3'
produce = [1,1] #Analytic, Sims

### Set-up


SNRRange = [150., 201., 50.] #Min, Max, Interval
minimiseMethod = 'Powell' #Acceptable are: simplex, powell, cg, ncg, bfgs, l_bfgs_b. See scipy documentation for discussion of these methods

##Input default values for parameters which will be fitted (this is used to set fitParams, so parameters to be fit must be entered here)
#fittedParameters = dict(e1 = 0.3, e2 = 0.2)
#initialGuess = dict(e1 = 0.0, e2 = 0.0)
fittedParameters = dict(e1 = 0.3)
initialGuess = dict(e1 = 0.25)
#fittedParameters = dict(size = 1.2) ##Edit to include all doen in fitParamsLabels etc.
#initialGuess = dict(size = 1.0)


fitParamsLabels = fittedParameters.keys(); fitParamsValues = fittedParameters.values()



##Initial Galaxy Set up
imageShape = (15., 15.) #size = 0.84853
imageParams = dict(size = 1.2, e1 = 0.0, e2 = 0.0, centroid = (np.array(imageShape)+1)/2, flux = 4.524, \
                   magnification = 1., shear = [0., 0.], noise = 10., SNR = 50., stamp_size = imageShape, pixel_scale = 1.,\
                   modelType = 'gaussian')

## Model Lookup Defintions - Overridden in the number of fitted parameters is greater than one
useLookup = False
## 1D Ellipticity lookup

lookupRange = [-0.99, 0.99]
lookupWidth = [0.001]
'''
##2D Ellipticity Lookup
lookupRange = [[-0.99, 0.99],[-0.99, 0.99]]
lookupWidth = [0.01,0.01]
'''

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


def bias_bySNR_analytic():
    '''
    Produces the analytic bias as a function of SNR.

    ToDo:
    Add this to bias_bySNR routine
    '''
    import src.measure_Bias as mBias
    import src.model_Production as modPro

    global imageParams
    imageParams.update(fittedParameters)

    handle = intialise_Output(Output+filePrefix+'_AnaBias.dat', mode = 'a')
    handle.write('## Recovered statistics as a result of bias run, single fit at a time, done analytically. Output of form [Bias] repeated for all fit quantities \n')
    for k in fitParamsLabels:
        handle.write('#'+str(k)+' = '+str(fittedParameters[k])+'\n')

    S = -1 #Counter
    filenames = []

    e = 0 ##Loop over this for multiple runs
    while True:
        print '\n'+400*'-'
        
        S += 1
        SNR = SNRRange[0] + S*SNRRange[2]

        ##Exit Condition
        if(SNR > SNRRange[1]):
            break

        ##Set Model
        imageParams['SNR'] = SNR

        ##Produce image to update noise to correct value (THIS IS A HACK AND NEEDS CHANGED) - estimate_Noise works fairly well, but you need to specify the noise correctly
        #disc, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G', outputImage = False)

        print 'Analytic Bias Check: For SNR', SNR, ' has noise var:', imageParams['noise']
        ## Check image to get SNR
        imageSB, imageParams = modPro.user_get_Pixelised_Model(imageParams, outputImage = True, sbProfileFunc = modPro.gaussian_SBProfile)
        imageParams['noise'] = modPro.SNR_Mapping(imageSB, SNR = SNR)

        bias = np.array(mBias.analytic_GaussianLikelihood_Bias(fitParamsValues, fitParamsLabels, imageParams, diffType = 'ana'))
        #bias = np.array([mBias.analytic_GaussianLikelihood_Bias(fitParamsValues[e], fitParamsLabels[e], imageParams, diffType = 'ana')])

        print '\n Analytic Bias for SNR:', SNR, ' is :', bias
        print ' '

        ### different to bias_bySNR
        print 'bias shape check:', bias.reshape(bias.shape[0], 1).shape, np.array(SNR).reshape(1,1).shape
        print ':', np.hstack((np.array(SNR).reshape(1,1),bias.reshape(1,bias.shape[0]))).shape
        np.savetxt(handle, np.hstack((np.array(SNR).reshape(1,1),bias.reshape(1,bias.shape[0]))))#.reshape(1,bias.shape[1]+1))

    handle.close()

def bias_bySNR():
    '''
    Run multiple realisations and produces SB profile estimates for each run. Set up for measurment of e1, should be generalised.
    To Do:
    --Generalise to multiple parameters: fitParams and setting defaults for thes fitParams, so that routine does not have any information on these parameters
    --Mean output: Header information - fitParams and input value
    '''
    print 'Producing Bias by SNR ratio'

    nRealisation = 10000000 ##This labels the maximum number of iterations
    percentError = 1

    global imageParams
    imageParams.update(fittedParameters)

    ##Get NoiseFree Image
    noiseFreeImage, disc = modPro.user_get_Pixelised_Model(imageParams, noiseType = None, sbProfileFunc = modPro.gaussian_SBProfile)

    modelLookup = None
    if(len(fitParamsLabels) <= 2 and useLookup):
        modelLookup =  modPro.get_Model_Lookup(imageParams, fitParamsLabels, lookupRange, lookupWidth, noiseType = None, sbProfileFunc = modPro.gaussian_SBProfile)
        print 'Created model lookup table'


    print 'Simulating Bias in SNR bins:\n'
    S = -1 #Counter
    filenames = []; SNRStore = []
    while True:
        S += 1
        SNR = SNRRange[0] + S*SNRRange[2]

        ##Exit Condition
        if(SNR > SNRRange[1]):
            break

        ##Set Model
        imageParams['SNR'] = SNR

        ## Store SNR for output
        SNRStore.append(SNR)

        ##Intialise output and set header
        filenames.append(Output+filePrefix+'_SNR'+str(SNR)+'.dat')
        handle = intialise_Output(filenames[S], mode = 'a')
        ##Write Header
        handle.write('# Bias Run Output. Following is input image parameters \n')
        for k in imageParams.keys():
            handle.write('#'+str(k)+' = '+str(imageParams[k])+'\n')

        MaxL = np.zeros((nRealisation, len(fitParamsLabels)))
        for real in range(nRealisation):
            ## This version uses GALSIM default
            #image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G')

            ## GALSIM with user-defined SB Profile
            #image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G', sbProfileFunc = modPro.gaussian_SBProfile)
            ## SYMPY - Very slow
            #modPro.get_Pixelised_Model_wrapFunction(0., imageParams, noiseType = 'G', outputImage = False, sbProfileFunc = SBPro.gaussian_SBProfile_Sympy)

            ## Entirely user-defined
            image, imageParams = modPro.user_get_Pixelised_Model(imageParams, noiseType = 'G', sbProfileFunc = modPro.gaussian_SBProfile, inputImage = noiseFreeImage)

            #MLEx = ML.find_ML_Estimator(image, modelLookup = None, fitParams = fittedParameters.keys(),  outputHandle = None, setParams = imageParams, e1 = 0.35) ##Needs edited to remove information on e1 (passed in for now) - This should only ever be set to the parameters being fit

            ##Find usign lookup table where appropriate
            MaxL[real,:] = ML.find_ML_Estimator(image, modelLookup = modelLookup, fitParams = fitParamsLabels,  outputHandle = handle, searchMethod = minimiseMethod, setParams = imageParams.copy(), **initialGuess) ##Needs edited to remove information on e1 (passed in for now) - This should only ever be set to the parameters being fit

            if(real > 10000 and real%1000 == 0 and percentError > 0.):
                Mean = (MaxL[:real,:].mean(axis = 0)-fitParamsValues); Err = MaxL[:real,:].std(axis = 0)/np.sqrt(real)
                if((np.absolute(100.*(Err/Mean)) < percentError).sum() == MaxL.shape[1]):
                    print '\n For SNR:', SNR, ' percentage error was reached in ', real, ' simulated images'
                    print 'With mean, std, %Err:', Mean, Err, np.absolute(100.*(Err/Mean))
                    break

            #print '----- Realisation:', real, ':: Ex:', MLEx, ' Look:', MLLook, ' :: Ratio:', MLEx/MLLook
            #raw_input('Check')

        handle.close()


    ### Construct and output mean, std and error on mean for each fit parameter
    handle1 = intialise_Output(Output+filePrefix+'_Statistics.dat', mode = 'a')
    handle1.write('## Recovered statistics as a result of bias run. Output of form [Mean, StD, Error on Mean] repeated for all fit quantities \n')
    for k in fittedParameters.keys():
        handle1.write('#'+str(k)+' = '+str(fittedParameters[k])+'\n')

    handle2 = intialise_Output(Output+filePrefix+'_Bias.dat', mode = 'a')
    handle2.write('## Recovered statistics as a result of bias run. Output of form [Bias, StD, Error on Bias] repeated for all fit quantities \n')
    for k in fittedParameters.keys():
        handle2.write('#'+str(k)+' = '+str(fittedParameters[k])+'\n')

    ##Produce Mean, StD from sampling and output
    for f in range(len(filenames)):
        Input = np.genfromtxt(filenames[f])

        SNR = SNRStore[f]
        Mean = Input.mean(axis = 0); StD = Input.std(axis = 0); MeanStD = StD/np.sqrt(Input.shape[0])

        if(len(fittedParameters.keys()) == 1):
            out = np.array([SNR, Mean, StD, MeanStD])
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([SNR, Mean-fittedParameters.values()[0], StD, MeanStD])
            np.savetxt(handle2, out.reshape(1,out.shape[0]))

        else:
            out = np.array([ [SNR, Mean[l], StD[l], MeanStD[l]] for l in range(len(fittedParameters.keys())) ]).flatten()
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([ [SNR, Mean[l]-fittedParameters.values()[l], StD[l], MeanStD[l]] for l in range(len(fittedParameters.keys())) ]).flatten()
            np.savetxt(handle2, out.reshape(1,out.shape[0]))


    print 'Finished SNR Bias loop without incident'


if __name__ == "__main__":

    # Bias by SNR run
    if(produce[0]):
        bias_bySNR_analytic()
    if(produce[1]):
        bias_bySNR()


    print 'Finished Normally'
    

    ''' #Single Run - ML Estimate
    print 'Running'

    handle = intialise_Output('./ML_Output/SNR_500./Test.dat', mode = 'a')

    for i in range(10000000):
        print 'Doing:', i

        image, imageParams = ML.get_Pixelised_Model(imageParams, noiseType = 'G')

        print 'Finding ML'
        ML.find_ML_Estimator(image, fitParams = ['e1'],  outputHandle = handle, initialParams = imageParams)
    
    print 'Finished Normally'
    '''
