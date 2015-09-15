'''
Touch Date: 31 Aug 2015
Produces the statistics on simulated runs of produce bias. This has been implemented in produce bias itself, however is included here seperately as backup when the produce bias run does not finish
'''
import numpy as np

directory = '/disk1/cajd/Euclid/ML_Estimator/ML_Output/SNRBias/28Aug2015/2D/e1_e2/xtol_minus5/NoLookup/Simplex/BiasCorrected/'
filePrefix = 'e10p3_e20p3'
fileSuffix = '.dat'
doBC = [1,1]
SNRRange = [150.,201.,25.]
nPar = 2 ##This should be read in from input, but problem where it si just a list
inputValues = [0.3, 0.3]

print 'Input Values are:', inputValues, ' with filePrefix:', filePrefix, ' for nPar:', nPar
raw_input('Press <Enter> if this is ok...')

def intialise_Output(filename, mode = 'w', verbose = True):
    import os
    '''
    copied from produce_bias.py
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
                        
if(doBC[0]):
    handle1 = intialise_Output(directory+filePrefix+'_Statistics.dat', mode = 'a')
    handle1.write('## Recovered statistics as a result of bias run. Output of form [Mean, StD, Error on Mean] repeated for all fit quantities \n')
    
    handle2 = intialise_Output(directory+filePrefix+'_Bias.dat', mode = 'a')
    handle2.write('## Recovered statistics as a result of bias run. Output of form [Bias, StD, Error on Bias] repeated for all fit quantities \n')
    
    for SNR in np.arange(*SNRRange):
        Input = np.genfromtxt(directory+filePrefix+'_SNR'+str(SNR)+fileSuffix)
        
        Mean = Input.mean(axis = 0); StD = Input.std(axis = 0); MeanStD = StD/np.sqrt(Input.shape[0])
        
        
        if(nPar == 1):
            out = np.array([SNR, Mean, StD, MeanStD])
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([SNR, Mean-inputValues[0], StD, MeanStD])
            np.savetxt(handle2, out.reshape(1,out.shape[0]))
            
        else:
            out = np.array([ [SNR, Mean[l], StD[l], MeanStD[l]] for l in range(nPar) ]).flatten()
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([ [SNR, Mean[l]-inputValues[l], StD[l], MeanStD[l]] for l in range(nPar) ]).flatten()
            np.savetxt(handle2, out.reshape(1,out.shape[0]))
            
        print 'Finished SNR:', SNR, ' (mean, err, err on mean, %err)::', Mean, StD, MeanStD, MeanStD/(Mean-inputValues)

if(doBC[1]):
    print '\n Calculating Statisitcsw for bias corrected run'
    
    handle1 = intialise_Output(directory+filePrefix+'_Statistics_BC.dat', mode = 'a')
    handle1.write('## Recovered statistics as a result of bias run. Output of form [Mean, StD, Error on Mean] repeated for all fit quantities \n')
    
    handle2 = intialise_Output(directory+filePrefix+'_Bias_BC.dat', mode = 'a')
    handle2.write('## Recovered statistics as a result of bias run. Output of form [Bias, StD, Error on Bias] repeated for all fit quantities \n')
    
    for SNR in np.arange(*SNRRange):
        Input = np.genfromtxt(directory+filePrefix+'_SNR'+str(SNR)+'_BC'+fileSuffix)
        
        Mean = Input.mean(axis = 0); StD = Input.std(axis = 0); MeanStD = StD/np.sqrt(Input.shape[0])
        

        if(nPar == 1):
            out = np.array([SNR, Mean, StD, MeanStD])
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([SNR, Mean-inputValues[0], StD, MeanStD])
            np.savetxt(handle2, out.reshape(1,out.shape[0]))
            
        else:
            out = np.array([ [SNR, Mean[l], StD[l], MeanStD[l]] for l in range(nPar) ]).flatten()
            np.savetxt(handle1, out.reshape(1,out.shape[0]))
            
            out = np.array([ [SNR, Mean[l]-inputValues[l], StD[l], MeanStD[l]] for l in range(nPar) ]).flatten()
            np.savetxt(handle2, out.reshape(1,out.shape[0]))
            
        print 'Finished SNR:', SNR, ' (mean, err, err on mean)::', Mean, StD, MeanStD


print 'Finshed Normally'
        
