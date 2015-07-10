'''
Reads in files listing the measurement of glaxy SB profile parameters in independent realisations of the galaxy image, and plots the mean and standard deviation
'''
import numpy as np

Files = ['ML_Output/SNR_500./Test.dat']

Input = np.genfromtxt(Files[0])

print 'Input', Input

print Input.mean(), Input.std(), Input.std()/np.sqrt(Input.shape[0])
