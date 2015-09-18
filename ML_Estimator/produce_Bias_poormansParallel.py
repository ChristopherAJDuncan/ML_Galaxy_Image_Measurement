#!/usr/bin/python
import numpy as np
import os
import time

SNRRange = [20., 36., 5.]
##This manually needs changed in both routines
outputDirectory = './ML_Output/SNRBias/28Aug2015/2D/e1_e2/xtol_minus5/NoLookup/Simplex/BiasCorrected/LowSNR/'

for SNR in np.arange(*SNRRange):
    command = './produce_Bias.py '+str(SNR)+' > '+outputDirectory+'/Log_'+str(SNR)+' &'
    print 'Running Bias:', SNR, ':', command
    os.system(command)
    print 'Running...'
    time.sleep(10)
