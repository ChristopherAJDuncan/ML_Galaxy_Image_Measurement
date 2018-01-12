
# coding: utf-8

import numpy as np;
import scipy;
import scipy.stats as scist;
import scipy.interpolate as sciInterp;
from scipy.stats import poisson, norm, uniform;
import pylab as pl


def pdf_conv(D1, D2):
    #Works for continous PDFs is both deifned over the same grid - edit to allow different grid, but same grid intervals
    int1 = D1.interval(.99999);
    int2 = D2.interval(.99999);
    supp = [min(int1[0], int2[0]), max(int1[1], int2[1])]
    grid = np.linspace(supp[0], supp[1], 1000)
    
    return grid, scipy.convolve(D1.pdf(grid), D2.pdf(grid), 'same')

def pdf_conv_dis_cont(D,C):
    #Puts discrete PMF onto continuous grid via interpolation: likely incorrect
    intD = D.interval(.9999);
    intC = C.interval(.9999);
    
    grid = np.linspace(intC[0],intC[1],1000)
    dgrid = np.arange(*intD)
    
    #Put discrete distribution onto same grid as output for easier convolution (Q: can this be done in the inverse)
    Dpmf = np.interp(grid, dgrid, D.pmf(dgrid))
    
    f = pl.figure(); ax = f.add_subplot(111)
    ax.plot(dgrid)
    
    return grid, scipy.convolve(C.pdf(grid), Dpmf, 'same')

def pdf_conv_dis_cont_full(D,C):
    #Works, likely slow
    intC = C.interval(.999999);
    intD = D.interval(.999999);
    grid = np.linspace(intD[0]+intC[0],intD[1]+intC[1],1000);
    
    conv = np.zeros(grid.shape[0])
    for i in range(grid.shape[0]):
        Dlow = max(intD[0],int(grid[i]+intC[0]))
        Dhi = min(intD[1], int(grid[i]+intC[1]+1))
            
        dGrid = np.arange(Dlow,Dhi)
         
        Dpmf = D.pmf(dGrid)
        
        #Convolve by summing all discrete non-zero entries
        toSum = Dpmf*C.pdf((grid[i]*np.ones(dGrid.shape[0]))-dGrid)
        conv[i] = toSum.sum()
        
    #Manually edit limits to zero
    conv[0] = 0.;
    conv[-1] = 0.;
        
    return grid, conv

def N_Likelihood(phot, spec):
    if "sigma" in spec:
        sig = spec["sigma"]
    elif "readout" in spec and "ADUf" in spec:
        sig = spec['readout']/spec['ADUf']
    else:
        raise ValueError("N_Likelihood: Sigma uncertainty not recognised in input specification dictionary")
    C = norm(phot, sig)

    nSig = 8.

    grid = np.linspace(phot-nSig*sig, phot+nSig*sig, 1000)

    return grid, C.pdf(grid)

def PN_Likelihood(phot, spec):
    #Returns p(counts|photons, spec), on a grid of counts
    D = poisson(phot*spec['qe']+spec['charge']);
    C = norm(0., spec['readout']/spec['ADUf']);
    
    return pdf_conv_dis_cont_full(D,C);

def inverse_Sample(xt,ft, nSamples = 1):
    from math import fabs

    #Construct CDF
    CDF = np.zeros(ft.shape[0])
    if(fabs(ft[0]) > 1e-10 or fabs(ft[-1]) > 1e-10):
        print "PDF at limits:", ft[0], " :: ", ft[-1]
        raise ValueError("Inverse Sampling: PDF must be zero at limits")
    CDF[0] = 0;
    for i in range(1,CDF.shape[0]):
        CDF[i] = CDF[i-1] + 0.5*(ft[i]+ft[i-1])*(xt[i]-xt[i-1])
    
    if(CDF[-1] < 0.99 or CDF[-1] > 1.01):
        raise ValueError("Inverse Sampling: CDF does not sum to accepted tolerance")
    CDF[-1] = 1 #NOTE: This needs edited out
    if(CDF[-1] != 1):
        print "CDF sums to ", CDF[-1]
        raise ValueError("Inverse sampling: CDF does not sum to one")
    
    #Invert CDF with interpolation
    inv_cdf = sciInterp.interp1d(CDF, xt)
    
    #Generate Random Number
    r = np.random.rand(nSamples)
    
    try:
        res = inv_cdf(r)
    except ValueError:
        print "Fatal Error in inverse sampling."
        print "CDF has ", CDF[0], CDF[-1]
        print "xt has ", xt[0], xt[-1]
        print "Random limits ", np.amin(r), np.amax(r)
        exit()

    #Sample
    return res

def add_PN_Noise(vals, readnoise, gain, sky, returnState = False, state = None):
    """
    Return the image (in photons) as observed through the telescope, adding noise processes
    :param vals:
    :param readout:
    :param gain:
    :param sky:
    :
    :return:
    """

    if state is not None:
        np.random.set_state(state)

    if returnState:
        state = np.random.RandomState()

    valsShape = vals.shape
    vals = vals.flatten()

    # Add sky background
    vals += float(sky)

    # Sample from poisson
    vals = np.random.poisson(vals).astype(np.float64)

    # Remove sky
    vals -= sky

    # Add read noise
    vals += np.random.normal(loc = 0, scale = readnoise, size = vals.shape[0])

    # Digitize and return to photons
    # 1000. is an offset to deal with rounding close to zero. 0.5 ensures rounds up in appropriate setting
    vals = gain * ( (1000. + vals/gain + 0.5).astype(np.int) - 1000.)

    # Reshape back into input
    res = vals.reshape(valsShape)

    if(returnState):
        res = [res, state]


    return res

def estimate_PN_noise(vals, readnoise, gain, sky):
    """
    Estimate the mean noise properties for the poisson-normal system
    :param vals:
    :param readout:
    :param gain:
    :param sky: Sky background *per pixel*
    :return:
    """

    vals[vals<0] = 0.
    # Poisson: sigma^2 = sky+vals. Normal: sigma = readout
    return sky+vals + (readnoise*readnoise)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enter in arguments as:')
    
    parser.add_argument('-PLam', type = float, help='A required float which gives the shape parameter for the Poisson distribution')
    parser.add_argument('-Nsi', type = float, help ='A required float which gives the width of the Gaussian')

    args = parser.parse_args()

    grid, conv = pdf_conv_dis_cont_full(poisson(args.PLam), norm(0., args.Nsi))
#Plot Convolved
    f = pl.figure(); ax = f.add_subplot(211)
    ax.plot(grid, conv)
    
#Interpolate
    sample = inverse_Sample(grid,conv,100000)
    print "Finished sampling", sample
    ax.hist(sample, bins = 100, normed = True)
    
    ax = f.add_subplot(212)
#Plot un-convolved
    dGrid = np.arange(0, int(grid[-1]+1))
    ax.plot(dGrid, poisson.pmf(dGrid, args.PLam))
    pl.show()

