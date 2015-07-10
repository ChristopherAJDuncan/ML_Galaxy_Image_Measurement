import src.model_Production as modPro
import src.measure_Bias as mBias
import src.image_measurement_ML as ML
import numpy as np

## Bias Measurement
S0 = 1.2; derLabel = 'T'
imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 50.
imageParams['size'] = 1.2
imageParams[derLabel] = S0
imageParams['stamp_size'] = np.array([50,50])
imageParams['centroid'] = (imageParams['stamp_size']+1)/2.

###Get image
image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G')

modelParams = imageParams.copy()

print 'modelParams test:', modelParams['stamp_size'], modelParams['noise']
print 'Estimated variance:', ML.estimate_Noise(image, maskCentroid = modelParams['centroid'])
raw_input('Check')

##Produce analytic bias
print 'Getting analytic bias:'
anbias = mBias.analytic_GaussianLikelihood_Bias(S0, derLabel, modelParams, diffType = 'ana')
print '\n ****** Analytic Bias is:', anbias

##Produce analytic bias
print 'Getting analytic bias:'
numanbias = mBias.analytic_GaussianLikelihood_Bias(S0, derLabel, modelParams, diffType = 'num')
print '\n ****** Numerical Analytic Bias is:', numanbias


##Produce fully numerical bias
bias = mBias.return_numerical_ML_Bias(S0, derLabel, modelParams)
print '\n ****** Fully Numerical Bias is found to be:', bias

