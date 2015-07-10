import src.model_Production as modPro
import src.measure_Bias as mBias
import numpy as np

## Bias Measurement
S0 = 6.; derLabel = 'size'
imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 20.
imageParams[derLabel] = S0
imageParams['stamp_size'] = np.array([100,100])
imageParams['centroid'] = (imageParams['stamp_size']+1)/2.

###Get image
image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G')

modelParams = imageParams

print 'modelParams test:', modelParams['stamp_size']

##Produce fully numerical bias
#bias = mBias.return_numerical_ML_Bias(0.1, 'e1', modelParams)
#print 'Bias is found to be:', bias


##Produce analytic bias
print 'Getting analytic bias:'
anbias = mBias.analytic_GaussianLikelihood_Bias(S0, derLabel, modelParams, diffType = 'ana')
print 'Analytic Bias is:', anbias
