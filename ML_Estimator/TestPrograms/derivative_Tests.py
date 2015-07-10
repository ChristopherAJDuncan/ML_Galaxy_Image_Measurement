


#Single Run - Derivative
print 'Running'

S0 = 0.3; derLabel = 'e1'
imageParams = modPro.default_ModelParameter_Dictionary()
imageParams['SNR'] = 200.
imageParams[derLabel] = 2*S0
imageParams['stamp_size'] = [10,10]

###Get image
image, imageParams = modPro.get_Pixelised_Model(imageParams, noiseType = 'G')

##Reset noise value to accomodate easier comparison (note that noise on image, which must be measured, affects the derivative value)

diffIm = modPro.differentiate_Pixelised_Model_Numerical(imageParams, S0, derLabel, n = 1, interval = 0.1)[0]
#print 'Derivative of image is:'
#print diffIm.sum()

import math


modelParams = imageParams; modelParams[derLabel] = S0
model, modelParams = modPro.get_Pixelised_Model(modelParams)
print '---Model Test:', model.sum()
print '---Image Test:', image.sum()
print 'and resultant derivative of pixelised log-Likelihood is:'
modelParams['noise'] = imageParams['noise']
print 'Noise is:',  modelParams['noise']
diffpixlnL_analytic = ((model-image)*diffIm)/(math.pow(modelParams['noise'],2.))
print diffpixlnL_analytic.sum()


print 400*'----'
print ' Analytic lnL derivative check:'
print 'Image:', image.shape, image.sum()
print 'Model:', model.shape, model.sum()
print 'Difference:', (model-image).shape, (model-image).sum()
print 'Der Model:', diffIm.shape, diffIm.sum()
print 'Noise:', math.pow(modelParams['noise'],2.)
print 'Sum then divide:', ((model-image)*diffIm).sum(), ((model-image)*diffIm).sum()/math.pow(modelParams['noise'],2.)
print 400*'----'

from src.derivatives import finite_difference_derivative

print 'Derivative of pixelised ln_likelihood is:'
diffpixlnL = finite_difference_derivative(ML.get_logLikelihood, S0, args = [derLabel, image, modelParams], n = 1, dx = [0.001, 0.001])[0]
print diffpixlnL

print 'Ratio:', diffpixlnL/(diffpixlnL_analytic).sum()

