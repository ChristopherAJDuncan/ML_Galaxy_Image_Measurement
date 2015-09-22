This is the readme for the Maximum Likelihood fit optimal bias corrected shape/size estimator for application to galaxy images. The most common application of this program is the measurement of surface birghtness profile parameters assuming an underlying model profile: this can be done with the application of find_ML_Estimator() in image_measurement_ML.py, which also acts as a wrapper to the implementation fo further additional routines such as error estimation and noise bias correction. Other functionality includes the evaluation of the analytic noise bias using routines in measure_Bias.py, and image production (with Gaussian noise) in model_Production.py.


__________USE________________________________________________________________________________________________________________

---------------------- ML Estimation ----------------------------------------------------------------------------------------

** For further information about the struture and form of dummy variables and returns the reader is encouraged to reference the accompanying docstrings, through interactive python or the headers of the modules/routines themselves **

For all applications, the routines can be accessed by importing dir_to_prog/src/module_name.



BASIC APPLICATION

The ML estimation routine can be called suign a one line call by evaluating
-- find_ML_Estimator(image, fitParams)
where image is the image to be input, and fitParams is a list of the model parameters which are free to vary (and are being fit). 

This is the most basic way of calling the routine, and as such it makes some assumptions which limit its applicability: application of this form assumes:
-- The default model dictionary for input model parameters, inclduing intial guesses for parameters being fit.
-- No model lookup is used.
-- No presearch of parameter space to identify global minimum
-- No prior applied
-- Simplex minimisation method
-- No analytic noise bias correction
-- No run-time output

As such, it is therefore recommended that this option is not used unless steps are taken to ensure the result will be accurate, including:
-- Ensuring the default model parameter dictionary reflects the fixed model parameters, and a good initial estimate of the free model parameters.

Even if these steps are taken, the output cannot be assumed to be accurate due to the possibility of the routine isolating local minima, rather than global minima.

This method returns:
[ML, Error]
where:
ML is [nPar] list of maximum likelihood values for parameters entered in fitParams
Error is [nPar] list of parameter uncertainties, estiamted from the default method.



RECOMMENDED APPLICATION

It is recommended that the method is implemented as 
-- find_ML_Estimator(image, fitParams, setParams = ****,  searchMethod = 'simplex', preSearchMethod = 'brute', bruteRange = ***, biasCorrect = 1, error = 'Fisher')
where:
:: setParams is a model parameter dictionary which contains the fixed parameter values of the model image being fit
:: bruteRange specifies the parameter range used in an initial grid search of the likelihood in parameter space, as an [Npar,2] list (e.g. [[-0.99, 0.99], [-0.99, 0.99]] is an example when fitting e1 and e2). Note: the code can still search outside this region for the true minimum, especially if the grid search identifies extrema values of the bruteRange as the minima; this therefore *does not specify a prior range of compact support*. It is recommended that the entered bruteRange encompasses a large enough fraction of the viable parameter range to ensure that the global minima is encompassed.

This form of implementation:
-- Uses an intial grid search of specified parameter space to identify the global minimum, which is used to set an initial guess in the final minimisation routine
-- Outputs an estimate of the margnialised uncertainty of each parameter using the Fisher Matric formalism. This is quick, but assumes that the likelihood is Gaussian in parameter space.
-- Corrects the recovered ML estimate for first order noise bias. NOTE: To do so, the pixel variance MUST be accurately specified (this is unimportant for the minimisation routine as this corresponds to a global change in amplitude of the chi^2), as the analytic noise bias correction scales as pixel variance, or equivalently SNR. Iis therefore HIGHLY RECOMMENDED that the setParams['noise'] varaible is set before calling, and a noise estimate can be made using the estimate_Noise() routine.

This method returns:
[ML, Bias Corrected ML, Error]
where:
ML is [nPar] list of maximum likelihood values for parameters entered in fitParams.
Bias Corrected ML is [nPar] list of ML values, corrected for first order noise bias.
Error is [nPar] list of parameter uncertainties, estimated from the default method.


NOTE ON MINIMISATION METHODS



ADDITIONAL DUMMY VARIABLES

Any of the above methods can be run using additional dummy variables, described briefly here. For further detail, it is recommended to consider the docstring.

-- modelLookup:: Default None:: Dictionary describing the lookup table used (as defined in model_Production.py), which contains full copies of the model evaluated over a grid in parameter space. This is significantly faster for single parameter and marginally faster for double parameter fitting, but speed gain reduces when one considers high-dimensionality fits. In using this, one is trading off CPU time and runtime for higher memory usage. NOTE: For the 2 parameter case especially, one is limited in the resolution to which one can produce a lookup table before the required memeory store becomes too large. A bias in recovered parameter values has been observed to 10^-4 due to insufficiently sampled lookup table. This could be reduced by the use of linear interpolation across the image, however this so far has only been coded for the single parameter case: for the 2D case, the speed-up may no longer be worht it, particularly if one has to consider a large section of parameter space.

-- noiseCalc:: Default None:: Link to routine which estimates the pixel noise of the image. When None, the noise contained in the model dictionary is assumed to be true.

--outputHandle, bcoutputHandle:: Default None. Handle to the output file for the ML estmate, and bias corrected version. If handle is supplied, then result is output at runtime.

--**iParams: General unpacked dictionary, allows for model parameters to be set individually by passing into the end of the routine call. Useful when a preSearch method is not used to set initial guesses for the fit parameters.


---------------------- Noise Bias Estimation --------------------------------------------------------------------------------

The first order analytic noise bias can be calcualted directly, using two methods:

1. ---- Fully Analytic Method ----

The fully analytic method uses the fact that we have knowledge of the form of the likelihood (in image space at least), and knowledge of the statisitcs of the field (assumed to be Gaussian, with variance given by pixel noise, around mean given by the PSF-convolved pixelised surface brightness profile at that point) - See background and assumptions for theoretical underpinnings. 

This is the quickest method to calculate the noise bias, provided the derivative of the model wrt the free model parameters is known, and is the form applied in the bias correction in find_ML_Estimator().

BASIC APPLICATION

The most basic application is the also the RECOMMENDED application:

analytic_GaussianLikelihood_Bias(parameter_value, parameter_label, imageParams)

where parameter_value is a list of free parameter values around which the bias is to be evaluated (taken to be the ML point in application to ML finding method), parameter_labels is a list of strings which label the parameters (as defined in the model parameter dictionary), and imageParams is the model parameter dictionary where fixed parameter values are set.

Returns:
:: bias: list [nParameter] containing bias for each parameter entered.

2. ---- Numerical Method (Simulations) -----

This method makes no assumptions on the form of the likelihood, or on the statistics of the noise of the image (provided the routine get_logLikelihood() in image_measuremnt_ML.py encompasses this information), but instead considers the measurement of the log-Likelihood and its derivatives using a large number of simulated images (noise realisations) and finite difference derivatives.

Note that unlike typical measurement simulations, this does not need to reflect any underlyign distribution of source model parameters, as the aim is to evaluate the statisitcs of the field around the *true* value, which is input by the user.

This routine represents the more general application, and since it requires a large number of simulated images and logL evaluations, it should only ever be used for debugging purposes. One should alos be aware that there will be a level of noise due to the use of finite differencing (although since at least in the Gaussian case the analytic derivatives are known and coded, this could be easily by-passed). It is included for completeness, however the user should be wary of using it as development of this routine has lagged since the analytic form was completed. It is also only coded for a single parameter case: the extension to multi-parameter is trivial, and could mostly be copied from the analytic case once <K> etc have been derived for all parameters.

BASIC APPLICATION

return_numerical_ML_Bias(parameter_value, parameter_label, imageParams) with arguments as with the analytic case.

Return: as above.

---------------------- Model Image Production -------------------------------------------------------------------------------


__________Background and Assumptions_________________________________________________________________________________________


NOISE BIAS CALCULATION

Under the assumption that the likelihood is a guassian with chi^2 = -\frac{0.5}{\sigma^2}(I-Im(\beta))^2, and the pixel noise on the image is guassian with mean the PSF-convolved pixelised SB profile at that point, and standard deviation \sigma, it can be shown that the components entering into the first order noise bias can be given as:

F_{ij} = \frac{1}{n\sigma^2}\Sum_{pix} [Im,i*Im,j]^{pix}
K_{ijk} = -\frac{1}{n\sigma^2}\Sum_{pix} [Im,k*Im,ij + Im,i*Im,ik + Im,iIm,jk]^{pix}
J_{ijk} = \frac{1}{n^sigma^2}\Sum_{pix} [Im,j*Im,ik]^{pix}

where \sigma is pixel noise standard deviation, n is the number of pixels, Im(\beta) denotes the noise-free model which is being fit with parameters \beta, and ,i denotes the derivative wrt the parameter \beta_i. 

The noise bias on parameter \beta_s is then given by:

b_s = \frac{1}{n}(F^{-1})^{si}(F^{-1})^{jk}[0.5*K_{ijk} + J_{ijk}]

This is the form used in the analytic_GaussianLikelihood_Bias() routine, and the correction in find_ML_Estimator.


__________Source Files________________________________________________________________________________________________________

Source files are located in directory `src/', and this contains all the necessary modules to run the maximum liklelihood estaimator, as well as a few extra features. Source modules include:
--- image_measurment_ML.py :: The main module, this contains routines which define the likelihood, and the method of obtaining the ML estimate, as well as error estimates and analytic bias correction.
--- measure_Bias.py :: Module containing the routines to calcualte the analytic first order noise bias correction.
--- model_Production.py :: Module containing routines which produce the model images which are fit to the input image, as well as defining model parameters and data structure.
--- PSF_Models.py :: Module containing routines for produceing a centered PSF model which can be convolved with the model image as part of the fitting routine, and its derviatives.
--- surface_Brightness_Profiles.py :: Routines for producing the model SB profile, and its derivatives up to 2nd order for model parameters.
--- derivatives.py :: Contains a routine for finite difference derivatives of a givne order, with basic automated convergence test.
--- generalManipulation.py :: Module containg helper routines for structure manipulation.


