Author:: cajd (Christopher Duncan, c.a.j.duncan[{at}]gmail.com)

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
Error is [nPar] list of parameter uncertainties, estimated from the default method.



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

The code allows for a number of different minimisation methods to be used, and these have been tested to various extents. The most accurate has been shown to be the simplex method after a course grid search. Powell works well even if the initial guess was significantly different to the true value, but some residual bias was observed. Derivative methods such as cg of bfgs will find an accurate minimum, but are not quicker than simplex method for the 2-Parameter case. Further tests in the form of residual bias is needed for these methods.

In the gFit paper (http://arxiv.org/abs/1211.4847), the authors conclude that a cyclic coordinate descent (CCD) algorithm is the most accurate, but not necessarily the quickest. The gFit routine thus allows the use of CCD, LVM and MCMC. None of these are coded up in this version, however one can note that LVM is present in scipy as `leastsq', and there are excellent MCMC pre-fabs for python (e.g. emcee). The use of any of these methods are excellent possible extensions.

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

The source module `model_Production.py' contains the routines to produce the PSF-convolved, pixelised surface brightness profile used to fit to the image in find_ML_Estimator(). As such, many of the routine in this module are particularly important in the application of the result, however provided the model parameter dictionary is well-defined in the use of find_ML_Estimator, most of the work is done by these modules in the background.

As well as producing model images for the parameter fit, the routines can be used to produce noisy- or noise-free images. This can be acheived by two methods: The first is a `user-defined' method (that is, a method authored by cajd), which considers explicity the convolution of the underlying profile with the PSF and pixel-response-function using FFT, and GALSIM. In all applications including ML finding and analytic noise bias calcualtion, the former is used, and development has focussed on the use of that method. The use of GALSIM is therefore strongly discouraged, and the user is encouraged to take extra care whe using the GALSIM routines. GALSIM routines are known to give bad answers when derivatives of the SB profile are required.


MODEL PARAMETER DICTIONARY:

The model parameter dictionary used at all levels should follow the structure set out in
default_ModelParameter_Dictionary(**setters)
which corresponds to a constructor for the parameter data structure. When used, the free dictionary **setters can be used to override individual parameter values from their default value. Where parameters are contained witin a sub-dictionary, (e.g. the PSF or SB parameters), the code should take this into account provided they are given unique identifier labels. The return is a dictionary structure.

MODEL PRODUCTION:


--- user_get_Pixelised_Model(Params, inputImage = None, Verbose = False, noiseType = None, outputImage = False, sbProfileFunc = None, der = None, **sbFuncArgs)::
User defined pixelised model image production routine. Minimum input is model params and sbProfileFunc: In this case the image is noise free and is not the derivative.

sbProfileFunc must link to a function that returns the surface brightness profile across an xy grid, with call-sign sbProfileFunc(xy, cen,size,e1,e2,flux, der = ****, **sbFuncArgs), where cen ins the center of the model, and size, e1, e2 define the shape and size of the model image SB profile. In default this is not set, leading to an error, but all applications link the the C++ version of the Guassian SB profile in surface_Brightness_Profiles.py

noiseType supports `Gaussian` or `gaussian`, or None. If None (default), noise-free image is produced. If gaussian, gaussian noise is added by sampling from distribution with mean zero and std given in model params `noise'.

der is a list of parameter labels which specifies whether the returned image is the derivative. E.g. if der = [e1, e1, size] then the returend image is d^3(Im)/(de1 de2 dT).

--- get_Pixelised_Model_wrapFunction(x, Params, xKey, returnOrder = 1, **kwargs)
Wrapper routine for the above method, where the parameters labelled by xKey can be set to values x in the model parameter dictionary Params before call.

OTHER:

SNR_Mapping(model, var = None, SNR = None):: returns the pixel std (if SNR is entered) or SNR (if pixel variance is entered) for the model image `model', according to the filter-matched defintition used in GREAT08.

get_Model_Lookup(setParams, pLabel, pRange, dP, **modelFuncArgs):: Create an instance of model image lookup table.

return_Model_Lookup(lookup, P):: return the model lookup image for parameter values P (<list>). Return is the index of the parameter value in the lookup grid, and the model image itself.

__________________Development Considerations________________________________________________________________________________

EDITING DEFAULT MODEL PARAMETERS

The default model parameter dictionary is defined using sub-dictionaries SB and PSF for clarity. The helper routines such as set_Model_Parameter() account for such a subdivision, but in doing so the compare to a default model dictionary to identify parameters that can be edited without needing to refer the toe sub dictionary. E.g. as long as size is a unique label for the SB profile size, the size = 0.3 can be entered and the code will automatically identify that it belongs to sub-dictionary SB.

When adding/editing parameters, it is therefore important that **all parameters are uniquely labelled, irrespective of waht sub directory they exist in.** Also, seperate_Keys_byModel() should do the subdirectory seperation, but take care when editing to ensure that it behaves as expected. In particualr, if you add a new subdirectory, this routine will need edited to account for this.

CODING THE SB PROFILE

The background code which produces the SB profile should be done in C++ to limit runtime, and is implemented using Weave. To add a new SB profile, one must code in the derivative in C++ style as done in e.g. runWeave_GaussSB_de2dT. This must also be done and linked from the wrapper routine for each derivative combination condidered.

Running surface_Brightness_Profiles.py on terminal with arguements will allow for the C++ form of the SB profile to be output using SymPy symbolic maths. The arguements are then: size, e1, e2, flux, [der]. The output can then be coded up following the previoous examples e.g. runWeave_GaussSB_de2dT. 

This has been done for a Gaussian SB profile. For other profiles, gaussian_SBProfile_Sympy() can be altered to output derivatives, and the method used in gaussian_SBProfile_Sympy() should be follwed as an example.

CODING THE PSF

As the SB profile, the background is done in C++, and the first version contains the code for a Gaussian PSF model using the C++ version of the SB profile. An edit to other profiles requires an edit much like that described in `CODING THE SB PROFILE'

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


__________Possible Extensions________________________________________________________________________________________________

image_measurement_ML.py::

-- PRIOR ADDITION
Addition of a prior requires only edits to:
image_measurement_ML.py:: Addition of prior data structure construction routines
find_ML_Estimator():: Accept and pass prior
get_logLikelhood():: Accept prior. lnL -> lnPr + lnL
measure_Bias.py:: analytic_GaussianLikelihood_Bias():: lnL,i -> lnPr,i + lnL,i, and extend to full formalism.

__________Source Files________________________________________________________________________________________________________

Source files are located in directory `src/', and this contains all the necessary modules to run the maximum liklelihood estaimator, as well as a few extra features. Source modules include:
--- image_measurment_ML.py :: The main module, this contains routines which define the likelihood, and the method of obtaining the ML estimate, as well as error estimates and analytic bias correction.
--- measure_Bias.py :: Module containing the routines to calcualte the analytic first order noise bias correction.
--- model_Production.py :: Module containing routines which produce the model images which are fit to the input image, as well as defining model parameters and data structure.
--- PSF_Models.py :: Module containing routines for produceing a centered PSF model which can be convolved with the model image as part of the fitting routine, and its derviatives.
--- surface_Brightness_Profiles.py :: Routines for producing the model SB profile, and its derivatives up to 2nd order for model parameters.
--- derivatives.py :: Contains a routine for finite difference derivatives of a givne order, with basic automated convergence test.
--- generalManipulation.py :: Module containg helper routines for structure manipulation.


