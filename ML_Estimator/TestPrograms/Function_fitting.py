import numpy as np
import math as m
import matplotlib.pyplot as mpl
from scipy.optimize import fmin


SNR = np.array([7,10,12.5,15,17.5,19,20,21,25,30,35,40,45])
Deviation = np.array([0.02379,0.01086,0.00598,0.00472,0.00380,0.00293,0.00223,0.00263,0.00149,0.00098,0.00095,0.00073,0.00036])

def power_law(parameters, x_data , y_data):
	pre_factor = parameters[0]
	power_factor = parameters[1]
	constant = parameters[2]

	out_put = np.zeros_like(y_data)

	for i in range(len(y_data)):
		out_put[i] = ((pre_factor*(x_data[i])**power_factor + constant)-y_data[i])**2 

	return np.sum(out_put)

parameters_dict = {'pre_factor': 2, 'power_factor': -1, 'constant': 0 }
x0 = [2, -1, 0] 
min_values = fmin(power_law,x0, args = (SNR, Deviation))

def power(parameters, x_value):

	pre_factor = parameters[0]
	power_factor = parameters[1]
	constant = parameters[2]	

	return (pre_factor*(x_value)**power_factor + constant)


print power(min_values, 20)

'''
continous_SNR = np.linspace(5,50)
y_to_plot = power(min_values, continous_SNR)

from matplotlib.pyplot import *

errorbar(SNR,Deviation, yerr= 0.0032, fmt = 'x', color = 'b', label = 'Deviation in area fitting')
axhline(0, color = 'k')
xlabel("SNR")
ylabel("Deviation of fitted area")
title("Effect of SNR on the deviation of fitted area (100,000 iterations)")
xlim([5,50])
plot(continous_SNR,y_to_plot, color = 'k', label = 'Power law fit')
legend()
show()
'''