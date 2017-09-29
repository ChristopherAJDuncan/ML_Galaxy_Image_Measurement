"""
Author: D.D.
Touch Date: 26th September 2017
Purpose: Reads in CSV files produced in 'Magnification_field.py' or 'Analysis_with_gems.py' and produces standard deviation graphs. The code is
in two sections. Section 1 - produces multiple plots of different bins, e.g. radial displacement. Section 2 - places all data points onto the 
same graph
"""

import csv 
from matplotlib.pyplot import *
import numpy as np
import math
from scipy.stats import linregress



numb_cores = 4 # This is the number of CSV files produced, each core makes its own file.  

### This section reads in the data from the files ###

DataDict = {}

for i in range(numb_cores):

	values  = np.zeros([5,3])
	file = "Data_" +str(i)+".csv"
	out = open(file, 'rU')
	data = csv.reader(out)
	data = [row for row in data]

	out.close()

	Data = np.empty([len(data)-1, 3])

	for j in range(len(data)-1):
		Data[j,0] = float(data[j+1][0])
		Data[j,1] = float(data[j+1][1])
		Data[j,2] = float(data[j+1][2])
		
	DataDict['Plot_'+str(i+1)] = Data
	
	if i ==0:
		Total_data = Data
	else:
		Total_data = np.vstack((Total_data,Data))
	

#### SECTION 1 ####

### Data analysis bit ###

Rad_position = [3.0,6.0,10.0,14.0] # Defines the different radial bins 
size = np.linspace(0,12) # Creates the size vector for the fit
Analysis = [] # Creates a list of all the anaylsis for each radial bin

for i in range(numb_cores):
	Analysis.append(linregress(DataDict['Plot_'+str(i+1)][:,0], DataDict['Plot_'+str(i+1)][:,1]))


print Analysis

### Plots up the data & trendlines ###

rc('text', usetex=True)
rc('font', family='serif')

for i in range(4):
	print DataDict['Plot_'+str(i+1)][:,2]

fig, AX = subplots(nrows =2, ncols = 2, sharex =True, sharey =True) # Sets up the subplots
i =1

for row in AX: # Adds the subplots with trend lines and the values of mu & c 
	for ax in row:
		mu = (Analysis[(i-1)][0])
		sigma = Analysis[(i-1)][4]
		ax.errorbar(DataDict['Plot_'+str(i)][:,0],DataDict['Plot_'+str(i)][:,1], fmt = 'x',yerr = DataDict['Plot_'+str(i)][:,2])
		ax.set_title(r'Radial displacement bin %.1f px, $ \mu=%.3f \pm %.3f$' %(Rad_position[i-1], mu, sigma,)) # \mu=%.3f \pm \sigma=%.3f$' %(mu, sigma)
		y = (mu)*size + Analysis[(i-1)][1]
		ax.plot(size, y, color ='k')
		ax.set_xlim([.90,1.1])
		ax.set_ylim([-.025,.025])


		i +=1

suptitle(r'True size vs fitted size}', fontsize = 20)

fig.text(0.5, 0.04, r'True size (GALFIT half light radius) [px]', ha='center', fontsize =18)
fig.text(0.04, 0.5, r'Fitted size (half light radius) [px]', va='center', rotation='vertical', fontsize =18)
show()


mu = []
err = []
for i in range(len(Analysis)): # Places the values of mu and the error for each radial bin into a list 
	mu.append(Analysis[i][0])
	err.append(Analysis[i][-1])

print 'hello ', Analysis[:][1]

errorbar(Rad_position, mu, yerr = err, fmt = 'x')
title(r'$\mu$ Vs radial displacement of secondary galaxy')
xlabel(r'Radial displacement of second galaxy from the centriod of the primary galaxy [px]')
ylabel(r'$\mu$')
show()

'''

#### SECTION 2 ####

### Data Anaylsis section ###

fit = linregress(Total_data[:,0], Total_data[:,1])

#for i in range(len(Total_data[:,0])):
#	Total_data[i,0] = math.log(Total_data[i,0])

Fitted_line =(Total_data[:,0]*fit[0]+fit[1]) # Creates the trendline

### Plotting section ### 

plt.errorbar(Total_data[:,0],Total_data[:,1], yerr = Total_data[:,2], fmt = 'x', label = 'Deviation of primary galaxy area', )
plt.plot(Total_data[:,0], Fitted_line )
plt.axhline(0, color ='k')
plt.xlim([.9,1.1])
plt.axhline()
plt.xlabel(r'$\mathrm{Magnification}$')
plt.ylabel(r'$\mathrm{Deviation\ from\ mean}$')
plt.title(r"$\mathrm{Bias\ in\ magnification\ fitting\ algorithm\ for\ two\ galaxies,\ combing\ the\ data\ set\ of\ } 10^4 \mathrm{realizations}$")
plt.show()

print (Fitted_line-Total_data[:,1])
'''

