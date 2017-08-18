import csv 
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import linregress

numb_cores = 4
'''
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
	
	#if i ==0:
	#	Total_data = Data
	#else:
	#	Total_data = np.vstack((Total_data,Data))

# Data analysis bit
Rad_position = [3.0,6.0,10.0,14.0]
size = np.linspace(0,12)
Analysis = []

for i in range(numb_cores):
	Analysis.append(linregress(DataDict['Plot_'+str(i+1)][:,0], DataDict['Plot_'+str(i+1)][:,1]))


print Analysis

# Plotting bit
from matplotlib.pyplot import *

rc('text', usetex=True)
rc('font', family='serif')


fig, AX = subplots(nrows =2, ncols = 2, sharex =True, sharey =True)
i =1

for row in AX:
	for ax in row:
		mu = (Analysis[(i-1)][0])
		sigma = Analysis[(i-1)][4]
		ax.errorbar(DataDict['Plot_'+str(i)][:,0],DataDict['Plot_'+str(i)][:,1], fmt = 'x',yerr = DataDict['Plot_'+str(i)][:,2])
		ax.set_title(r'Radial displacement bin %.1f px, $ \mu=%.3f \pm %.3f$' %(Rad_position[i-1], mu, sigma,)) # \mu=%.3f \pm \sigma=%.3f$' %(mu, sigma)
		y = (mu)*size + Analysis[(i-1)][1]
		ax.plot(size, y, color ='k')
		

		i +=1
suptitle(r'True size vs fitted size}', fontsize = 20)

fig.text(0.5, 0.04, r'True size (GALFIT half light radius) [px]', ha='center', fontsize =18)
fig.text(0.04, 0.5, r'Fitted size (half light radius) [px]', va='center', rotation='vertical', fontsize =18)

print DataDict

show()

mu = []
err = []
for i in range(len(Analysis)):
	mu.append(Analysis[i][0]-1)
	err.append(Analysis[i][-1])


errorbar(Rad_position, mu, yerr = err, fmt = 'x')
title(r'$\mu$ Vs radial displacement of secondary galaxy')
xlabel(r'Radial displacement of second galaxy from the centriod of the primary galaxy [px]')
ylabel(r'$\mu$')
show


'''
numb_cores = 4

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




#for i in range(len(Total_data[:,0])):
#	Total_data[i,0] = math.log(Total_data[i,0])

plt.errorbar(Total_data[:,0],Total_data[:,1], yerr = Total_data[:,2], fmt = 'x', label = 'Deviation of primary galaxy area', )

plt.axhline(0, color ='k')
plt.xlim([.3,1.5])

plt.xlabel(r'$\mathrm{Log\ of\ the\ size\ of\ the\ galaxy}$')
plt.ylabel(r'$\mathrm{Deviation\ from\ mean}$')
plt.title(r"$\mathrm{Bias\ in\ area\ fitting\ algorithm\ for\ two\ galaxies,\ combing\ the\ data\ set\ of\ } 10^4 \mathrm{realizations}$")
plt.show()

