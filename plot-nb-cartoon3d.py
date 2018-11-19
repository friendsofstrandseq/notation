#!/usr/bin/env python

from scipy import stats
from scipy.stats import nbinom
import numpy as np
import pandas as pd
 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.neighbors.kde import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

## create dictionary of breeds, with ((mean heigh, SD height),(mean weight, SD weight))
#breeddict = {}
#breeddict['Terrier'] = ((38.5,1.25),(7.95,0.575))
#breeddict['Melitan'] = ((29.2,0.65),(5.05,0.225))
#breeddict['Schnauzer'] = ((48.5,0.75),(6.6,0.8))
#breeddict['Wolfhound'] = ((54.6,1.1),(11.6,1.75))
#listBreeds = breeddict.keys()
 
## create a colour palette
#listColours = ['red','blue','green','orange']
 
## function to create random height & weight observations for a dog breed
## Because there is going to be a correlation between height & weight, we let that reflect in covariance
#np.random.seed(123)
#def dogobs(breed):
    #((muheight,sdheight),(muweight,sdweight)) = breeddict[breed]
    #cov1=np.random.randint(1,3)
    #cov2=np.random.randint(1,20)
    #observations = np.random.multivariate_normal([muheight, muweight], [[sdheight**2, cov1],[cov2, sdweight**2]], 30)
    #obsdf = pd.DataFrame(data=observations,columns=['height','weight'])
    #obsdf['breed'] = breed
    #return obsdf
 
## create a dataframe with 30 observations for each breed
#dogsdf = pd.DataFrame(columns=['height','weight','breed'])
#for breed in listBreeds:
    #obsdf = dogobs(breed)
    #dogsdf = dogsdf.append(obsdf)
 
## use KDE to get joint distribution across height, weight within each class
#KDESchnauzer = KernelDensity(kernel='gaussian', bandwidth=2).fit(dogsdf[dogsdf['breed']=='Schnauzer'][['height','weight']])
#KDEWolfhound = KernelDensity(kernel='gaussian', bandwidth=2).fit(dogsdf[dogsdf['breed']=='Wolfhound'][['height','weight']])
#KDETerrier = KernelDensity(kernel='gaussian', bandwidth=2).fit(dogsdf[dogsdf['breed']=='Terrier'][['height','weight']])
#KDEMelitan = KernelDensity(kernel='gaussian', bandwidth=2).fit(dogsdf[dogsdf['breed']=='Melitan'][['height','weight']])
 
## calculate p(height,weight | breed) for each point in our decision space
# define decision space
rangeHeight = np.array(range(1,18))
rangeWeight = np.array(range(1,18))
X, Y = np.meshgrid(rangeHeight, rangeWeight)

p = 0.4

n_cn0 = 0.3*(1-p)/p
n_cn1 = 3*(1-p)/p
n_cn2 = 6*(1-p)/p

# PLOT 1

Z =  nbinom.pmf(X, n_cn0, p) * nbinom.pmf(Y, n_cn0, p)
Z2 = nbinom.pmf(X, n_cn1, p) * nbinom.pmf(Y, n_cn1, p)
Z3 = nbinom.pmf(X, n_cn2, p) * nbinom.pmf(Y, n_cn2, p)

fig = plt.figure(figsize=(10,8), dpi=1600) # specifies the parameters of our graphs
plt.subplots_adjust(hspace=.5)            # extra space between plots
 
cm_red = plt.get_cmap('Reds')
#cm_grey = plt.get_cmap('gray')

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cm_grey = mpl.colors.ListedColormap(cmap[10:,:-1])

ax1 = plt.subplot2grid((2,2),(0,0), projection='3d')
#ax1.plot_surface(X, Y, Z, cmap=cm) #  rstride=5, cstride=5
ax1.plot_surface(X, Y, Z2, cmap=cm_red) #  rstride=5, cstride=5
#ax1.set_axis_off()
ax1.contourf(X, Y, Z, zdir='x', cmap=cm_grey, alpha=1, offset = -1) # offset=65
ax1.contourf(X, Y, Z, zdir='y', cmap=cm_grey, alpha=1, offset = -1) # offset=65
#ax1.contourf(X, Y, Z3, zdir='x', cmap=cm_grey, alpha=1, offset = -1) # offset=65
#ax1.contourf(X, Y, Z3, zdir='y', cmap=cm_grey, alpha=1, offset = -1) # offset=65
ax1.contourf(X, Y, Z2, zdir='x', cmap=cm_red, alpha=0.6, offset = -1) # offset=65
ax1.contourf(X, Y, Z2, zdir='y', cmap=cm_red, alpha=0.6, offset = -1) # offset=16
ax1.view_init(azim=240, elev=50)
ax1.set_zlim(0,0.025)
ax1.invert_xaxis()
ax1.invert_yaxis()
#plt.title("Joint Likelihood\n p(height,weight | breed = schnauzer)")

plt.savefig('test1.pdf') 

# PLOT 2

Z =  nbinom.pmf(X, n_cn0, p) * nbinom.pmf(Y, n_cn1, p)
Z2 = nbinom.pmf(X, n_cn1, p) * nbinom.pmf(Y, n_cn0, p)
Z3 = nbinom.pmf(X, n_cn2, p) * nbinom.pmf(Y, n_cn2, p)

fig = plt.figure(figsize=(10,8), dpi=1600) # specifies the parameters of our graphs
plt.subplots_adjust(hspace=.5)            # extra space between plots
 
#cm_blue = plt.get_cmap('Blues')
#cm_grey = plt.get_cmap('gray')
cmap = mpl.cm.Blues(np.linspace(0,1,20))
cm_blue = mpl.colors.ListedColormap(cmap[5:,:-1])

cmap = mpl.cm.Greys(np.linspace(0,1,20))
cm_grey = mpl.colors.ListedColormap(cmap[10:,:-1])

ax2 = plt.subplot2grid((2,2),(0,0), projection='3d')
#ax2.plot_surface(X, Y, Z, cmap=cm) #  rstride=5, cstride=5
ax2.plot_surface(X, Y, Z2, cmap=cm_blue) #  rstride=5, cstride=5
ax2.set_axis_off()
ax2.contourf(X, Y, Z, zdir='x', cmap=cm_grey, alpha=1, offset = -1) # offset=65
ax2.contourf(X, Y, Z, zdir='y', cmap=cm_grey, alpha=1, offset = -1) # offset=65
#ax2.contourf(X, Y, Z3, zdir='x', cmap=cm_grey, alpha=1, offset = -1) # offset=65
#ax2.contourf(X, Y, Z3, zdir='y', cmap=cm_grey, alpha=1, offset = -1) # offset=65
ax2.contourf(X, Y, Z2, zdir='x', cmap=cm_blue, alpha=0.6, offset = -1) # offset=65
ax2.contourf(X, Y, Z2, zdir='y', cmap=cm_blue, alpha=0.6, offset = -1) # offset=16
ax2.view_init(azim=240, elev=50)
ax2.set_zlim(0,0.025)
ax2.invert_xaxis()
ax2.invert_yaxis()
#plt.title("Joint Likelihood\n p(height,weight | breed = schnauzer)")

 
plt.savefig('test2.pdf') 

