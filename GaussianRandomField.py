# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:05:31 2021

@author: Arian Kamphuis
"""
import time
starttime=time.time()
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from numpy import genfromtxt
#---------------------------------------------------
#previously calculated:
len_scale = 2.639
#---------------------------------------------------

psi = 12 # semi-manmade structures
#len_scale with auto_correlation is print(depth[130]-depth[0]) = 2.6m
v_SOF = len_scale
h_SOF = v_SOF * psi
variance = 0.07
#---------------------------------------

export = True
plot = True


x = np.arange(0,100,1)
y = np.arange(-30,0,30/1254)

if export ==True:
    #generating the random field
    seed = gs.random.MasterRNG(1)
    rng = np.random.RandomState(seed())
    
    model = gs.Gaussian(dim=2,var=variance,len_scale=[h_SOF,v_SOF])
    srf = gs.SRF(model)
    field = srf.structured((x,y))
    export = np.asarray(field)
    np.savetxt('field.csv',export,delimiter=',')


lin_approx = np.linspace(0.1,2.8,1254)
field_test = genfromtxt('field.csv', delimiter=',')
field_test[field_test < 0] = 0
for i in range(len(lin_approx)):
    field_test[:,i] += lin_approx[i]

if plot == True:
    plt.figure(figsize=(20,15))
    plt.imshow(field_test.T,aspect='auto',extent=[-40,80,-28,-2.16])
    plt.colorbar()
    plt.xlabel('horizontal distance [m]')
    plt.ylabel('depth [m]')
    title = '2D Gaussian Random Field '+ str(len(x))+':'+str(len(y))
    plt.title(title)
    plt.savefig('RandomField.jpg')


print('total runtime:',time.time()-starttime)