# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:52:08 2021

@author: Arian Kamphuis
"""
#Importing packages
import time
starttime=time.time()

import feather
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
#%matplotlib inline

class Layering:
    def __init__(self):
        self.matrix = {}
        
    
    def Unpackdf(self,file):
        dataframe = feather.read_dataframe(file)
        df = dataframe
        dfNumpy = df.to_numpy()
        return dfNumpy, df

    
def Su_avg(df1363,Layer_lst,filename):
    plt.figure(figsize=(20,10))
    Su = df1363['Su']
    d = df1363['depth']
    cone_size = 164
    avg_step = d.iloc[-1]/len(d)*1000
    k = -int(np.round(cone_size/avg_step,0))
    moving_averages = []
    window_size = k*2
    i = 0
    d.drop(d.tail(window_size-1).index,inplace=True) # drop last n rows
    
    while i < len(Su) - window_size + 1:
        this_window = Su[i : i + window_size]
    
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
        
    plt.plot(Su,df1363['depth'],c='yellow')
    plt.plot(moving_averages,d,c='black')
    plt.axhline(df1363['depth'][Layer_lst[0]])
    plt.axhline(df1363['depth'][Layer_lst[1]])
    plt.axhline(df1363['depth'][Layer_lst[2]])
    plt.axhline(df1363['depth'][Layer_lst[3]])
    plt.title('Undrained shear stress'+filename)
    plt.xlabel('MPa')
    plt.ylabel('depth')
    plt.savefig(filename+'graphs_4.png')
    
def Isbt_avg(df1363,filename):
    plt.figure(figsize=(20,10))
    Ic = df1363['Isbt']
    d = df1363['depth']
    cone_size = 164
    avg_step = d.iloc[-1]/len(d)*1000
    k = -int(np.round(cone_size/avg_step,0)) #should be 8
    moving_averages = []
    window_size = k
    i = 0
    d.drop(d.tail(window_size-1).index,inplace=True) # drop last n rows
    
    while i < len(Ic) - window_size + 1:
        this_window = Ic[i : i + window_size]
    
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    plt.plot(Ic,df1363['depth'],c='yellow')
    plt.plot(moving_averages,d,c='black')
    plt.axvline(2.05,c='red')
    plt.axvline(2.60,c='red')
    plt.axvline(2.95,c='red')
    plt.axvline(3.6,c='red')
    plt.title("Isbt"+filename)
    
    Layer_lst = []
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <3.6))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.95))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.60))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.05))
    
    plt.axhline(df1363['depth'][Layer_lst[0]])
    plt.axhline(df1363['depth'][Layer_lst[1]])
    plt.axhline(df1363['depth'][Layer_lst[2]])
    plt.axhline(df1363['depth'][Layer_lst[3]])
    plt.savefig(filename+'graphs_3.png')
    
    return Layer_lst

    
    # def correlationFunc():
    #     tau_length = 50
    #     C_tau = np.zeros(tau_length)
    #     for j in range(tau_length):
    #         lst = []
    #         for i in range(len(sigma)-j):
    #             result = (sigma[i]-sample_avg)*(sigma[i+j-1]-sample_avg)
    #             lst.append(result)
    #         C_tau[j] = 1/(len(sigma)-j)*np.sum(lst)
    #     return C_tau
    
    # def Markov(theta=0.27):
    #     tau_length=np.arange(0,50,1)
    
    #     result = np.exp((-2*np.abs(tau_length))/theta)
    #     return result
        
     
    # C_tau = correlationFunc()
    # sample_correlation = C_tau/C_zero
    # theta = np.trapz(sample_correlation,dx=0.005)*2     
    # theory_correlation = Markov(theta)   
    
    
    # if plotGraphs ==True:
    #     tau = np.linspace(0,49,50)
    #     plt.figure()
    #     plt.plot(tau,sample_correlation, label='sampled autocorrelation')
    #     plt.title('Sample Auto-correlation function')
    #     plt.xlabel('lag (m)')
    #     plt.ylabel('Autocorrelation')
    #     plt.plot(tau, theory_correlation, c="red", label='Markov estimation')
    #     plt.legend()
    #     plt.savefig(filename+'graphs_4.png')

def readFiles():
    cwd = os.getcwd()
    print('Current directory is:', cwd)
    retval = os.getcwd()
    files = [f for f in listdir(retval) if isfile(join(retval,f))]
    return files,cwd

def checkFiles(files):
    lst = []
    for i in files:
        name = str(i)
        last_char = name[-7:]
        if last_char == 'feather':
            lst.append(i)
    print('Found', len(lst), '.feather files')
    return lst

def loopAnalyse(lst,cwd):
    for i in lst:
        if __name__=='__main__':
            p = Layering()
            dfNumpy, df = p.Unpackdf(i)
            print('Unpacked', i)
        filename = i.strip('.feather')
        Layer_lst = Isbt_avg(df,filename)
        Su_avg(df,Layer_lst,filename)
        
        
files,cwd = readFiles()
lst = checkFiles(files)
loopAnalyse(lst,cwd)
    

    
print('total runtime:',time.time()-starttime)