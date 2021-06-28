# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:20:29 2021

@author: Arian Kamphuis
"""

import time
starttime=time.time()

import feather
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import gstools as gs

import warnings
warnings.filterwarnings("ignore")
import statistics
import skgstat as skg
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


CPT1363_xy = 110007.320, 434797.180
CPT1364_xy = 110029.560, 434806.570 #middle one
CPT1365_xy = 110047.630, 434778.380

#%matplotlib inline

filelaptop = 'C:\\Users\\StudiePC\\OneDrive\\MSc Thesis\\CPT_Reader\\CPT000000081363_IMBRO_A.feather'
filePC_1363 = 'C:\\Users\\Arian Kamphuis\\OneDrive\\MSc Thesis\\CPT_Reader\\CPT000000081363_IMBRO_A.feather'
filePC_1364 = 'C:\\Users\\Arian Kamphuis\\OneDrive\\MSc Thesis\\CPT_Reader\\CPT000000081364_IMBRO_A.feather'
filePC_1365 = 'C:\\Users\\Arian Kamphuis\\OneDrive\\MSc Thesis\\CPT_Reader\\CPT000000081365_IMBRO_A.feather'
plotGraphs = True
class Layering:
    def __init__(self):
        self.matrix = {}
        
    
    def Unpackdf(self,file):
        dataframe = feather.read_dataframe(file)
        df = dataframe
        dfNumpy = df.to_numpy()
        return dfNumpy, df

#-------------------------------------
if __name__=='__main__':
            p = Layering()
            dfNumpy1363, df1363 = p.Unpackdf(filePC_1363)
if __name__=='__main__':
            p = Layering()
            dfNumpy1364, df1364 = p.Unpackdf(filePC_1364)        
if __name__=='__main__':
            p = Layering()
            dfNumpy1365, df1365 = p.Unpackdf(filePC_1365)
    
def Distance(CPT1363_xy,CPT1364_xy,CPT1365_xy):
    CPT1,CPT2,CPT3 = CPT1363_xy,CPT1364_xy,CPT1365_xy
    dist_1 = np.sqrt((CPT1[0]-CPT2[0])**2+(CPT1[1]-CPT2[1])**2)
    dist_2 = np.sqrt((CPT1[0]-CPT3[0])**2+(CPT1[1]-CPT3[1])**2)
    dist_3 = np.sqrt((CPT2[0]-CPT3[0])**2+(CPT2[1]-CPT3[1])**2)
    lst = [dist_1,dist_2,dist_3]
    return lst


dist_CPT = Distance(CPT1363_xy,CPT1364_xy,CPT1365_xy)

def Su_avg(df1363,Layer_lst):
    plt.figure()
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
    plt.title('Undrained shear stress')
    plt.xlabel('MPa')
    plt.ylabel('depth')
    return moving_averages
    
def Isbt_avg(df1363):
    plt.figure()
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
    plt.title("Isbt")
    
    Layer_lst = []
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <3.6))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.95))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.60))
    Layer_lst.append(next(x[0] for x in enumerate(moving_averages) if x[1] <2.05))
    
    plt.axhline(df1363['depth'][Layer_lst[0]])
    plt.axhline(df1363['depth'][Layer_lst[1]])
    plt.axhline(df1363['depth'][Layer_lst[2]])
    plt.axhline(df1363['depth'][Layer_lst[3]])
    
    return Layer_lst

#soil boundary lines
#removed BL[2] to improve analysis, it is still manual
BL = Isbt_avg(df1363)
MovingAvgSu = Su_avg(df1363,BL)


#this particular order is important, otherwise BL list gets initial 0 value
#BL.insert(0,0)
#del BL[2]
BL.insert(0,0)
BL.append(len(MovingAvgSu))
# print(BL)
del BL[2]
del BL[2]
#----------------------------------
#Adjusting for this particular CPT (adjusting of the trend)
BL[2] = 300
# plt.figure()
# plt.plot(MovingAvgSu,df1363['depth'][BL[0]:BL[-1]])

print(BL)
layer_slope = np.zeros(len(BL))
layer_cept = np.zeros(len(BL))

for i in range(len(BL)-1):
    slope, intercept, r_value, p_value, std_err = stats.linregress(MovingAvgSu[BL[i]:BL[i+1]],df1363['depth'][BL[i]:BL[i+1]])
    layer_slope[i] = slope
    layer_cept[i] = intercept


# for i in range(len(BL)-1):
#     y = df1363['depth'][BL[i]:BL[i+1]]
#     x = (y-layer_cept[i])/layer_slope[i]
#     plt.plot(x,y,c='purple')
#     plt.xlim(-0.5,2.3)


Su = df1363['Su']
Su_avg = np.array(MovingAvgSu)
sigma = np.zeros(len(MovingAvgSu))
lin_approx = np.zeros(len(MovingAvgSu))

for i in range(len(BL)-1):
    
    y = df1363['depth'][BL[i]:BL[i+1]]
    sigma[BL[i]:BL[i+1]] = MovingAvgSu[BL[i]:BL[i+1]] - ((y-layer_cept[i])/layer_slope[i])
    lin_approx[BL[i]:BL[i+1]] = ((y-layer_cept[i])/layer_slope[i])


#plt.figure()
#plt.plot(sigma,df1363['depth'][BL[0]:BL[-1]],c='red')
#plt.plot(np.abs(sigma),df1363['depth'][BL[0]:BL[-1]],c='black')

depth = dfNumpy1363[BL[0]:BL[-1],[0]]
Vcu = np.divide(sigma,Su_avg)


if plotGraphs == True:
    fig = plt.figure(figsize=(30,15))
    ax1 = fig.add_subplot(141)
    ax1.plot(Vcu, depth)
    ax1.set_title("coefficient of variation")
    
    ax2 = fig.add_subplot(142)
    ax2.plot(Su_avg,depth,label="Cu calculated")
    ax2.plot(lin_approx,depth,label='Cu linearized')
    ax2.legend()
    
    ax3 = fig.add_subplot(143)
    ax3.plot(sigma,depth,label='sigma')
    ax3.set_xlabel('Cu detrended')
    ax3.set_ylabel('depth [m]')
    ax3.legend()
    
    ax4 = fig.add_subplot(144)
    ax4.plot(np.abs(sigma),depth,label='absolute sigma')
    ax4.set_xlabel('[MPa]')
    ax4.set_ylabel('depth [m]')
    ax4.set_title('Absolute sigma')


sample_avg = np.array(np.mean(np.abs(sigma)))
sigma_var = np.abs(np.var(sigma,ddof=1))
y = np.linspace(0,depth[-1],len(sigma))
idx = np.argwhere(np.diff(np.sign(sigma))).flatten()


if plotGraphs ==True:
    plt.figure(figsize=(10,20))
    plt.plot(sigma,y)
    plt.axvline(0,linestyle='dashed',linewidth=1)
    plt.plot(sigma[idx],y[idx],'ro')

avg_lenScale = []
for i in range(len(idx)-1):
    temp = float(depth[idx[i]]-depth[idx[i+1]])
    avg_lenScale.append(temp)
len_scale = 1/len(avg_lenScale)*np.sum(avg_lenScale)
print('Average length scale:',len_scale)



def correlateNumpy(sigma,length=500):
    mean=sigma.mean()
    var=np.var(sigma)
    xp=sigma-mean
    corr=np.correlate(xp,xp,'full')[len(sigma)-1:]/var/len(sigma)
    lags = range(length)
    return corr[:len(lags)]

def Second_Order_Markov(theta=len_scale):
    tau_length=np.arange(0,8,8/70)
    result = (1+(4*np.abs(tau_length))/theta)*np.exp(-(4*np.abs(tau_length))/theta)
    return result
    
correlation_length = correlateNumpy(sigma)
temp3 = np.argwhere(np.diff(np.sign(correlation_length))).flatten()
len_scale2 = np.round(float(depth[0] - depth[temp3[0]]),3)

print('Calculated correlation distance vertical:', temp3[0], 'in lags')    
print('Calculated correlation distance vertical:', len_scale2, 'in m')
theory_correlation = Second_Order_Markov(theta=len_scale) 
  
seed = gs.random.MasterRNG(1)
rng = np.random.RandomState(seed())
y = rng.randint(0,100,size=1254)
x = rng.randint(0,100,size=1254)

model = gs.Gaussian(dim=1,var=np.abs(sigma_var),len_scale=len_scale2)
srf = gs.SRF(model)
field = srf([y]) + lin_approx

imshow_field = np.zeros((len(field),2))
imshow_field[:,0] = field
imshow_field[:,1] = field


if plotGraphs ==True:
    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot(121)
    tau = np.linspace(0,500,500)
    ax1.plot(tau,correlation_length, label='sampled autocorrelation')
    ax1.set_title('Auto-correlation function')
    ax1.set_xlabel('lag distance (index)')
    ax1.set_ylabel('Autocorrelation')
    ax1.axhline(0)
    ax1.axvline(130)

    plt.legend()
    ax2 = plt.subplot(122)
    tau2 = np.linspace(0,8,70)
    ax2.plot(tau2, theory_correlation, c="red", label='Second order Markov estimation')
    ax2.set_title('theory correlation')
    ax2.set_xlabel('lag distance [m]')
    ax2.axvline(2.6)
    plt.legend()
    
    plt.figure()
    ax = srf.plot()
    ax.set_aspect('auto')
    plt.figure(figsize=(30,15))
    plt.imshow(imshow_field,aspect='auto',extent=[-2,4,-28,-2.16])
    plt.plot(field,depth)
    plt.colorbar()
    
    fig = tsaplots.plot_acf(sigma, lags=150)
    plt.xlabel('number of lags')
    

print('total runtime:',time.time()-starttime)