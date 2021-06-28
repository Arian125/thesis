# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:53:14 2021

@author: Arian Kamphuis
"""
import datetime
begin_time = datetime.datetime.now()
print("Starting CPT-Reader")
import feather
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from os import listdir
from os.path import isfile, join

def readFiles():
    path ='\CPTs'
    cwd = os.getcwd()
    print('Current directory is:', cwd)
    os.chdir(cwd+path)
    retval = os.getcwd()
    print('Succesfully changed to:',retval)
    files = [f for f in listdir(retval) if isfile(join(retval,f))]
    print('Files:', files)
    return retval,files,cwd


class GEF:
    def __init__(self):
        self._data_seperator = ";"
        self._columns = {}

        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.dz = []
        self.qc = []
        self.pw = []
        self.wg = []
        self.qs = []
        self.qt = []
        self.k = []

        

    def readFile(self, filename):
        print(filename)
        lines = open(filename, 'r').readlines()
        reading_header = True
            
        for line in lines:  

            if reading_header:
                self._parseHeaderLine(line)
            else:
                self._parseDataLine(line)
                    
            if line.find('#EOH') > -1:
                if self._check_header():
                    reading_header = False
                else:
                    return
        self.qs = self.addSleeveResistance()
        self.qt = self.TotConeResistance()
            
    def _check_header(self):
        if not 1 in self._columns:
            print("Fatale fout > Dit GEF bestand mist een diepte kolom")
            return False
                
        if not 2 in self._columns:
            print("Fatale fout > Dit GEF bestand mist een qc (conusweerstand) kolom")
            return False
                
        if not 3 in self._columns:
            print("Fatale fout > Dit GEF bestand mist een fs (plaatselijke wrijving pw) kolom")
            return False 
        if not 6 in self._columns:
            print('Geen u2 (pore pressure) gevonden in de file. qt = qc')
        return True
    
    def _parseHeaderLine(self, line):
        keyword, argline = line.split('=')
        keyword = keyword.strip()
        argline = argline.strip()
        args = argline.split(',')

        if keyword=='#XYID':
            self.x = float(args[1].strip())
            self.y = float(args[2].strip())
        elif keyword=='#ZID':
            self.z = float(args[1].strip())
        elif keyword=='#COLUMNINFO':
            column = int(args[0])
            dtype = int(args[-1].strip())
            if dtype==11: 
                dtype = 1
            self._columns[dtype] = column - 1
    
    def _parseDataLine(self, line):
        args = line.split(self._data_seperator)

        dz =self.z -float(args[self._columns[1]])

        qc = float(args[self._columns[2]])
        pw = float(args[self._columns[3]])   
        
            
        self.dz.append(dz)
        self.qc.append(qc)
        self.pw.append(pw)
        
            
        if qc<=0:
            self.wg.append(10)
        else:
            wg = (pw / qc) * 100
            if wg > 10:
                wg = 10
            elif wg == 0:#remove wg of 0, to avoid division by 0
                wg = 0.01
            self.wg.append(wg)
        


    def asNumpy(self):
        Isbt = self.Robertson()
        Su = self.UndrainedShearStrength()
        sigma_vo, sigma_vo_eff,hydroPressure = self.InSituStress()
        return np.transpose(np.array([self.dz,self.qt,self.pw,self.qc,self.wg,self.qs,sigma_vo,sigma_vo_eff,hydroPressure, Isbt,Su]))
 
    
    def addSleeveResistance(self):
        product = []
        for i,j in zip(self.qc,self.wg):
            product.append(i*j)
        return product
    
       
    def asDataFrame(self):
        a = self.asNumpy()
        return pd.DataFrame(data=a, columns=['depth', 'qt','fs', 'qc','wg','qs','sigma_vo','sigma_vo_eff','u0','Isbt','Su'])
    
    def Robertson(self):
        Pa = 0.1 #MPa
        quotient = [n/Pa for n in self.qc]
        Isbt = ((3.47-np.log10(quotient))**2+(np.log10(self.wg)+1.22)**2)**0.5 #wg bevat een 0
        return Isbt
    
    def HydraulicConductivity(self):
        Isbt = self.Robertson()
        for i in Isbt:
            if i < 3.27:
                k = 10**(0.952-3.04*i) #m/s
            elif i>3.27 and i<4:
                k = 10**(-4.52-1.37*i) #m/s
            else:
                print("can't compute k")
                k= 0
            self.k.append(k)
    
    def u_porepressure(self):
        gamma_water = 9.81 #kN/m^3
        wrong = self.dz[0]
        corrected = 0 #assuming water table at ground level
        constant = wrong + corrected
        height_lst = -np.subtract(self.dz,constant)
        hydroPressure=[n*gamma_water for n in height_lst]
        hydroPressure = [n/1000 for n in hydroPressure]
        return hydroPressure
    
    def Soilpressures(self):
        wg = self.wg   
        qc = self.qc
        gam_ref = 19*np.ones(len(wg))
        qt_ref = 5*np.ones(len(wg))
        Rf_ref = 30*np.ones(len(wg))
        beta = 4.12*np.ones(len(wg))
        gamma_sat = (gam_ref - beta*((np.log10(qt_ref/qc))/(Rf_ref/wg)))/1000
        CorHeightlst = np.abs(self.dz - self.dz[0]*np.ones(len(self.dz)))
        sigma_vo = gamma_sat*CorHeightlst
        return sigma_vo
    
    def InSituStress(self,):
        hydroPressure = self.u_porepressure()
        sigma_vo = self.Soilpressures()
        sigma_vo_eff = np.subtract(sigma_vo,hydroPressure)
        sigma_vo = np.nan_to_num(sigma_vo)
        sigma_vo_eff = np.nan_to_num(sigma_vo_eff)
        return sigma_vo, sigma_vo_eff,hydroPressure
    
    def TotConeResistance(self):
        #u2 nodig ipv u0
        #inbouwen dat error als u2 niet bestaat, dan qt = qc
        hydroPressure = self.u_porepressure()
        a = 0.7 #usual value between 0.6-0.85 to remove adverse effects of the uneven shape of the cone
        product = [n*(1-a) for n in hydroPressure]
        #print("Product",hydroPressure)
        qt = [sum(x) for x in zip(self.qc,product)]
        return qt
    
    def NetConePressure(self):
        qt = self.TotConeResistance()
        sigma_vo, sigma_vo_eff,hydroPressure = self.InSituStress()
        qn = np.subtract(qt, sigma_vo)
        return qn
    
    def Normalization(self): #Doesn' work, negative Qt, thus NaN for Ic
        sigma_vo,sigma_vo_eff,hydroPressure = self.InSituStress()
        qt = self.TotConeResistance()
        Qt=(qt-sigma_vo)/sigma_vo_eff
        Fr= self.pw/(qt-sigma_vo)*100
        Ic = ((3.47-np.log10(Qt)**2)+np.log10(Fr+1.22)**2)**0.5
        return Qt,Fr,Ic 
    
    # def Average_qt(self):
    #     df = self.asDataFrame2()
    #     d = df['depth']
    #     qt = df['qt']
    #     window_size = 8 #improve with calculation #cone is 164mm long
    #     i = 0
    #     moving_averages = []
    #     d.drop(d.tail(window_size-1).index,inplace=True) # drop last n rows
        
    #     while i < len(qt) - window_size + 1:
    #         this_window = qt[i : i + window_size]
        
    #         window_average = sum(this_window) / window_size
    #         moving_averages.append(window_average)
    #         i += 1
    #     qt_avg = moving_averages
    #     return qt_avg,d
    
    def UndrainedShearStrength(self):
        sigma_vo,sigma_vo_eff,hydroPressure = self.InSituStress()
        qt = self.TotConeResistance()
        #qt = self.TotConeResistance()
        Nkt = 12*np.ones(len(sigma_vo)) #range 10-18, sometimes even 6 for weak soft soils
        Su = np.subtract(qt,sigma_vo)/Nkt
        Su = np.nan_to_num(Su)
        return Su
    
    def SoilSensitivity(self):
        Su = self.UndrainedShearStrength()
        for i in range(len(self.pw)):
            if self.pw[i] <0.01:
                self.pw[i]=np.nan
        St = Su * ((1*np.ones(len(self.pw)))/self.pw)
        return St
    
    def SBTplot(self,filename):
        Ic = self.Robertson()
        index = []
        for i in range(len(Ic)):
            if Ic[i] > 3.6:
                index.append(2)
            elif Ic[i] > 2.95 and Ic[i] <=3.6:
                index.append(3)
            elif Ic[i] > 2.60 and Ic[i] <=2.95:
                index.append(4)
            elif Ic[i] > 2.05 and Ic[i] <=2.60:
                index.append(5)
            elif Ic[i] > 1.31 and Ic[i] <=2.05:
                index.append(6)
            else:
                index.append(7)
        z = int(self.dz[-1])
        ylim = z+1

        fig = plt.figure(figsize=(30,15))
        ax1 = fig.add_subplot(131)
        ax1.plot(self.qc,self.dz,color='black', label="qc")
        ax1.set_ylim(0,ylim)
        ax1.set_xlim(0,55.0)
        
        ax1.invert_yaxis()
        ax1.minorticks_on()
        ax1.grid(True, which='both')
        ax1.legend(loc=3)
        ax1.set_ylabel("depth (m)",size=12)
        ax1.set_xlabel("qc (MPa)",size=12)
        
        ax2 = ax1.twiny()
        ax2.plot(self.wg,self.dz,color='green', label="wg")
        ax2.set_ylim(0,ylim)
        ax2.set_xlim(0,25.0)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.set_xlabel("wg (%)",size=12)
        
        ax3 = fig.add_subplot(132)
        ax3.plot(Ic,self.dz,color='yellow',linewidth=2)
        ax3.set_ylim(0,ylim)
        ax3.set_xlim(1.0,4.0)
        ax3.invert_yaxis()
        ax3.text(1.05, -9.8, "gravel - dense sand", va='bottom', rotation=90, size=17, color="white")
        ax3.text(1.55, -9.8, "clean sand - silty sand", va='bottom', rotation=90, size=17, color="white")
        ax3.text(2.22, -9.8, "silty sand - sandy silt", va='bottom', rotation=90, size=17, color="white")
        ax3.text(2.65, -9.8, "clayey silt - silty clay", va='bottom', rotation=90, size=17, color="white")
        ax3.text(3.2, -9.8, "silty clay - clay", va='bottom', rotation=90, size=17, color="white")
        ax3.text(3.65, -9.8, "peat", va='bottom', rotation=90, size=17, color="white")

        ax3.add_patch(patches.Rectangle((0,0),1.31,ylim,facecolor='darkgoldenrod'))
        ax3.add_patch(patches.Rectangle((1.31,0),0.74,ylim,facecolor='goldenrod'))
        ax3.add_patch(patches.Rectangle((2.05,0),0.55,ylim,facecolor='mediumseagreen'))
        ax3.add_patch(patches.Rectangle((2.60,0),0.35,ylim,facecolor='seagreen'))
        ax3.add_patch(patches.Rectangle((2.95,0),0.65,ylim,facecolor='cadetblue'))
        ax3.add_patch(patches.Rectangle((3.6,0),0.4,ylim,facecolor='sienna'))
        ax3.yaxis.grid(which="minor")
        ax3.yaxis.grid(which="major")
        ax3.xaxis.grid(which="major")
        ax3.minorticks_on()

        ax3.set_xlabel("Ic SBT",size=12)
        ax3.set_ylabel("depth (m)")
        
        ax4 = fig.add_subplot(133)
        for i in range(len(Ic)-1):
            if index[i+1] == 2:
                colour = 'sienna'
            elif index[i+1] == 3:
                colour = 'cadetblue'
            elif index[i+1] == 4:
                colour = 'seagreen'
            elif index[i+1] == 5:
                colour = 'mediumseagreen'
            elif index[i+1] == 6:
                colour = 'goldenrod'
            elif index[i+1] == 7:
                colour = 'darkgoldenrod'
            ax4.add_patch(patches.Rectangle((0,self.dz[i]),index[i+1],(0.1),facecolor=colour))
        ax4.set_ylim(0,ylim)
        ax4.set_xlim(0,10.0)
        ax4.set_ylabel("depth (m)")
        ax4.invert_yaxis()
        ax4.yaxis.grid(which="major")
        ax4.yaxis.grid(which="minor")
        ax4.xaxis.grid(which="major")
        ax4.minorticks_on()
        
        plt.savefig(filename)
        plt.close(None)
        
    def plot(self, filename):    
        df = self.asDataFrame()
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20,12), gridspec_kw = {'width_ratios':[3, 1, 2, 1, 1,1]})
        axes[0].set_xlim([0,40])
        df.plot(x='qc', y='depth', ax=axes[0], sharey=True, label='qc, cone resistance [MPa]')
        df.plot(x='qt', y='depth', ax=axes[0], sharey=True, label='qt, corrected cone resistance [MPa]')
        df.plot(x='fs', y='depth', ax=axes[1], sharey=True, label='fs, local/friction resistance [MPa]')
        df.plot(x='wg', y='depth', ax=axes[2], sharey=True, label='wg, friction ratio [%]')
        df.plot(x='qs', y='depth', ax=axes[3], sharey=True, label='qs, sleeve resistane[kPa]')
        df.plot(x='u0', y='depth', ax=axes[4], sharey=True, label='u0, water pressure[MPa]')
        df.plot(x='Isbt', y='depth',ax=axes[5], sharey=True, label='Isbt, non-normalized Soil behaviour Type index')

                
        for i in range(6): axes[i].grid()
        plt.savefig(filename)
        plt.close(None)
    

retval,files,cwd = readFiles()
purpose = 'export'



def LoopReader(retval,file,purpose,cwd):
    path ='\CPTs'
    os.chdir(cwd+path)
    if __name__=="__main__":
        g = GEF()
        g.readFile(file)
        file = file.strip('.gef')
        print(file)
        if purpose == 'export':
            Export_df = g.asDataFrame()
            filename = str(file) + '.feather'
            os.chdir(cwd)    
            feather.write_dataframe(Export_df, filename)
            print("Finished exporting data")
        elif purpose =='plot':
            os.chdir(cwd)
            g.plot(file + "_graphs_1.png")
            # g.plot2(file + '_graphs_2.png')
            g.SBTplot(file+"_SBT.png")
            print("Finished plotting graphs")
        else:
            print("Purpose not found")

for i in files:
    LoopReader(retval,i,purpose,cwd)


print("Total runtime:",datetime.datetime.now() - begin_time)





















