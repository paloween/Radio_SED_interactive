#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:53:35 2018

@author: pallavipatil
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import astropy.io
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, join
import numpy as np
from astropy.io import ascii
import os
import glob
import numpy
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator
from operator import itemgetter, attrgetter
import scipy.optimize as optimization
from scipy import stats
import matplotlib.cm as cm
import scipy.optimize
import function_class
from file_prep_radio_fitting import data_prep

def EFFA_func(nu, s0, alpha, nu_t): 
        return s0*nu**alpha*np.exp(-(nu/nu_t)**(-2.1))
   
        
##########################################################################
# Internal Free Free Absorption:
# Thermal plasma mixed with relativistic particles can produce spectral TO.
# Similar to escape probability mechanism. 
# S_nu = s0* nu**-alpha * [(1-exp**(-tau)) / (tau)]
# Where tau ~ nu**(-2.1) 
        
def IFFA_func(nu, s0, alpha, nu_t): 
    return s0*nu**alpha*( (1-np.exp(-(nu/nu_t)**(-2.1)))/((nu/nu_t)**(-2.1)))

##########################################################################
# Synchrotron Self Absorption
# The source TO is becuase the source cannot have a brightness temp greater
# than the plasma temperature of the nonthermal electrons. 
# alpha = -(beta-1)/2--- beta = 1-2*alpha 
# S_nu = s0 *(nu/nu_t)**-(beta-1)/2 * (1-exp**(-tau)/ tau)
# Where tau = (nu/nu_t)**-(beta+4)/2

def SSA_func(nu, s0, alpha, nu_t): 
    beta = 1-2*alpha
    tau = (nu/nu_t)**(-(beta+4)/2)
    return s0*((nu/nu_t)**(-(beta-1)/2))* ((1-np.exp(-tau))/tau )
    
    
def check_sourcestruct(row):
    
    if np.ma.is_masked(row['P_flux'])== False :
        if row['result'] =='UR':
            flux = row['P_flux']
            flux_err = row['pflux_err']
        else:
            flux = row['I_flux']
            flux_err = row['iflux_err']    
    else:
        flux = -999.0
        flux_err= -999.0
    return flux, flux_err    
    
def check_Spindex(row):
    if np.ma.is_masked(row['SpIdx'])== False :
        spid = row['SpIdx']
        sperr = row['SpIdx_err'] 
    else:
        spid = -9999.0
        sperr = -9999.0
    return spid, sperr   
    
def model_effa(nu_arr, s0, alpha, nu_t):
    flux = np.copy(nu_arr)
    for kk,nu in enumerate(nu_arr):
        if nu >0:
            flux[kk]=  EFFA_func(nu,s0, alpha, nu_t)
        else:
            nv = -nu
            n_low = nv*0.99
            n_up = nv*1.01
            flux[kk] = np.log10(EFFA_func(n_up,s0, alpha, nu_t)/EFFA_func(n_low,s0, alpha, nu_t))/np.log10(n_up/n_low)
    return flux
    
def model_iffa(nu_arr, s0, alpha, nu_t):
    flux = np.copy(nu_arr)
    for kk,nu in enumerate(nu_arr):
        if nu >0:
            flux[kk]=  IFFA_func(nu,s0, alpha, nu_t)
        else:
            nv = -nu
            n_low = nv*0.99
            n_up = nv*1.01
            flux[kk] = np.log10(IFFA_func(n_up,s0, alpha, nu_t)/IFFA_func(n_low,s0, alpha, nu_t))/np.log10(n_up/n_low)
    return flux
        
def model_ssa(nu_arr, s0, alpha, nu_t):
    flux = np.copy(nu_arr)
    for kk,nu in enumerate(nu_arr):
        if nu >0:
            flux[kk]=  SSA_func(nu,s0, alpha, nu_t)
        else:
            nv = -nu
            n_low = nv*0.99
            n_up = nv*1.01
            flux[kk] = np.log10(SSA_func(n_up,s0, alpha, nu_t)/SSA_func(n_low,s0, alpha, nu_t))/np.log10(n_up/n_low)
    return flux


def plot_bowtie(alpha_x, flux, e_flux,ax):
    s0 = flux
    nu_t = abs(alpha_x[0])
    sp1 = alpha_x[1]
    sp2 = sp1+alpha_x[2]
    sp3 = sp1-alpha_x[2]
    nu_cen = np.linspace(8,12,num=5)
    f1 = s0*np.power((nu_cen/nu_t), sp1)
    f2 = s0*np.power((nu_cen/nu_t), sp2)
    f3 = s0*np.power((nu_cen/nu_t), sp3)
    ax.plot(nu_cen,f1,'-', linewidth = 1.5)
    ax.plot(nu_cen,f2,'-', color = 'blue',)
    ax.plot(nu_cen,f3,'-', color= 'blue', )


##############################################################################
datadir = '/Users/pallavipatil/Desktop/VLA/Radio_Fits_v2/'
os.chdir(datadir)

hdulist = fits.open('WISE_Xmatch.fits')
scat = hdulist[1].data 
cols = hdulist[1].columns
atscat = Table(scat)
vla_ax = ascii.read('JMFIT_CASA_A_results.dat', format = 'csv', delimiter = '\t', fast_reader = False, fill_values =('--', np.nan))
vla_bx = ascii.read('JMFIT_CASA_B_results.dat', format = 'csv', delimiter = ',', fast_reader = False, fill_values =('--', np.nan))
    
vla_ax_grp = vla_ax.group_by(keys = 'WISEname')
vla_bx_grp = vla_bx.group_by(keys = 'WISEname')






#    freq_arr, flux_arr, eflux_arr, alpha_X = data_prep(i)
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.25, bottom=0.3, right=0.8)

infoax = plt.axes([0.025,0.1,0.15, 0.5], facecolor= '#C8C9C7', alpha=0.1)
infoax.tick_params(axis='both', bottom='off', left='off', labelbottom='off', labelleft='off')

#####Sliders
axcolor = 'lightblue'
axalpha = plt.axes([0.25, 0.07, 0.65, 0.02], facecolor=axcolor)
axs0 = plt.axes([0.25, 0.12, 0.65, 0.02], facecolor=axcolor)
axnu = plt.axes([0.25, 0.17, 0.65, 0.02], facecolor=axcolor)

salpha = Slider(axalpha, r'$\alpha$', -3, 3.0, valinit=0.0)
ss0 = Slider(axs0, r'$S_{0}$', 0.1, 100, valinit=1.0)
snu = Slider(axnu, r'$\nu_{peak}$', 0.001, 20, valinit=0.0)   
# Reset Button
resetax = plt.axes([0.8, 0.025, 0.1, 0.03])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Model selection
rax = plt.axes([0.025, 0.75, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ( 'EFFA', 'IFFA', 'SSA', 'All'), active=0)

# Refit button
refitax = plt.axes([0.6,0.025,0.1,0.03])
ref_button = Button(refitax, 'Refit', color=axcolor, hovercolor='0.975')

#Next and Previous button



class Plot_class:
    def __init__(self):
        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.nu_mod = np.logspace(-3, 2.0, 5000)
        self.info  =None
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.fit_effa = None
        self.fit_iffa = None
        self.fit_ssa = None
        self.nu_arr=None
        self.s_nu=None
        self.e_s_nu=None
        


    def plotting(self,i):  
        name = atscat['WISEname'][i] 
        
        FAX_10GHZ = 0 
        EAX_10GHZ = 0 
        FBX_10GHZ = 0 
        EBX_10GHZ = 0 
        freq_arr, flux_arr, eflux_arr, alpha_X = data_prep(i)    
        
        jvla_AX = vla_ax_grp.groups[i]
        jvla_BX = vla_bx_grp.groups[i]    
            
        for reg in jvla_AX:
            [f, ferr] = check_sourcestruct(reg)
            FAX_10GHZ +=f
            EAX_10GHZ +=ferr
        for reg in jvla_BX:
            [f, ferr] = check_sourcestruct(reg)
            FBX_10GHZ +=f
            EBX_10GHZ +=ferr 
        
        #observed data points
        ax.errorbar(freq_arr,flux_arr, yerr=eflux_arr, marker ='o', linestyle='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plot_bowtie(alpha_X, FAX_10GHZ, EAX_10GHZ,ax)
        ax.set_ylim(0.01*min(flux_arr), 100*max(flux_arr))
        ax.set_xlim(0.01*min(freq_arr), 100*max(freq_arr))
        ax.set_xlabel(r'log $\frac{\nu}{GHz}$', fontdict=dict(fontsize=12))
        ax.set_ylabel(r'log $\frac{S_{\nu}}{mJy}$', fontdict=dict(fontsize=12))
        ax.tick_params(axis = 'both', which = 'minor', direction='in', length=4, top=True, right=True)
        ax.tick_params(axis = 'both', which = 'major', direction='in', length=9, top=True, right=True)
        ax.set_title(name)
        #Guess parameters
        s0= atscat['FNVSS'][i]
        nu_t = 1.4
        alpha = -0.7
        guess_par=np.array([s0,alpha,nu_t])
        #Initial fitting and plot
        #arrays for fitting:
        self.nu_arr = freq_arr
        self.s_nu= flux_arr
        self.e_s_nu = eflux_arr
        self.nu_arr = np.append(self.nu_arr, alpha_X[0])
        self.s_nu = np.append(self.s_nu, alpha_X[1][0])
        self.e_s_nu = np.append(self.e_s_nu, alpha_X[2][0])
        
        
        #### Fitting Routine #########
        self.fit_effa,cov1 = scipy.optimize.curve_fit(model_effa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)   
        self.fit_iffa,cov2 = scipy.optimize.curve_fit(model_iffa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
        self.fit_ssa, cov3 = scipy.optimize.curve_fit(model_ssa,  self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
             
        # Line to plot the model
        self.s1 = EFFA_func(self.nu_mod, self.fit_effa[0],self.fit_effa[1], self.fit_effa[2])
        self.s2 = IFFA_func(self.nu_mod, self.fit_iffa[0],self.fit_iffa[1], self.fit_iffa[2])
        self.s3 = SSA_func(self.nu_mod, self.fit_ssa[0],self.fit_ssa[1], self.fit_ssa[2])
        
        self.l1, = ax.plot(self.nu_mod, self.s1, 'blue', linestyle = '-', label='EFFA')
        self.l2, = ax.plot(self.nu_mod, self.s2, 'red', linestyle = '-.', label='IFFA')
        self.l3, = ax.plot(self.nu_mod, self.s3, 'green', linestyle = ':', label='SSA')
        ax.legend(loc = 'upper right')
            # Sliders
    
        salpha.on_changed(self.update)
        ss0.on_changed(self.update)
        snu.on_changed(self.update)
        button.on_clicked(self.reset)
        radio.on_clicked(self.modelfunc)
        ref_button.on_clicked(self.refit)
    
        #Fit prameters
        mystring = 'Fitted Parameters \n'
        mystring += '\n EFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_effa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_effa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_effa[2])+'\n'
        mystring += '\n IFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_iffa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_iffa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_iffa[2])+'\n'
        mystring += '\n SSA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_ssa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_ssa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_ssa[2])+'\n'
        self.info, = infoax.text(0.05,0.05,mystring, wrap=True ) 
        plt.show()
    
    
    
    def update(self,val):
        alpha = salpha.val
        s0 = ss0.val
        nu_t = snu.val
        if radio.value_selected=='EFFA':
            s_up = EFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l1.set_ydata(s_up)
        if radio.value_selected=='IFFA':
            s_up = IFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l2.set_ydata(s_up)
        if radio.value_selected=='SSA':
            s_up = SSA_func(self.nu_mod, s0, alpha, nu_t)
            self.l3.set_ydata(s_up)  
        fig.canvas.draw_idle()
    
    def reset(self,event):
        salpha.reset()
        ss0.reset()
        snu.reset()
        if radio.value_selected=='All':
            self.l1.set_ydata(self.s1)
            self.l2.set_ydata(self.s2)       
            self.l3.set_ydata(self.s3)
     
    def modelfunc(self,label):
        if label=='EFFA':
            s_up = EFFA_func(self.nu_mod, self.fit_effa)
            self.l1.set_ydata(s_up)
            self.l1.set_color('blue')
            self.l2.set_visible(False)
            self.l3.set_visible(False)
        if label=='IFFA':
            s_up = IFFA_func(self.nu_mod, self.fit_iffa)
            self.l1.set_ydata(s_up)
            self.l1.set_color('red')
            self.l2.set_visible(False)
            self.l3.set_visible(False)
        if label=='SSA':
            s_up = SSA_func(self.nu_mod, self.fit_ssa)
            self.l1.set_ydata(s_up)
            self.l1.set_color('green')
            self.l2.set_visible(False)
            self.l3.set_visible(False)
        if label=='All':
            s_1 = EFFA_func(self.nu_mod,self.fit_effa)
            s_2 = IFFA_func(self.nu_mod, self.fit_iffa)
            s_3 = SSA_func(self.nu_mod, self.fit_ssa)
            self.l1.set_ydata(s_1)
            self.l1.set_color('blue')
            self.l1.set_linestyle('-')
            self.l2.set_visible(True)
            self.l3.set_visible(True)
            self.l2.set_ydata(s_2)
            self.l2.set_color('red')
            self.l2.set_linestyle('-.')
            self.l3.set_ydata(s_3)
            self.l3.set_color('green')
            self.l3.set_linestyle(':')
        fig.canvas.draw_idle()
        
    def refit(self,event):
        guess_par = [ss0.val, salpha.val, snu.val]
        fit_effa,cov1 = scipy.optimize.curve_fit(model_effa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)   
        fit_iffa,cov2 = scipy.optimize.curve_fit(model_iffa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
        fit_ssa, cov3 = scipy.optimize.curve_fit(model_ssa,  self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
    
        s1 = EFFA_func(self.nu_mod, fit_effa[0],fit_effa[1], fit_effa[2])
        s2 = IFFA_func(self.nu_mod, fit_iffa[0],fit_iffa[1], fit_iffa[2])
        s3 = SSA_func(self.nu_mod, fit_ssa[0],fit_ssa[1], fit_ssa[2])
    
        self.l1.set_ydata(s1)
        self.l2.set_ydata(s2)
        self.l3.set_ydata(s3)
    
        #Fit prameters
        mystring = 'Fitted Parameters \n'
        mystring += '\n EFFA: \n'+  r'$S_0$ = {:.2f}'.format(fit_effa[0])+' \n'+r'$\alpha$= {:.2f}'.format(fit_effa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(fit_effa[2])+'\n'
        mystring += '\n IFFA: \n'+  r'$S_0$ = {:.2f}'.format(fit_iffa[0])+' \n'+r'$\alpha$= {:.2f}'.format(fit_iffa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(fit_iffa[2])+'\n'
        mystring += '\n SSA: \n'+  r'$S_0$ = {:.2f}'.format(fit_ssa[0])+' \n'+r'$\alpha$= {:.2f}'.format(fit_ssa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(fit_ssa[2])+'\n'
        self.info.set_text(mystring )         

class Index(object):
    
    def __init__(self):
        self.ind = 0
        self.obj_plot = Plot_class()
    def next(self, event):
        self.ind += 1
        i = self.ind
        ax.clear()
        infoax.clear()
        self.obj_plot.plotting(i)

    def prev(self, event):
        self.ind -= 1
        i = self.ind 
        ax.clear()
        infoax.clear()
        self.obj_plot.plotting(i)


callback = Index()
axprev = plt.axes([0.85, 0.75, 0.1, 0.03])
axnext = plt.axes([0.85, 0.80, 0.1, 0.03])
bnext = Button(axnext, 'Next',color=axcolor, hovercolor='0.975')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous',color=axcolor, hovercolor='0.975')
bprev.on_clicked(callback.prev)


def main():
    i = 0
    obj_pl = Plot_class()
    obj_pl.plotting(i)
    
main()

