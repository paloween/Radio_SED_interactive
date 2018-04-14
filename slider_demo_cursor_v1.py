#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:51:39 2018

@author: pallavipatil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 06:53:39 2018

@author: pallavipatil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:53:35 2018

@author: pallavipatil
"""

import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons, Cursor
from astropy.io import fits,ascii
import matplotlib.pyplot as plt
from astropy.table import Table, join
import os
import glob
import scipy
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator
from operator import itemgetter, attrgetter
import scipy.optimize as optimization
from scipy import stats
import matplotlib.cm as cm
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
cursor = Cursor(ax, useblit=True, color='k', linewidth=1 )

infoax = plt.axes([0.025,0.1,0.15, 0.5], facecolor= '#C8C9C7', alpha=0.1)
infoax.tick_params(axis='both', bottom='off', left='off', labelbottom='off', labelleft='off')

#####Sliders
axcolor = 'lightblue'
axalpha = plt.axes([0.25, 0.07, 0.65, 0.02], facecolor=axcolor)
axs0 = plt.axes([0.25, 0.12, 0.65, 0.02], facecolor=axcolor)
axnu = plt.axes([0.25, 0.17, 0.65, 0.02], facecolor=axcolor)

salpha = Slider(axalpha, r'$\alpha$', -3, 3.0, valinit=-1)
#ss0 = Slider(axs0, r'$S_{0}$', 0.1, 100, valinit=1.0)
#snu = Slider(axnu, r'$\nu_{peak}$', 0.001, 20, valinit=0.0) 
snu = Slider(axnu, r'log $\nu_{peak}$', -3, 2, valinit=-1) 
ss0 = Slider(axs0, r'log $S_{0}$', -1, 3, valinit=0)

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
axprev = plt.axes([0.85, 0.75, 0.1, 0.03])
axnext = plt.axes([0.85, 0.80, 0.1, 0.03])
bnext = Button(axnext, 'Next',color=axcolor, hovercolor='0.975')
bprev = Button(axprev, 'Previous',color=axcolor, hovercolor='0.975')

#save button
axsave = plt.axes([0.85, 0.65, 0.1, 0.03])
bsave = Button(axsave, 'Save', color=axcolor, hovercolor = '0.975')

#replot button
axrepl= plt.axes([0.85, 0.55, 0.1, 0.03])
brepl = Button(axrepl, 'replot', color=axcolor, hovercolor = '0.975')


fitTab = ascii.read('GuessPar_Radiofits.csv', format='basic', delimiter=',')
fitTab.add_index('Name')
class Plot_class:
    def __init__(self):
        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.nu_mod = np.logspace(-3, 2.0, 5000)
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.fit_effa = None
        self.fit_iffa = None
        self.fit_ssa = None
        self.nu_arr=None
        self.s_nu=None
        self.e_s_nu=None
        self.keypress = None
        #self.clickbutton = []
        self.clickx_data = None
        self.clicky_data = None
        self.name = None
        


    def plotting(self,i):  
        self.name = atscat['WISEname'][i] 

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
        ax.set_title(self.name)
        #Guess parameters
        s0= atscat['FNVSS'][i]
        nu_t = 1.4
        alpha = -0.7
        indg = fitTab.loc[self.name].index
        row_g = list(fitTab[indg])
        guess_par = [s0, alpha,nu_t]
        guess_effa = guess_par; guess_iffa=guess_par; 
        guess_ssa=guess_par
        
        if np.ma.is_masked(row_g[1]) ==False:
            guess_effa = row_g[1:4]
        else:
            guess_effa=guess_par
        if np.ma.is_masked(row_g[4]) ==False:
            guess_iffa = row_g[4:7]
        else:
            guess_iffa=guess_par
            
        if np.ma.is_masked(row_g[7]) ==False:
            guess_ssa = row_g[7:]
        else:
            guess_ssa=guess_par

        
        #Initial fitting and plot
        #arrays for fitting:
        self.nu_arr = freq_arr
        self.s_nu= flux_arr
        self.e_s_nu = eflux_arr
        self.nu_arr = np.append(self.nu_arr, alpha_X[0])
        self.s_nu = np.append(self.s_nu, alpha_X[1][0])
        self.e_s_nu = np.append(self.e_s_nu, alpha_X[2][0])
        
        
        #### Fitting Routine #########
        self.fit_effa,cov1 = scipy.optimize.curve_fit(model_effa, self.nu_arr, self.s_nu, guess_effa, self.e_s_nu)   
        self.fit_iffa,cov2 = scipy.optimize.curve_fit(model_iffa, self.nu_arr, self.s_nu, guess_iffa, self.e_s_nu)        
        self.fit_ssa, cov3 = scipy.optimize.curve_fit(model_ssa,  self.nu_arr, self.s_nu, guess_ssa, self.e_s_nu)        
             
        chisq_effa = np.square((model_effa(self.nu_arr, self.fit_effa[0],self.fit_effa[1],self.fit_effa[2]) -self.s_nu)/(self.e_s_nu))
        chisq_iffa = np.square((model_iffa(self.nu_arr, self.fit_iffa[0],self.fit_iffa[1],self.fit_iffa[2]) -self.s_nu)/(self.e_s_nu))
        chisq_ssa = np.square((model_ssa(self.nu_arr, self.fit_ssa[0],self.fit_ssa[1],self.fit_ssa[2]) -self.s_nu)/(self.e_s_nu))
        print(np.sqrt(chisq_effa))
        print(np.sqrt(chisq_iffa))
        print(np.sqrt(chisq_ssa))
        mystr = 'EFFA= '+str(np.sum(np.sqrt(chisq_effa)))+'\n'
        mystr += 'IFFA= '+str(np.sum(np.sqrt(chisq_iffa)))+'\n'
        mystr += 'SSA= '+str(np.sum(np.sqrt(chisq_ssa)))+'\n'


        # Line to plot the model
        self.s1 = EFFA_func(self.nu_mod, self.fit_effa[0],self.fit_effa[1], self.fit_effa[2])
        self.s2 = IFFA_func(self.nu_mod, self.fit_iffa[0],self.fit_iffa[1], self.fit_iffa[2])
        self.s3 = SSA_func(self.nu_mod, self.fit_ssa[0],self.fit_ssa[1], self.fit_ssa[2])
        
        self.l1, = ax.plot(self.nu_mod, self.s1, 'blue', linestyle = '-', label='EFFA')
        self.l2, = ax.plot(self.nu_mod, self.s2, 'red', linestyle = '-.', label='IFFA')
        self.l3, = ax.plot(self.nu_mod, self.s3, 'green', linestyle = ':', label='SSA')
        ax.legend(loc = 'upper right')
        ax.text(0.05,0.1, mystr, wrap=True )
            # Sliders
           
        salpha.on_changed(self.update)
        ss0.on_changed(self.update)
        snu.on_changed(self.update)
        
        button.on_clicked(self.reset)
        radio.on_clicked(self.modelfunc)
        ref_button.on_clicked(self.refit)
        bsave.on_clicked(self.save_values)
        brepl.on_clicked(self.replot)
    
        #Fit prameters
        mystring = 'Fitted Parameters \n'
        mystring += '\n EFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_effa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_effa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_effa[2])+'\n'
        mystring += '\n IFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_iffa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_iffa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_iffa[2])+'\n'
        mystring += '\n SSA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_ssa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_ssa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_ssa[2])+'\n'
        info = infoax.text(0.05,0.05,mystring, wrap=True ) 
        plt.show()
    
    
    
    def update(self,val):
        alpha = salpha.val
        s0 = 10**ss0.val
        nu_t = 10**snu.val
        if radio.value_selected=='EFFA':
            s_up = EFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l1.set_ydata(s_up)
        if radio.value_selected=='IFFA':
            s_up = IFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l2.set_ydata(s_up)
        if radio.value_selected=='SSA':
            s_up = SSA_func(self.nu_mod, s0, alpha, nu_t)
            self.l3.set_ydata(s_up)             
        if radio.value_selected=='All':
            s_up = EFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l1.set_ydata(s_up)
            s_up = IFFA_func(self.nu_mod, s0, alpha, nu_t)
            self.l2.set_ydata(s_up)
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
        if radio.value_selected=='EFFA':
            self.l1.set_ydata(self.s1)
        if radio.value_selected=='IFFA':
            self.l2.set_ydata(self.s2)
        if radio.value_selected=='SSA':
            self.l3.set_ydata(self.s3)           
        fig.canvas.draw_idle()

     
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
        label=radio.value_selected
        print (label+'\n')
        print('Refitting with new guess values\n')
        
        print("Click anywhere in the plot to selecr turnover frequency .")
        print("Press q to stop")
        # set up the key-press and mouse-click event watcher
        #clicker = fig.canvas.mpl_connect('button_press_event',
        #                                      self.onclick)       
        presser = fig.canvas.mpl_connect('key_press_event',
                                             self.onkeypress)       
        while True:
            # this function returns True if a key was pressed,
            # False if a mouse button was clicked, and None
            # if neither happened within timeout
            keypressed = fig.waitforbuttonpress()
            if keypressed and self.keypress =='c':
                nu_peak= 10**snu.val
                s0 = 10**ss0.val
                alpha = salpha.val
                label=radio.value_selected
                guess_par=[s0, alpha, nu_peak]
                if label=='EFFA':
                    self.fit_effa,cov1 = scipy.optimize.curve_fit(model_effa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)   
                    ss0.set_val(np.log10(self.fit_effa[0]))
                    snu.set_val(np.log10(self.fit_effa[2]))
                    salpha.set_val(self.fit_effa[1])
                    print ('Func: '+label+'\n'+'Guess Values: ')
                    print(guess_par)

                if label=='IFFA':
                    self.fit_iffa,cov2 = scipy.optimize.curve_fit(model_iffa, self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
                    ss0.set_val(np.log10(self.fit_iffa[0]))
                    snu.set_val(np.log10(self.fit_iffa[2]))
                    salpha.set_val(self.fit_iffa[1])
                    print ('Func: '+label+'\n'+'Guess Values: ')
                    print(guess_par)


                if label=='SSA':
                    self.fit_ssa, cov3 = scipy.optimize.curve_fit(model_ssa,  self.nu_arr, self.s_nu, guess_par, self.e_s_nu)        
                    ss0.set_val(np.log10(self.fit_ssa[0])) 
                    snu.set_val(np.log10(self.fit_ssa[2]))
                    salpha.set_val(self.fit_ssa[1])
                    print ('Func: '+label+'\n'+'Guess Values: ')
                    print(guess_par)
                if keypressed  and self.keypress =='q' :
                    break
                mystring = 'Fitted Parameters \n'
                mystring += '\n EFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_effa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_effa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_effa[2])+'\n'
                mystring += '\n IFFA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_iffa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_iffa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_iffa[2])+'\n'
                mystring += '\n SSA: \n'+  r'$S_0$ = {:.2f}'.format(self.fit_ssa[0])+' \n'+r'$\alpha$= {:.2f}'.format(self.fit_ssa[1])+' \n'+ r'$\nu_p$ {:.2f} GHz'.format(self.fit_ssa[2])+'\n'
                infoax.clear()
                info = infoax.text(0.05,0.05,mystring, wrap=True ) 
                print (mystring)
                fig.canvas.draw_idle()

        # kill the event watchers
        #fig.canvas.mpl_disconnect(clicker)
        fig.canvas.mpl_disconnect(presser)
        
    def replot(self,event):
        label=radio.value_selected
        print (label+'\n')
        print('Replot with new guess values\n')
        
        print("Click anywhere in the plot to selecr turnover frequency .")
        print("Press q to stop")
        # set up the key-press and mouse-click event watcher
        #clicker = fig.canvas.mpl_connect('button_press_event',
        #                                      self.onclick)       
        presser = fig.canvas.mpl_connect('key_press_event',
                                             self.onkeypress)       
        while True:
            # this function returns True if a key was pressed,
            # False if a mouse button was clicked, and None
            # if neither happened within timeout
            keypressed = fig.waitforbuttonpress()
            if keypressed and self.keypress =='r':
                nu_peak= self.clickx_data
                S_nu_peak = self.clicky_data
                label=radio.value_selected
                if label=='EFFA':
                    alpha = self.fit_effa[1]
                    s0= S_nu_peak/(nu_peak**alpha*np.exp(-1.0))
                    
                    #guess_par=[s0, alpha, nu_peak]
                    #s1 = EFFA_func(self.nu_mod, guess_par[0],guess_par[1], guess_par[2])
                    #self.l1.set_ydata(s1)
                    ss0.set_val(np.log10(s0))
                    snu.set_val(np.log10(nu_peak))
                    salpha.set_val(alpha)
                    
                    

                if label=='IFFA':
                    alpha = self.fit_iffa[1]                   
                    s0 = S_nu_peak/(nu_peak**alpha*(1-np.exp(-(1.0)**(-2.1))))
                    ss0.set_val(np.log10(s0))
                    snu.set_val(np.log10(nu_peak))
                    salpha.set_val(alpha)

                    #snu.set_val(np.log10(S_nu_peak))

                    #guess_par=[s0, alpha, nu_peak]
                    #s2 = IFFA_func(self.nu_mod, guess_par[0], guess_par[1], guess_par[2])
                    #self.l2.set_ydata(s2)


                if label=='SSA':
                    tau=1
                    alpha = self.fit_ssa[1]
                    s0= S_nu_peak/(1-np.exp(-tau))
                    ss0.set_val(np.log10(s0))
                    snu.set_val(np.log10(nu_peak))
                    salpha.set_val(alpha)
                    #snu.set_val(np.log10(S_nu_peak))
                    
                    #guess_par=[s0, alpha, nu_peak]
                    #s3 = SSA_func(self.nu_mod, guess_par[0], guess_par[1], guess_par[2])
                    #self.l3.set_ydata(s3)
                   # print(guess_par)
                    
                if label=='All':
                    s0= S_nu_peak/(nu_peak**alpha*np.exp(-1.0))
                    ss0.set_val(np.log10(s0))
                    snu.set_val(np.log10(nu_peak))
                    salpha.set_val(alpha)

                    #snu.set_val(np.log10(S_nu_peak))
                   
                if keypressed  and self.keypress =='q' :
                    break
                
                fig.canvas.draw_idle()
                

        # kill the event watchers
        #fig.canvas.mpl_disconnect(clicker)
        fig.canvas.mpl_disconnect(presser)




        #Fit prameters
        
    def save_values(self, event):
        label=radio.value_selected
        ind = fitTab.loc[self.name].index

        if label=='EFFA':
            fitTab[ind]['EFFA_s0']=self.fit_effa[0]
            fitTab[ind]['EFFA_alpha']=self.fit_effa[1]
            fitTab[ind]['EFFA_nup']=self.fit_effa[2]
        if label=='IFFA':
            fitTab[ind]['IFFA_s0']=self.fit_iffa[0]
            fitTab[ind]['IFFA_alpha']=self.fit_iffa[1]
            fitTab[ind]['IFFA_nup']=self.fit_iffa[2]
        if label=='SSA':
            fitTab[ind]['SSA_s0']=self.fit_ssa[0]
            fitTab[ind]['SSA_alpha']=self.fit_ssa[1]
            fitTab[ind]['SSA_nup']=self.fit_ssa[2]
        ascii.write(fitTab, output='GuessPar_Radiofits.csv', format='csv', overwrite=True)

        
        
        
    def onclick(self,event):
        """
        Handle mouse click event
        """
        if not event.inaxes:
            # skip if event happens outside plot
            return
        print("Mouse button {0} pressed at ({1:.2f},{2:.2f})".format(event.button,event.xdata,event.ydata))
        #self.keypress.append(None) # no key press
        #self.clickbutton.append(event.button)
        #self.clickx_data.append(event.xdata)
        #self.clicky_data.append(event.ydata)

    def onkeypress(self,event):
        """
        Handle key press event
        """
        if not event.inaxes:
            # skip if event happens outside plot
            return
        print("Key {0} pressed at ({1:.2f},{2:.2f})".format(event.key,event.xdata,event.ydata))
        self.keypress = event.key
        #self.clickbutton.append(None) # no mouse click
        self.clickx_data = event.xdata
        self.clicky_data = event.ydata
        
        
        
 
    def to_selection(self,event):
        """
        Here is where we will do stuff with the plot
        """
        print("Click anywhere in the plot to selecr turnover frequency .")
        print("Press q to stop")
        # set up the key-press and mouse-click event watcher
        clicker = fig.canvas.mpl_connect('button_press_event',
                                              self.onclick)       
        presser = fig.canvas.mpl_connect('key_press_event',
                                             self.onkeypress)       
        while True:
            # this function returns True if a key was pressed,
            # False if a mouse button was clicked, and None
            # if neither happened within timeout
            keypressed = fig.waitforbuttonpress()
            if keypressed and self.keypress =='r':
                nu_peak= self.clickx_data
                S_nu_peak = self.clicky_data
                alpha = salpha.val
                label=radio.value_selected
                if label=='EFFA':
                    s0= S_nu_peak/(nu_peak**alpha*np.exp(-1.0))
                if label=='IFFA':
                    s0 = S_nu_peak/(nu_peak**alpha*(1-np.exp(-(1.0)**(-2.1))))
                if label=='SSA':
                    tau=1
                    s0= S_nu_peak/(1-np.exp(-tau))
                self.refit([s0, alpha, nu_peak])
            # stop if a key is pressed and that key is "q"
            if keypressed  and self.keypress =='q' :
                break
        # kill the event watchers
        fig.canvas.mpl_disconnect(clicker)
        fig.canvas.mpl_disconnect(presser)
    
            

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



i = 0
obj_pl = Plot_class()
obj_pl.plotting(i)
callback = Index()
bnext.on_clicked(callback.next)
bprev.on_clicked(callback.prev)

#bcursor.on_clicked(obj_pl.to_selection)
#bsave.on_clicked(obj_pl.save_par)



