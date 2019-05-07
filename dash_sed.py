#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:17:31 2019

@author: pallavipatil
"""

from astropy.io import fits,ascii
import matplotlib.pyplot as plt
from astropy.table import Table, unique, join
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from file_prep_radio_fitting import data_prep
import aplpy as apl
from astropy.visualization import ZScaleInterval
import scipy.optimize
import pandas as pd
import Radio_Models_func as rm
import plotly
import plotly.plotly as py
import plotly.graph_objs as go 
import plotly.io as pio
from datetime import datetime
from ipywidgets import interactive, HBox, VBox, widgets, interact
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

plottab = Table.read('SED_plots_data.csv', format = 'ascii.csv', delimiter=',')
final_unq_jvla = Table.read('Final_quants_cal.csv')
final_unq_jvla.add_index('Source_name')
fit_res = Table.read('fit_results_allmodels.csv')
fit_res.add_index('Source_name')

vla_ax = Table.read('../VLA-NewWork/JMFIT_CASA_A_results.csv', format = 'csv', delimiter = ',')
vla_bx = Table.read('../VLA-NewWork/JMFIT_CASA_B_results.csv', format = 'csv', delimiter = ',')
    
vla_ax.add_index('Source_name')
vla_bx.add_index('Source_name')

         
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
       
nu_mod = np.logspace(-3, 3.0, 5000)

#Load Data:
def load_data(ii):
    global plottab
    source = plottab['Source_name'][ii]
    nu =eval(plottab['radio_nu_arr'][ii])
    fnu = eval(plottab['radio_flux_arr'][ii])
    e_fnu = eval(plottab['radio_eflux_arr'][ii])
    labels = eval(plottab['radio_cols'][ii])
    return source, nu, fnu, e_fnu, labels


def prep_model_fit_data(ii,IB_flag):
    global plottab
    source, nu, fnu, e_fnu, labels = load_data(ii)  
    alphas = [eval(plottab['alpha_AX'][ii]) ,eval(plottab['alpha_BX'][ii]),
             eval(plottab['alpha_GL'][ii]) ]
    if IB_flag:
        for alph in alphas:
            if len(alph) >1:
                nu.append(alph[0])
                fnu.append(alph[1])
                e_fnu.append(alph[2])
    return nu, fnu, e_fnu



def model_fitting(ii, func_model, model_name, input_par):
    nu, fnu, e_fnu = prep_model_fit_data(ii,False)
    if input_par == None:
        s0 = np.max(fnu)
        nu_t = np.min(np.abs(nu))
        alpha = -0.7       
        if model_name =='PL':
            input_par = [s0, alpha]
        else:
            input_par = [s0, alpha,nu_t]    

    #### Fitting Routine #########
    try:
        fit_mod,cov1 = scipy.optimize.curve_fit(func_model,  
                                nu, fnu, input_par, e_fnu)   
        perr_mod = np.sqrt(np.diag(cov1))
    except:
        fit_mod = [-999,-999,-999]       
    return fit_mod

def generate_model_data(ii):
    global nu_mod
    global plottab
    global fit_res
    source = plottab[ii]['Source_name']
    models = ['PL', 'CPL', 'EFFA', 'SSA']
    fun_models = [rm.model_pl, rm.model_cpl, rm.model_effa, rm.model_ssa]
    data_calc = [rm.power_law, rm.curved_power, rm.EFFA_func, rm.SSA_func]
    model_data = []
    model_label = []
    fit_pars = fit_res.loc[source]
    for model_name, func,calc_f in zip(models, fun_models, data_calc):
        if model_name  =='PL':
            keys = [model_name+'_S0', model_name+'_alpha']
        elif model_name =='CPL':
            keys = [model_name+'_S0', model_name+'_alpha', model_name+'_q']
        else:
            keys = [model_name+'_S0', model_name+'_alpha', model_name+'_nup']
        input_par = list(fit_pars[keys])     
        fit_mod = model_fitting(ii,func, model_name, input_par)
        print(fit_mod, model_name)
        if fit_mod[0] > -999:
            model_data.append(calc_f(nu_mod, *fit_mod))
            model_label.append(model_name)
    return model_data, model_label, fit_mod

def generate_tab_data(ii):
    global plottab, vla_ax, vla_bx
    global fit_res
    global final_unq_jvla
    #keys
    source = plottab[ii]['Source_name']
    keys_unq = ['Source_name', 'Morphology', 'Spectral Shape', 'Final_Flux',
                'Final_Flux_err', 'SSize','SSErr', 'nu_p', 'nu_p_err', 'alpha_thin', 'alpha_thin_err', 
                'Final_al_IB', 'Final_al_IB_err', 'LinS', 'LinS_l', 'Plobe',
                'LNVSS']
    keys_vla = ['region', 'P_flux', 'pflux_err', 'I_flux', 'iflux_err', 
                'SpIdx', 'SpIdx_err']
    unq_info = final_unq_jvla.loc[source][keys_unq]
    ax_info = vla_ax.loc[source][keys_vla]
    bx_info = vla_bx.loc[source][keys_vla]
    return keys_vla, keys_unq, unq_info, ax_info, bx_info
    

def prep_tab_rows(ax_info, bx_info, ii, z):
    global plottab
    source = plottab[ii]['Source_name']
    tab_values = []
    if len(ax_info)<5:
        for row in ax_info:
            tab_values.append([source,'AX',z]+list(row))
    else:
        tab_values.append([source,'AX',z]+list(ax_info))
    if len(bx_info)<5:
        for row in bx_info:
            tab_values.append([source,'BX',z]+list(row))
    else:
        tab_values.append([source,'BX',z]+list(bx_info))
    cols = ['Source','config','z', 'Reg', 'P_flux', 'pflux_err', 'I_flux', 'iflux_err',
            'SpIdx', 'SpIdx_err']
    df = pd.DataFrame(data=tab_values, columns=cols)
    return df
    
    
#header = dict(values = [['Source'], ['Morph'],['Sp_Shape'],[r'$S_f$'],[r'$S_{f,err}$'],
#                                [r'$\nu_p$'],[r'$\nu_{p,err}$'],[r'$\alpha_{thin}$'],
#                                [r'$\alpha_{thin, err}$'],[r'$\alpha_{IB}$'],
#                                [r'$\alpha_{IB, err}$']]
#                    ),


def prep_plotting(ii):
    global plottab
    nu_mod = np.logspace(-3, 3.0, 5000)
    source, nu, fnu, e_fnu, labels = load_data(ii)
    model_data, model_label,fit_mod = generate_model_data(ii)
    keys_vla, keys_unq, unq_info, ax_info, bx_info = generate_tab_data(ii)
    z = plottab[ii]['z']
    img_flg = False
    img = '../cutouts/'+source+'_reg1.png'

    df = prep_tab_rows(ax_info, bx_info, ii, z)

    table_trace1 = go.Table(
        domain=dict(x=[0, 1.0],
                    y=[0.25,0.4]),
        header = dict(values = ['Source','Morph','Sp_Shape','Flux','E_flux','Ang_Size',
                                'E_Ang_Size','Nu_p','E_nu_p','al_thin',
                                'E_al_thin','al_IB','E_al_IB']
                    ),
        cells = dict(values= list(unq_info)[:13],
                     format = [None,None, None]+['.3f']*10))     

    table_trace2 = go.Table(
        domain=dict(x=[0, 1.0],
                    y=[0.0,0.15]),
        header = dict(values = list(df.columns)),
        cells = dict(values= [df[k].tolist() for k in df.columns],
                     format = [None,None, '.3f',None,'.2f', '.3f','.2f','.3f','.2f','.3f'])
        )
    fm_list = eval(unq_info[15])
    plobe = 'None'
    if z >0:
        fmt = len(fm_list)*'{:.1e}, '
        plobe = fmt.format(*fm_list)
    
    table_trace3 = go.Table(
                domain=dict(x=[0, 1.0],
                            y=[0.17,0.25]),
                header = dict(values = ['Source','z','LinS','LNVSS','Plobe']
                    ),
                cells = dict(values= [unq_info[0],z,unq_info[13],unq_info[16],
                                      plobe],
                     format = [None,'.3f','.3f','.1e',None]))  


    layout = go.Layout(  
                width=1200,
                height=800,    
                autosize = False,
                title= source,
                xaxis = dict(type='log',title='Log Frequency,(nu/GHz)',
                             range=[-1.3,1.7], **dict(domain=[0.0,1.0])),
                yaxis = dict(type='log', title='Log Flux, (S/mJy)',
                             range=[np.log10(min(fnu)/2), np.log10(max(fnu)*2)],
                             **dict(domain=[0.5,1.0]))                                                        
            )
  
    trace = [table_trace1, table_trace2,table_trace3]
    ls_mods = ['solid', 'dash', 'dot', 'dashdot']
    for name, data, ls in zip(model_label, model_data,ls_mods):
        trace.append(go.Scatter(x= nu_mod,y=data, 
                    error_y=dict(type='data', array=e_fnu, visible=True, width=3),
                    name=name, 
                    text=labels,
                    line={'width':2,
                          'dash':ls}))
    trace.append(go.Scatter(
                    x=nu ,y=fnu, 
                    error_y=dict(type='data', array=e_fnu, visible=True, width=3),
                    name='Data', 
                    text=labels,
                    mode='markers',
                    marker = {'size':7,'color':'black' }
                    
                ))
    return trace, layout


def get_my_img(ii):
    global plottab 
    source = plottab['Source_name'][ii]
    img_src = '../cutouts/'+source+'_reg1.png'
    if os.path.exists(img_src):
        return img_src
    else:
        return None
    
ii=0
mytrace, mylayout = prep_plotting(ii)
          
app.layout = html.Div(children=[
    html.H1(children='RSED',style={'textAlign':'center'}),

    html.Div(
        children=''' Radio SED Fitting: An Interactive Tool.''',style={'textAlign':'center'}),
    
    html.Div([
        html.Label('SED Models'),
            dcc.RadioItems('SEDMod',
            options=[
            {'label': 'EFFA', 'value': 'EFFA'},
            {'label': 'SSA', 'value': 'SSA'},
            {'label': 'PL', 'value': 'PL'},
            {'label': 'CPL', 'value': 'CPL'}
        ],
        value='PL')    ]),
            

    dcc.Graph(
        id='example-graph',
        figure={
            'data': mytrace,
            'layout': mylayout
        }
    ),

    html.Div([
        html.Button('Prev', id='prev', n_clicks=0),
        html.Button('Next', id='next', n_clicks=0),
        html.Button('Fit', id='fit', n_clicks_timestamp=0),
        html.Button('Reset', id='reset', n_clicks_timestamp=0),],
        style={'textAlign':'center'}),
            
    html.Img(id='image', 
                       src='/Users/pallavipatil/Desktop/VLA/cutouts/J0010+16_reg1.png')
        
])
   
n_btn1 = 0
n_btn2 = 0

@app.callback(
      Output('example-graph', 'figure'),
      inputs=[Input('next','n_clicks'),
              Input('prev','n_clicks')])
        
def update_figure(nxt,prv):
    global n_btn1
    global n_btn2
    global ii
    if nxt > n_btn1:
        ii = ii+1
        newtrace, newlayout = prep_plotting(ii)
        n_btn1 = nxt
    if prv > n_btn2:
        ii = ii-1
        newtrace, newlayout = prep_plotting(ii)
        n_btn2 = prv
    return {'data':newtrace, 
            'layout':newlayout}        
 

    
if __name__ == '__main__':
    app.run_server(debug=True)
