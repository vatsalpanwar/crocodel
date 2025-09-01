#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import astropy.io.fits
import yaml
import sys 
import dynesty
import emcee 
import time
from shutil import copyfile
import corner
import datetime
import os
from multiprocessing import Pool
import pymultinest
from subprocess import call
import json

from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import cross_correlation_utils as crocut
from crocodel.crocodel import data
from crocodel.crocodel import model

import glob
import distinctipy


##############################################################################
### Read in the path to the directory where the outputs of multinest are saved
##############################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-wdir','--workdir', help = "Path to the working directory where all the outputs of the multinest to be post processed are saved.",
                    type=str, required=True)
args = vars(parser.parse_args())
savedir = args['workdir']

KpVsys_savedir = savedir + 'KpVsys_maps/'

config_file_path = savedir+ 'croc_config.yaml'

with open(config_file_path) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Plot the Kp-Vsys maps for the Retrieved model ##########
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
sp_list = {
    'all':'All species',
    'h2o':'H$_2$O', 
    'co':'CO', 
    'oh':'OH', 
    'tio':'TiO', 
    'mgo':'MgO', 
    'fe':'Fe', 
    'ti':'Ti',
    'ca': 'Ca',
    'al': 'Al',
    'cr': 'Cr',
    'v': 'V',
    'si': 'Si',
    'mg': 'Mg'
    
}

KpVsys_maps = {}

for sp in sp_list:
    if sp == 'all':
        KpVsys_maps[sp] = np.load(KpVsys_savedir + 'KpVsys_fast_no_model_reprocess_dict.npy', allow_pickle = True).item()
    else:
        KpVsys_maps[sp] = np.load(KpVsys_savedir  + 'KpVsys_fast_no_model_reprocess_dict_without_' + sp + '.npy', allow_pickle = True).item()
        
Kp_exp = config['model']['Kp']
Vsys_exp = config['model']['Vsys']
Kp_exp_err = 0
Vsys_exp_err = 0


fig, axes = plt.subplots(5, 3, figsize=(18, 30))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# import pdb; pdb.set_trace()

for ax, (sp, sp_name) in zip(axes.flatten(), sp_list.items()):
    
    KpVsys_dict = KpVsys_maps[sp]
    if sp != 'all':
        # logL_total_sigma_all = crocut.get_sigma_contours(logL_KpVsys=KpVsys_maps['all']['total']['logL'], dof=2)
        # logL_total_sigma_without_sp = crocut.get_sigma_contours(logL_KpVsys=KpVsys_maps[sp]['total']['logL'], dof=2)
        # logL_total_sigma = logL_total_sigma_all - logL_total_sigma_without_sp ## Old way, not correct. 
        # logL_total = KpVsys_maps['all']['total']['logL'] - KpVsys_maps[sp]['total']['logL']
        # logL_total_sigma = crocut.get_sigma_contours(logL_KpVsys=logL_total, dof=2)
        KpVsys_diff = KpVsys_maps['all']['cc'] - KpVsys_maps[sp]['cc']
        
    else:
        KpVsys_diff = KpVsys_maps['all']['cc']
         
    hnd1 = crocut.subplot_cc_matrix(axis=ax,
                                    cc_matrix=KpVsys_diff,
                                    phases=KpVsys_dict['Kp_range'],
                                    velocity_shifts=KpVsys_dict['Vsys_range_windowed'],
                                    title=sp_name,
                                    setxlabel=True, 
                                    plot_type='pcolormesh',
                                    cmap='viridis')
    
    cb = fig.colorbar(hnd1, ax=ax, pad=0.01)
    # cb.ax.set_ylabel('N$\sigma$', fontsize=MEDIUM_SIZE, rotation=270, labelpad=15)
    cb.ax.set_ylabel('N$\sigma$', fontsize=20, rotation=270, labelpad=15)
    
    ax.axvline(x=Vsys_exp, color='w', linestyle='dashed')
    ax.axvspan(xmin=Vsys_exp - Vsys_exp_err, xmax=Vsys_exp + Vsys_exp_err, color='w', alpha=0.2)
    ax.axhline(y=Kp_exp, color='w', linestyle='dashed')
    ax.axhspan(ymin=Kp_exp - Kp_exp_err, ymax=Kp_exp + Kp_exp_err, color='w', alpha=0.2)
    
    ax.set_ylabel('K$_{\mathrm{p}}$ [km/s]')
    ax.set_xlabel(r'V$_{\mathrm{sys}}$ [km/s]')

plt.savefig(savedir + 'KpVsys_maps_retrieved_model_all_species.png', bbox_inches='tight', dpi = 300)
# plt.show()


