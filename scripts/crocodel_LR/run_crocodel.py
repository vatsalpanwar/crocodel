#!/usr/bin/env python3
"""Script to run a MultiNest sampling for high-resolution cross-correlation spectroscopy data.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from shutil import copyfile
import datetime
import os
import pymultinest
import json

# Matplotlib rcparams
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Add the path to the code snippet modules
from crocodel.crocodel_LR import data
from crocodel.crocodel_LR import model

################################################################
now = datetime.datetime.now()
# Format the date and time
d1 = now.strftime("%d-%m-%YT%H-%M-%S")
print('Date tag for this run which will be used to save the results: ', d1)

################################################################
"""Take the path to the croc_config.yaml file as the input"""
################################################################

parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-wdir','--work_dir', help = "Path to the working directory.",
                    type=str, required=False)
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=False)

args = vars(parser.parse_args())

if args['config_file_path'] is not None:
    config_file_path = args['config_file_path']
    with open(config_file_path) as f:
        config_dd = yaml.load(f,Loader=yaml.FullLoader)
    infostring = config_dd['infostring'] + '_' + d1
    savedir = config_dd['workdir']['results'] + infostring + '/'

    """Create the directory to save results."""
    try:
        os.makedirs(savedir)
    except OSError:
        savedir = savedir

    print('Saving files in directory: ', savedir)
    ### Save the config file in the savedir 
    copyfile(config_file_path, savedir + 'croc_config.yaml')
    outputfiles_basename = savedir + infostring + 'multinest_output_'
    resume = False

elif args['work_dir'] is not None:
    savedir = args['work_dir']
    config_file_path = savedir + 'croc_config.yaml'
    
    with open(config_file_path) as f:
        config_dd = yaml.load(f,Loader=yaml.FullLoader)
    
    resume = True
    directory_path = os.path.dirname(savedir)
    # Split the directory path into a list of components
    infostring = directory_path.split(os.path.sep)[-1]
    outputfiles_basename = savedir + infostring + 'multinest_output_'

################################################################
"""Define the data class first, for all instruments."""
################################################################
planet_data = data.Data(config_dd = config_dd)
################################################################
"""Define the planetary model class next."""
################################################################
planet_atmosphere = model.Model_LR(config_dd = config_dd, data = planet_data)

# INST_LIST_GLOBAL = ['nirspec_g395h', 'niriss_soss_ord2', 'niriss_soss_ord1']
INST_LIST_GLOBAL = ['nirspec_g395h']

free_param_dict = planet_atmosphere.free_param_dict
################################################################
################################################################
"""
Define Likelihood and prior functions
"""
################################################################
################################################################
def prior_transform(cube, ndim, nparams):
    ### Just do uniform priors for all parameters within their bounds given in self.bounds
    """
    u is a list of sampled points for each parameter in the range from 0. to 1. This must be transformed to the bounds for your own parameter
    ### Assuming u is in the same order as the parameters in the free_param_dict
    """

    for i, nm in enumerate(free_param_dict.keys()):
        cube[i] = free_param_dict[nm]["bound"][0] + (free_param_dict[nm]["bound"][1] - free_param_dict[nm]["bound"][0])*cube[i]

def log_likelihood_multinest(cube, ndim, nparams):
    logL_total = planet_atmosphere.logL(cube, inst_list = INST_LIST_GLOBAL)
    # try:
    #     logL_total = planet_atmosphere.logL(cube, inst_list = INST_LIST_GLOBAL)
    # except (AssertionError, ValueError):
    #     logL_total = -1e90
    # print(logL_total)
    return logL_total

################################################################
################################################################
"""
Run the MultiNest sampler
"""
################################################################
################################################################

ndim = len(free_param_dict.keys())
parameters = list(free_param_dict.keys())
globalStart = time.time()
pymultinest.run(log_likelihood_multinest, 
                prior_transform, ndim, 
                outputfiles_basename = outputfiles_basename, ## savedir + infostring + 'multinest_output_', 
                resume = resume, 
                verbose = True, n_live_points = 100, 
                multimodal = False, 
                importance_nested_sampling = False, 
                const_efficiency_mode = True,
                sampling_efficiency = 0.05,
                evidence_tolerance = 0.5,
                )
json.dump(parameters, open(savedir + infostring + 'multinest_output_' + 'params.json', 'w')) # save parameter names
globalEnd = time.time()
print('Total computation took {:5} seconds'.format(globalEnd-globalStart))
## Try changing the constant efficiency for a faster convergence 







