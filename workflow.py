'''
workflow for convergence, sensitivity analysis etc
'''

import os
import calibration.POM as POM
import experiments.convergence as convergence
import imp
import copy
import code
import numpy as np
import pandas as pd
import pickle
import copy

def main():
    exp_name = 'test'
    pom_nreps = 2
    pom_nsim = 100

    ## 1. select models from POM
    # assume the POM has been run already
    params = load_POM_params(exp_name, pom_nsim, pom_nreps)
    print('{} model(s) from POM'.format(len(params)))

    ## 2. convergence analysis
    nsim_req = convergence.main(params)
    code.interact(local=dict(globals(), **locals()))

def load_POM_params(exp_name, pom_nsim, pom_nreps):
    '''
    select all parameter sets within (1-threshold)*100 percent
    of the best model
    the threshold is specified in the POM code.
    '''
    outdir = '../outputs/{}/POM/{}_{}reps/'.format(exp_name, pom_nsim, pom_nreps)
    params = []
    for root, dirs, files in os.walk(outdir):
        for fnam in files:
            if ('input_params_' in fnam) and ('.pkl' in fnam):
                with open(outdir + fnam, 'rb') as f:
                    params.append(pickle.load(f))

    return params

if __name__ == '__main__':
    main()