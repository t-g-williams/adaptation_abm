'''
convergence analysis
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import model.model as mod
from . import analysis_shock
import imp
import copy
import code
import tqdm
import numpy as np
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import logging
logger = logging.getLogger('sLogger')

def convergence_analysis(exp_name, inp_base, adap_scenarios, ncores):
    '''
    determine how many replications are required for a given level of accuracy
    do it for a bunch of different P(cc>ins) outcomes and choose the highest
    '''
    load = True
    # convergence settings
    epsilon = 0.01 # acceptance threshold for CV difference NOT CURRENTLY USED
    crit_val = 0.05 # for absolute error
    n_times = 5
    # simulation settings
    nreps = 1000
    shock_mags = [0.2]
    shock_times = [5,10,20] # measured after the burn-in period
    T_res = [1,3,5,7,9,11,13] # how many years to calculate effects over
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    outcomes = ['income']

    #### RUN THE MODELS ####
    exp_name_res = exp_name + '/convergence'
    results, results_baseline = analysis_shock.run_shock_sims(exp_name_res, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
    bools = results_baseline['cover_crop'].loc[(outcomes[0])] < results_baseline['insurance'].loc[(outcomes[0])]

    #### process the results ####
    # calculate estimated probabilities as a function of rep number
    ests = bools.copy().astype(float)
    sds = bools.copy().astype(float)
    for r in range(nreps):
        d_r = bools.query('rep<={}'.format(r))
        d_r2 = bools.query('rep=={}'.format(r))
        probs_r = d_r.astype(int).groupby(level=[0,1,2]).mean()
        sds_r = (d_r.astype(int)+1).groupby(level=[0,1,2]).std() # add 1 to these to stabilize CoV estimates
        ests.loc[d_r2.index] = np.array(probs_r.astype(float))
        sds.loc[d_r2.index] = np.array(sds_r.astype(float))
    
    # create figure
    covs = []
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    for mag in shock_mags:
        mag_str = str(mag).replace('.','_')
        for t_shock in shock_times:
            for t_res in T_res:
                tmp = ests.query('mag=="{}" and assess_pd=={} and time=={}'.format(mag_str, t_res, t_shock))
                rep_means = tmp.mean(axis=1)
                axs[0].plot(np.array(rep_means))
                cov = np.array(sds.loc[rep_means.index].mean(axis=1)) / (1+np.array(rep_means))
                axs[1].plot(cov) # divide by 1+mean for stability
                covs.append(cov)
    
    for ax in axs:
        ax.set_xlabel('Number of replications')
        ax.grid(False)
    axs[0].set_ylabel('Estimate')
    axs[0].set_title('Estimate convergence')
    axs[1].set_title('CoV convergence')
    axs[1].set_ylabel('Coefficient of variation')
    fig.savefig('../outputs/{}/plots/convergence_lines.png'.format(exp_name))
    plt.close('all')

    # calculate COV differences
    # following ralha2018
    # covs = np.array(covs)
    # cov_diffs = covs[:, 1:] - covs[:, :-1]
    # cov_diffs[np.isnan(cov_diffs)] = 100
    # cov_diffs_bool = copy.copy(cov_diffs).astype(bool)
    # cov_diffs_bool[cov_diffs>epsilon] = False
    # cov_diffs_bool[cov_diffs<=epsilon] = True

    ## figure of error relative to final estimate
    fig, ax = plt.subplots(figsize=(7,5))
    abs_errors = []
    final_ests = ests.query('rep=={}'.format(nreps-1))
    for mag in shock_mags:
        mag_str = str(mag).replace('.','_')
        for t_shock in shock_times:
            for t_res in T_res:
                query_str = 'mag=="{}" and assess_pd=={} and time=={}'.format(mag_str, t_res, t_shock)
                tmp = ests.query(query_str)
                final_est = np.array(final_ests.query(query_str).mean(axis=1))
                rep_means = np.array(tmp.mean(axis=1))
                abs_error = np.abs(rep_means - final_est)
                abs_errors.append(abs_error)
                ax.plot(abs_error, color='k', lw=0.3)

    # find critical value
    ax.axhline(y=crit_val, color='r', lw=1.2)
    abs_errors = np.array(abs_errors)
    abs_bool = abs_errors < crit_val
    nreps_req = np.where(abs_bool==False)[1].max()
    ax.axvline(x=nreps_req, color='r', lw=1.2)
    ax.text(0.95, 0.95, '{} replications required\nfor absolute error < {}'.format(nreps_req, crit_val), transform=ax.transAxes, va='top',ha='right')

    ax.set_xlabel('Number of replications')
    ax.set_ylabel('Absolute error')
    ax.grid(False)
    ax.set_ylim([0,0.4])
    fig.savefig('../outputs/{}/plots/convergence_absolute_error.png'.format(exp_name))

    return nreps_req