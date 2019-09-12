'''
pattern-oriented modeling for calibration of the ABM
'''
import numpy as np
import pandas as pd
import code
import pyDOE
import pickle
import os
import copy
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from plot import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

from model import base_inputs
from model import model
import plot.single_run as plt_single


def main():
    # specify experimental settings
    N_samples = 10000
    ncores = 2
    inputs = {
        'model' : {'n_agents' : 100, 'T' : 100, 'exp_name' : 'POM',
                    'adaptation_option' : 'none'}
    }

    # define the variables for calibration
    calib_vars = pd.DataFrame(
        # id, key1, key2, min, max
        [[1, 'land', 'rain_cropfail_low_SOM', 0, 0.5],
        [2, 'land', 'mineralization_rate', 0.05, 0.5],
        [3, 'land', 'wealth_N_conversion', 0.01, 0.5],
        [4, 'land', 'livestock_frac_crops', 0, 1],
        [5, 'land', 'residue_CN_conversion', 25, 100],
        [6, 'agents', 'cash_req_mean', 10000, 30000]],
        columns = ['id','key1','key2','min_val','max_val'])

    # generate set of RVs
    rvs = hypercube_sample(N_samples, calib_vars)

    # run the model and calculate fitting metrics
    fits = run_model(rvs, inputs, calib_vars, ncores)
    
    # process the fit data
    process_fits(fits, rvs, calib_vars, inputs)

def fitting_metrics(mod):
    '''
    determine whether the model displays the desired patterns
    '''
    n_yrs = 10 # how many years to do calculations over (from the end of the simulation)
    ag1 = mod.agents.n_plots == mod.agents.n_plots_init[0]
    ag2 = mod.agents.n_plots == mod.agents.n_plots_init[1]
    ag3 = mod.agents.n_plots == mod.agents.n_plots_init[2]
    
    ## 1. agent wealth
    ## we want agent type 1 (lowest land) to have -ve wealth with certainty
    ## and agent type 3 (highest land) to not have -ve wealth
    ## and agent type 2 to be somewhere in the middle
    p1 = True if np.mean(mod.agents.cant_cope[-1, ag1]) == 1 else False
    v2 = np.mean(mod.agents.cant_cope[-1, ag2])
    p2 = True if (v2 > 0.2) and (v2 < 0.8) else False
    p3 = True if np.mean(mod.agents.cant_cope[-1, ag3]) == 0 else False
    fit1 = bool(p1 * p2 * p3)

    ## 2a. yield nutrient effects -- no "dead" soil
    vals = mod.land.nutrient_factors[-n_yrs:]
    if np.sum(~np.isnan(vals)) == 0:
        fit2a = False
    else:
        fit2a = np.nanmin(vals) > 0.01

    ## 2b. every year there is some nutrient limitation
    fit2b = True
    for y in range(n_yrs):
        if fit2b:
            vals = mod.land.nutrient_factors[-y]
            if np.sum(~np.isnan(vals)) != 0:
                min_y = np.nanmin(vals)
                fit2b = True if min_y<1 else False

    ## 3. soil organic matter
    ## not consistently someone at maximum value
    maxs = np.max(mod.land.organic[-n_yrs:], axis=1)
    fit3 = False if all(maxs == maxs[-1]) else True

    return [fit1, fit2a, fit2b, fit3]

def hypercube_sample(N, calib_vars):
    '''
    create latin hypercube sample
    '''
    # generate uniform vars
    np.random.seed(0)
    rvs_unif = pyDOE.lhs(calib_vars.shape[0], N)

    # scale
    rvs = np.array(calib_vars.min_val)[None,:] + rvs_unif * \
            np.array(calib_vars.max_val - calib_vars.min_val)[None,:]

    return rvs

def run_model(rvs, inputs, calib_vars, ncores=1):
    '''
    run the model for each of the RV combinations
    '''
    # create the full set of inputs
    inp_all = base_inputs.compile()
    inp_all = overwrite_inputs(inp_all, inputs)
    outdir = '../outputs/{}/'.format(inputs['model']['exp_name'])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # run the model
    N = rvs.shape[0]
    sim_chunks = chunkIt(np.arange(N), ncores)
    if ncores > 1:
        fits_par = Parallel(n_jobs=ncores)(delayed(run_chunk_sims)(sim_chunks[i], rvs, inp_all, calib_vars) for i in range(len(sim_chunks)))
        fits = {}
        for fit in fits_par:
            for k, v in fit.items():
                fits[k] = v
    else:
        fits = run_chunk_sims(sim_chunks[0], rvs, inp_all, calib_vars)

    return fits

def run_chunk_sims(ixs, rvs, inp_all, calib_vars):
    '''
    run the "ixs" rows of rvs simulations
    '''
    inp_all = copy.deepcopy(inp_all) # just in case there are parallel issues with ids
    fits = {}

    with tqdm(ixs, disable = not True) as pbar:
        for ix in ixs:
            # initialize the inputs
            inp_all = overwrite_rv_inputs(inp_all, rvs[ix], calib_vars.key1, calib_vars.key2)
            
            # run the model
            m = model.Model(inp_all)
            for t in range(m.T):
                m.step()

            # calculate model fitting metrics
            fits[ix] = fitting_metrics(m)
            pbar.update()

    return fits

def process_fits(fits, rvs, calib_vars, inputs):
    '''
    process the POM results
    '''
    outdir = '../outputs/POM/'
    fit_pd = pd.DataFrame.from_dict(fits, orient='index')
    fit_pd['sum'] = fit_pd.sum(axis=1)

    vals = fit_pd['sum'].value_counts()

    # histogram of sums
    N = rvs.shape[0]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(fit_pd['sum'])
    ax.set_xlabel('Number of patterns matched')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(len(fits[0])+1))
    ax.set_xticklabels(np.arange(len(fits[0])+1).astype(int))
    ax.grid(False)
    for i in vals.index:
        ax.text(i, vals.loc[i]+1, str(vals.loc[i]), horizontalalignment='center')
    fig.savefig(outdir + 'histogram_{}.png'.format(N))

    # write outputs
    fit_pd.to_csv(outdir + 'fits_{}.csv'.format(N))
    np.savetxt(outdir + 'rvs_{}.csv'.format(N), rvs)

    # identify the best fitting models
    max_fit = vals.index[-1]
    max_fit_locs = fit_pd.loc[fit_pd['sum']==max_fit, 'sum'].index

    # plot their parameter values
    vals_best = rvs[max_fit_locs]
    vals_sc = (vals_best - np.array(calib_vars['min_val'])[None,:]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,:]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(np.transpose(vals_sc))
    ax.set_xticks(np.arange(calib_vars.shape[0]))
    ax.set_xticklabels(calib_vars.key2, rotation=90)
    ax.set_ylabel('Value (scaled)')
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    fig.savefig(outdir + 'param_vals.png')

    # run and plot one of them
    inp_all = base_inputs.compile()
    inp_all = overwrite_inputs(inp_all, inputs)
    inp_all = overwrite_rv_inputs(inp_all, rvs[max_fit_locs[0]], calib_vars.key1, calib_vars.key2)
    m = model.Model(inp_all)
    for t in range(m.T):
        m.step()
    plt_single.main(m)
    fits = fitting_metrics(m)

    # save the inputs -- as csv and pickle
    df = pd.DataFrame.from_dict({(i,j): inp_all[i][j] 
                               for i in inp_all.keys() 
                               for j in inp_all[i].keys()},
                           orient='index')
    df.to_csv(outdir + 'input_params_{}.csv'.format(N))
    with open(outdir + 'input_params_{}.pkl'.format(N), 'wb') as f:
        pickle.dump(inp_all, f)
    code.interact(local=dict(globals(), **locals()))

def overwrite_inputs(all_inputs, changes):
    '''
    place the elements in "changes" in the "all_inputs" dictionary
    '''
    all_inp = copy.deepcopy(all_inputs)
    for k, v in changes.items():
        for k2, v2 in v.items():
            all_inp[k][k2] = v2
    return all_inp

def overwrite_rv_inputs(all_inputs, rv_inputs, names1, names2):
    for i, rv in enumerate(rv_inputs):
        all_inputs[names1[i]][names2[i]] = rv
    return all_inputs

def chunkIt(seq, n):
    '''
    Split a list into n parts
    From: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    '''
    avg = len(seq) / float(n)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == '__main__':
    main()