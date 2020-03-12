'''
Run a monte carlo experiment (POM)
and classify each parameter set into a "trajectory"
'''
import numpy as np
import pandas as pd
import code
import pickle
import os
import copy
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.transforms as transforms
from collections import OrderedDict 
from joblib import Parallel, delayed
from plot import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

from . import POM
from model import base_inputs
from model import model
# import plot.single as plt_single

def main():
    exp_name = 'trajectories_test'

    ## 1. identify the model parameterizations
    params, calib_vars = identify_models(exp_name)

    ## 2. run and plot baseline simulations
    run_baseline_sims(params, calib_vars, exp_name)

def run_baseline_sims(params, calib_vars, exp_name):
    '''
    run baseline simulations (single replication)
    and plot model-level and agent-level differences
    '''
    ### 1. change the inputs: generate list of parameterizations to run
    inputs = {'model' : {'n_agents' : 200, 'T' : 50}}
    inp_base = base_inputs.compile()
    inp_all = POM.overwrite_inputs(inp_base, inputs)
    exps = OrderedDict()
    for cat, param_array in params.items():
        for ni in range(param_array.shape[0]): # loop over the N best models for this param
            name = '{}_{}'.format(cat, ni)
            exps[name] = POM.overwrite_rv_inputs(inp_all, param_array[ni], calib_vars.key1, calib_vars.key2)
    
    ### 2. run the models
    ncores = 40
    mods = Parallel(n_jobs=ncores)(delayed(run_model)(i, exps) for i in range(len(exps)))

    ### 3. plot the results
    plot_baseline(mods, exps, exp_name)

def identify_models(exp_name_overall):
    # specify experimental settings
    N_samples = 10000
    ncores = 40
    nreps = 10
    N_per_class = 10
    exp_name = '{}/pom_{}_{}rep'.format(exp_name_overall, N_samples, nreps)
    inputs = {
        'model' : {'n_agents' : 200, 'T' : 50, 'exp_name' : exp_name}
    }

    # define the variables for sampling
    calib_vars = POM.define_calib_vars()
    calib_vars.to_csv('../outputs/{}/calib_vars.csv'.format(exp_name))

    # generate set of RVs
    rvs = POM.hypercube_sample(N_samples, calib_vars)
    outdir = '../outputs/{}'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    np.savez_compressed('{}/rvs.npz'.format(outdir), data=rvs)

    # # run the model and calculate fitting metrics
    fits = POM.run_model(rvs, inputs, calib_vars, ncores, nreps, trajectory=True, load=True)

    params = select_parameterizations(fits, rvs, exp_name, N_per_class, outdir)

    return params, calib_vars

def fitting_metrics(m):
    '''
    determine the feasibility of the model
        0 = infeasible, 1 = feasible
    and assign it to a class
        0 : s+e+, 1 : s+e-, 2 : s-e+, 3 : s-e-
        -1 if none
    '''
    #### FEASIBILITY ####
    # A: P(wage or salary labor) < 0.3
    p_work = np.mean(np.logical_or(m.agents.wage_labor > 0, m.agents.salary_labor > 0))
    feas_a = True if 0 < p_work <= 0.3 else False
    # if feas_a:
    #     # print(p_work)
    #     # print(m.agents.living_cost_pp)
        # print(m.market.wage_jobs_availability)
    # B: max(livestock) < 60
    feas_b = True if m.agents.livestock.max() < 60 else False
    # combine
    feas = int(feas_a*feas_b)

    #### TRAJECTORIES ####
    ## social outcomes
    thresh = 0.5
    p_livestock = np.mean(m.agents.livestock > 0) # over all time / agents
    p_no_coping = np.mean(~m.agents.cons_red_rqd) # over all time / agents (high values are good)
    ## environmental outcomes
    frac_val = 0.8 # times the initial value
    som_frac = np.mean(m.land.organic[-1]) / m.land.organic_N_min_init # fraction relative to initial
    stable_som = True if som_frac >= frac_val else False
    R_frac = (m.rangeland.R.min() / m.rangeland.R_max) / m.rangeland.R0_frac # fraction relative to initial fraction
    stable_R = True if R_frac >= frac_val else False
    # combine
    if np.all([p_livestock>=thresh, p_no_coping>=thresh]): # s+ outcome
        if np.all([stable_som, stable_R]): # e+ outcome
            cat = 0
        elif np.logical_or(~stable_som, ~stable_R): # e- outcome -- either one is under limit
            cat = 1
        else: # neither
            cat = -1
    else: # s- outcome -- either one is bad
        if np.all([stable_som, stable_R]): # e+ outcome
            cat = 2
        elif np.logical_or(~stable_som, ~stable_R): # e- outcome
            cat = 3
        else: # neither
            cat = -1

    # return [feas, cat]
    return [feas, cat, feas_a, feas_b, p_livestock, p_no_coping, som_frac, R_frac]

def select_parameterizations(fits, rvs, exp_name, N_per_class, outdir):
    '''
    select the N most likely parameterizations for each class
    '''
    f = fits[:,:,0:2]
    ## calculate the probability that each model belongs to each class
    cats = [0,1,2,3]
    cat_names = ['s+e+', 's+e-', 's-e+', 's-e-']
    probs = {}
    for cat in cats:
        cati = (f[:,:,0]==1) * (f[:,:,1]==cat)
        probs[str(cat)] = np.mean(cati, axis=0)
    
    p_df = pd.DataFrame.from_dict(probs)

    ## select the best N for each class
    best_ixs = {}
    selected_params = {}
    for cat in cats:
        cat_df = p_df.copy().sort_values(by=str(cat), axis=0, ascending=False)
        best_ixs[str(cat)] = np.array(cat_df.index)[0:N_per_class]
        selected_params[str(cat)] = rvs[best_ixs[str(cat)]]

    ## plot a histogram for each class
    fig, axs = plt.subplots(1,4, figsize=(13,3), sharey=True)
    for c, cat in enumerate(cats):
        hist = np.histogram(probs[str(cat)], bins=np.arange(0,1.1,0.1))
        axs[c].hist(probs[str(cat)], bins=np.arange(0,1.1,0.1), color='0.5', ec='1', lw=1.5)
        axs[c].set_title(cat_names[c])
        axs[c].set_xlabel('Prob.')
        # trans = transforms.blended_transform_factory(axs[c].transData, axs[c].transAxes)
        for b in range(len(hist[1])-1):
            if hist[0][b] > 0:
                axs[c].text((hist[1][b] + hist[1][b+1])/2, hist[0][b], str(hist[0][b]), ha='center', va='center', fontsize=8)

    axs[0].set_ylabel('log(count)')
    axs[0].set_yscale('log')
    axs[0].set_ylim([1,10000])
    fig.savefig('{}/fit_histograms.png'.format(outdir), dpi=400)

    # save the params and ixs for the best
    with open('{}/selected_ixs.pkl'.format(outdir), 'wb') as fl:
        pickle.dump(best_ixs, fl)
    with open('{}/selected_params.pkl'.format(outdir), 'wb') as fl:
        pickle.dump(selected_params, fl)

    return selected_params

def run_model(i, exps):
    m = model.Model(exps[list(exps.keys())[i]]) # ordered dict so OK
    for t in range(m.T):
        m.step()
    return m

def plot_baseline(mods, exps, exp_name):
    '''
    create baseline plots showing agent-level and model-level dynamics
    '''
    cat_names = ['s+e+', 's+e-', 's-e+', 's-e-']
    ### 1. model-level averages for each type
    fig, axs = plt.subplots(9,4,figsize=(21, 12), sharex=True, sharey='row')
    for i, exp in enumerate(exps.keys()):
        ax_col = int(exp[0])
        # select the objects to plot
        m = mods[i]
        ag = m.agents
        objs = [ag.livestock, ag.savings, ag.income, ag.wage_labor>0, ag.salary_labor>0, m.rangeland.R, m.land.organic, m.land.yields]
        names = ['livestock','savings','income','wage lbr','salary lbr','grass','SOM','yield']
        for o, obj in enumerate(objs):
            ax = axs[o, ax_col]
            # extract the plot vals
            if names[o] == 'grass':
                plot_vals = obj
            else:
                plot_vals = np.mean(obj, axis=1)
            # plot
            ax.plot(plot_vals, lw=1, color='b')

            ## formatting
            if ax_col == 0:
                ax.set_ylabel(names[o])
            if o == 0:
                ax.set_title(cat_names[ax_col], fontsize=30)
            if names[o] == 'income':
                ax.axhline(0, color='k', lw=0.75)

        # add the rainfall
        axs[-1,ax_col].plot(m.climate.rain, color='r', lw=1)
    
    
    ## FORMATTING
    axs[-1,0].set_ylabel('rain')
    for axi in axs[-1]:
        # axi.set_xticks(np.arange(m.T+1))
        axi.set_xlim([0, m.T])
        axi.set_xlabel('year')
    ax_flat = axs.flatten()
    for ax in ax_flat:
        ax.grid(which='major', axis='y')

    fig.savefig('../outputs/{}/model_averages.png'.format(exp_name), dpi=200)
    sys.exit()
    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()