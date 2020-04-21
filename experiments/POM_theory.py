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
from collections import OrderedDict
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
import plot.single as plt_single
from . import trajectories


def main():
    # specify experimental settings
    N_samples = 100
    N_level1 = 30 # also do LHS for level 1 params
    ncores = 1
    nreps = 10 # to account for simulation-level variability (climate and prices)
    exp_name = 'pom_theory'
    inputs = {
        'model' : {'n_agents' : 200, 'T' : 50, 'exp_name' : exp_name},
        'decisions' : {'framework' : 'imposed'}
    }
    scenarios = {'baseline' : {'decisions' : {'imposed_action' : {'conservation' : False, 'fertilizer' : False}}},
        'conservation' : {'decisions' : {'imposed_action' : {'conservation' : True, 'fertilizer' : False}}},
        'fertilizer' : {'decisions' : {'imposed_action' : {'conservation' : False, 'fertilizer' : True}}}}
    fit_threshold = 0.8

    # define the variables for calibration
    calib_vars = define_calib_vars(exp_name)
    # generate set of RVs
    rvs, rvs1 = hypercube_sample(N_samples, N_level1, calib_vars)

    # # run the model and calculate fitting metrics
    fits = run_model(rvs, rvs1, inputs, calib_vars, scenarios, ncores, nreps, load=False)
    
    # # process the fit data
    process_fits(fits, rvs, calib_vars, inputs, nreps, exp_name, fit_threshold)
    plot_ex_post(exp_name, N_samples, nreps, rvs, calib_vars, fit_threshold)

def define_calib_vars(exp_name):
    calib_vars = pd.DataFrame(
        # key1, key2, key3, min, max, as_int, level0
        [['adaptation','conservation','area_req',0,0.4,False,True],
        ['agents','living_cost_pp',False,200,2000,True,True],
        ['decisions','risk_tolerance_mu',False,100,1000,True,True],
        ['decisions','discount_rate_sigma',False,0,0.2,False,True],
        ['market','farm_cost_baseline',False,50,500,True,True],
        ['market','fertilizer_cost',False,10,40,True,True],
        ['land','fast_mineralization_rate',False,0.25,0.75,False,True],
        ['livestock','frac_crops',False,0.5,1,False,True],
        ['livestock','frac_N_import',False,0,1,False,True],
        # level1 parameters (i.e. vary within a single set of level0 parameters)
        ['land','organic_N_min_init',False,1000,5000,True,False],
        ['climate','rain_mu',False,0.4,0.8,False,False],
        ['climate','rain_sd',False,0.1,0.3,False,False],
        ],
        columns = ['key1','key2','key3','min_val','max_val','as_int','level0'])

    outdir = '../outputs/{}'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    calib_vars.to_csv('{}/calib_vars.csv'.format(outdir))

    return calib_vars

def fitting_metrics(mods, nreps):
    '''
    determine whether the model displays the desired patterns
    '''
    # process data
    income = {}
    for sc_name, mods_sc in mods.items():
        income[sc_name] = []
        for mod_r in mods_sc:
            income[sc_name].append(mod_r.agents.income)
        income[sc_name] = np.mean(np.array(income[sc_name]), axis=0) # (t, agent) averaged over reps

    #### 1. fertilizer-SOM-poverty
    ## A.  at t=0, income is positive without fertilizer application (i.e., baseline management)
    a = income['baseline'][0] > 0

    ## B.  at t=0, fertilizer application is income-increasing relative to baseline management
    b = income['fertilizer'][0] > income['baseline'][0]

    tf = np.argmax(income['fertilizer'] < 0, axis=0) # note: what shows up as 0: {all F, all T, [T,F,F,...]}. most likely these arrays are sth like F,F,T... (i.e., increasing. the first T is when income is first < 0)
    tb = np.argmax(income['baseline'] < 0, axis=0)
    b_all_pos = np.min(income['baseline'], axis=0) > 0

    ## C.  at tf >0, fertilizer application yields negative income
    c = tf > 0
    
    ## D.  at tb >tf, baseline management yields negative income (tb > tf, which means that fertilizer use makes agents reach this state faster, due to its costs)
    d = ((tb > 0) * (tb > tf)) | (b_all_pos)
    
    one = a*b*c*d
    
    code.interact(local=dict(globals(), **locals()))










    # ## 1. wealth/poverty FOR AGENT TYPES (NON-EMPIRICAL)
    # ## A: one group always has >0 wealth
    # ## B: one group has a probability of having >0 wealth \in [0.2,0.8]
    # oneA = False
    # oneB = False
    # for t in types:
    #     ag_type = mod.agents.type == t
    #     prob = np.mean(mod.agents.cant_cope[-1,ag_type])
    #     cond1 = prob == 1
    #     oneA = True if cond1 else oneA
    #     cond2 = ((prob >= 0.2) and (prob <= 0.8))
    #     oneB = True if cond2 else oneB
    # one = bool(oneA * oneB)

    ## VULNERABILITY
    # %age of Hhs that can't initially meet food requirements (before wage labor or destocking) is \in (30%,45%)
    # note: it is 31% in OR1 and 44% in OR2
    p_cope = np.mean(mod.agents.cons_red_rqd[-n_yrs:])
    one = True if ((p_cope>=0.3) & (p_cope<=0.45)) else False
    #### NOTE: REMOVING THIS ONE BECAUSE IT CONFLICTS WITH THE WAGE AND SALARY REQUIREMENTS AND LIVESTOCK
    #### (IE HAVING 30-45% OF PEOPLE NEEDING TO COPE BUT >80% WITH LIVESTOCK AND <15% WITH WAGE IS ROUGH)

    ## 2. land degradation exists
    # not consistently someone at maximum value
    # (this is calculated over TIME, not a single AGENT that's at max) 
    maxs = np.max(mod.land.organic[-n_yrs:], axis=1)
    two = False if max(maxs) == mod.land.max_organic_N else True

    ## 3. rangeland is not fully degrated
    ## A: P(regional destocking required) \in [0.1,0.5]
    prob = np.mean(mod.rangeland.destocking_rqd)
    threeA = True if ((prob >= 0.05) and (prob <= 0.5)) else False
    ## B: min(reserve biomass) > 0.2*R_max
    threeB = True if min(mod.rangeland.R >= 0.2 * mod.rangeland.R_max) else False
    ## C: there are livestock on the rangeland in the last n_yrs
    # threeC = True if (min(mod.rangeland.livestock_supported[-n_yrs:]) > 0) else False
    three = bool(threeA * threeB)

    ## 4. livestock: 
    # >80% of HHs have livestock on average
    fourA = np.mean(mod.agents.livestock[-n_yrs:]>0) >= 0.8
    # 90th%ile agent has less than 10 livestock on average
    fourB = np.percentile(np.mean(mod.agents.livestock, axis=0), 90) < 10 # take mean over time for each agent
    # median agent has less than 5 on average
    fourC = np.percentile(np.mean(mod.agents.livestock, axis=0), 50) < 5
    # maximum ever is less than 50
    fourD = np.max(mod.agents.livestock)<50
    four = bool(fourA*fourB*fourC*fourD)
    # four = bool(fourA*fourB*fourD)

    ## 5. non-farm income
    # upper and lower limits on wage and salary income
    p_wage = np.mean(mod.agents.wage_labor[-n_yrs:] > 0)
    p_sal = np.mean(mod.agents.salary_labor[-n_yrs:] > 0)
    # print(p_sal)
    fiveA = ((p_wage>0.05) & (p_wage<0.15))
    fiveB = ((p_sal>0.05) & (p_sal<0.15))
    # fiveA = ((p_wage>0.1) & (p_wage<0.15))
    # fiveB = ((p_sal>0.05) & (p_sal<0.1))
    # code.interact(local=dict(globals(), **locals()))
    five = bool(fiveA*fiveB)

    # return [one,two,three,four,fiveA,fiveB]
    return [two,three,four,fiveA,fiveB]
    # return [two,three,four,five]

def hypercube_sample(N, N1, calib_vars):
    '''
    create latin hypercube sample
    sample the level0 variables and level1 variables separately
    '''
    #### level 0
    # generate uniform vars
    np.random.seed(0)
    ixs = np.array(calib_vars['level0'] == True)
    rvs_unif = pyDOE.lhs(ixs.sum(), N)

    # scale
    rvs = np.array(calib_vars.min_val[ixs])[None,:] + rvs_unif * \
            np.array(calib_vars.max_val[ixs] - calib_vars.min_val[ixs])[None,:]

    # convert integer params
    rvs[:,calib_vars.as_int[ixs]] = rvs[:,calib_vars.as_int[ixs]].astype(int)

    #### level 1
    ixs1 = np.array(calib_vars['level0'] == False)
    rvs_unif1 = pyDOE.lhs(ixs1.sum(), N1)

    # scale
    rvs1 = np.array(calib_vars.min_val[ixs1])[None,:] + rvs_unif1 * \
            np.array(calib_vars.max_val[ixs1] - calib_vars.min_val[ixs1])[None,:]

    # convert integer params
    rvs1[:,calib_vars.as_int[ixs1]] = rvs1[:,calib_vars.as_int[ixs1]].astype(int)

    return rvs, rvs1

def run_model(rvs, rvs1, inputs, calib_vars, scenarios, ncores, nreps, load=False):
    '''
    run the model for each of the RV combinations
    '''
    # create the full set of inputs
    inp_all = base_inputs.compile()
    inp_all = overwrite_inputs(inp_all, inputs)
    outdir = '../outputs/{}/'.format(inputs['model']['exp_name'])
    outname = 'fits_raw.npz'
    if os.path.isfile(outdir + outname) and load:
        return np.load(outdir+outname, allow_pickle=True)['data']

    # run the model
    N = rvs.shape[0]
    sim_chunks = chunkIt(np.arange(N), ncores)

    if ncores > 1:
        fits_par = Parallel(n_jobs=ncores)(delayed(run_chunk_sims)(sim_chunks[i], rvs, rvs1, inp_all, calib_vars, scenarios, nreps) for i in range(len(sim_chunks)))
        fits = {}
        for fit in fits_par:
            for k, v in fit.items():
                fits[k] = v
    else:
        fits = run_chunk_sims(sim_chunks[0], rvs, rvs1, inp_all, calib_vars, scenarios, nreps)
    fits_all = fits
    fits_all_np = np.array(pd.DataFrame.from_dict(fits, orient='index'))

    fits_all_np = np.array(fits_all_np)

    if trajectory:
        # return all fits
        np.savez_compressed(outdir+outname, data=fits_all_np)
        return fits_all_np
    else:
        ## POM
        # average over all reps
        fits_avg = np.mean(fits_all_np, axis=0)
        np.savez_compressed(outdir+outname, data=fits_avg)
        return fits_avg

def run_chunk_sims(ixs, rvs, rvs1, inp_all, calib_vars, scenarios, nreps):
    '''
    run the "ixs" rows of rvs simulations
    '''
    inp_all = copy.deepcopy(inp_all) # just in case there are parallel issues with ids
    fits = OrderedDict()

    for ix in (tqdm(ixs) if (0 in ixs) else ixs): # pbar will be rough
        # loop over the level 1 variables
        for j, rv_j in enumerate(rvs1):
            mods = {}
            for sc_name, sc_params in scenarios.items():
                # initialize the inputs
                inp_ix = overwrite_rv_inputs(inp_all, rvs[ix], rv_j, calib_vars, sc_params)
                mods[sc_name] = []

                for r in range(nreps):
                    inp_ix['model']['seed'] = r
                
                    # run the model
                    m = model.Model(inp_ix)
                    for t in range(m.T):
                        m.step()
                    mods[sc_name].append(m)
    
            # calculate model fitting metrics
            fits['{}_{}'.format(ix, j)] = fitting_metrics(mods, nreps)

    return fits

def plot_ex_post(exp_name, N, nreps, rvs, calib_vars, fit_threshold):
    '''
    plot the values in the save csv
    '''
    # load the fits
    outdir = '../outputs/{}/{}_{}reps/'.format(exp_name, N, nreps)
    fit_pd = pd.read_csv(outdir + 'fits.csv', index_col=0)

    # identify the best fitting models
    max_fit = fit_pd['sum'].iloc[0]
    max_fit_locs = fit_pd.loc[fit_pd['sum']==max_fit, 'sum'].index

    # plot their parameter values
    vals_best = rvs[max_fit_locs]
    vals_sc = (vals_best - np.array(calib_vars['min_val'])[None,:]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,:]
    fig, ax = plt.subplots(figsize=(8,3.5))
    # plot everything within 10% of this quality
    ok_fits = fit_pd.loc[fit_pd['sum'] >= fit_threshold*max_fit, 'sum'].index
    vals_ok = rvs[ok_fits]
    vals_ok_sc = (vals_ok - np.array(calib_vars['min_val'])[None,:]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,:]
    ax.plot(np.transpose(vals_ok_sc), alpha=0.5, lw=0.5, color='k', label='within 20%')
    ax.plot(np.transpose(vals_sc), color='b', label='Comparable models')
    # plot the best one in red
    ax.plot(np.transpose(vals_sc)[:,0], color='r', label='Selected model')

    ax.set_xticks(np.arange(calib_vars.shape[0]))
    symbols = [r'$C_{low}^{lower}$', r'$k_{fast}$', r'$WN_{conv}$', r'$c_{residues}$',
        r'$CN_{residue}$', r'$CR$', r'$l_N^{max}$', r'$W_0$']
    ax.set_xticklabels(symbols, rotation=90)

    ax.set_ylabel('Value (scaled)')
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    # ax.legend(ncol=3)
    fig.savefig(outdir + 'param_vals_ex_post.png', dpi=500)
    # code.interact(local=dict(globals(), **locals()))

def process_fits(fits, rvs, calib_vars, inputs, nreps, exp_name, fit_threshold):
    '''
    process the POM results
    '''
    N = rvs.shape[0]
    outdir = '../outputs/{}/{}_{}reps/'.format(exp_name, N, nreps)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # code.interact(local=dict(globals(), **locals()))
    fit_pd = pd.DataFrame(fits)
    # fit_pd = pd.DataFrame.from_dict(fits, orient='index')
    fit_pd['sum'] = fit_pd.sum(axis=1)

    vals = fit_pd['sum'].value_counts()

    # histogram of sums
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(fit_pd['sum'])
    ax.set_xlabel('Mean number of patterns matched')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(len(fits[0])+1))
    ax.set_xticklabels(np.arange(len(fits[0])+1).astype(int))
    ax.grid(False)
    i = max(vals.index)
    ax.text(i, vals.loc[i]+1, '{} patterns\n{} model(s)'.format(np.round(i, 1), vals.loc[i]), horizontalalignment='center')
    ax.set_xlim([0, fits.shape[1]])
    fig.savefig(outdir + 'histogram.png', dpi=500)
    # write outputs
    fit_pd = fit_pd.sort_values(by='sum', axis=0, ascending=False)
    fit_pd['sim_number'] = fit_pd.index
    fit_pd.to_csv(outdir + 'fits.csv')
    # np.savetxt(outdir + 'rvs.csv', rvs)

    # identify the best fitting models
    vals = vals.sort_index()
    max_fit = vals.index[-1]
    max_fit_locs = fit_pd.loc[fit_pd['sum']==max_fit, 'sum'].index

    # plot their parameter values
    vals_best = rvs[max_fit_locs]
    vals_sc = (vals_best - np.array(calib_vars['min_val'])[None,:]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,:]
    fig, ax = plt.subplots(figsize=(8,5))
    # plot everything within 10% of this quality
    ok_fits = fit_pd.loc[fit_pd['sum'] >= fit_threshold*max_fit, 'sum'].index
    vals_ok = rvs[ok_fits]
    vals_ok_sc = (vals_ok - np.array(calib_vars['min_val'])[None,:]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,:]
    ax.plot(np.transpose(vals_ok_sc), alpha=0.5, lw=0.5, color='k')
    ax.plot(np.transpose(vals_sc), color='b')
    ax.set_xticks(np.arange(calib_vars.shape[0]))
    ax.set_xticklabels(calib_vars.key2, rotation=90)
    ax.set_ylabel('Value (scaled)')
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend()
    fig.savefig(outdir + 'param_vals.png', dpi=500)

    # run and plot one of them
    inp_all = base_inputs.compile()
    inp_all = overwrite_inputs(inp_all, inputs)
    fits_mod = []
    for v, vi in enumerate(max_fit_locs):
        inp_all = overwrite_rv_inputs(inp_all, rvs[vi], calib_vars.key1, calib_vars.key2)
        inp_all['model']['exp_name'] = '{}/{}_{}reps/{}/'.format(exp_name, N, nreps, v)
        m = model.Model(inp_all)
        for t in range(m.T):
            m.step()
        # plt_single.main(m)
        fits_mod.append(fitting_metrics(m))

        # save the inputs -- as csv and pickle
        df = pd.DataFrame.from_dict({(i,j): inp_all[i][j] 
                                   for i in inp_all.keys() 
                                   for j in inp_all[i].keys()},
                               orient='index')
        df.to_csv(outdir + 'input_params_{}.csv'.format(v))
        with open(outdir + 'input_params_{}.pkl'.format(v), 'wb') as f:
            pickle.dump(inp_all, f)
    
    # code.interact(local=dict(globals(), **locals()))

def overwrite_inputs(all_inputs, changes):
    '''
    place the elements in "changes" in the "all_inputs" dictionary
    '''
    all_inp = copy.deepcopy(all_inputs)
    for k, v in changes.items():
        for k2, v2 in v.items():
            all_inp[k][k2] = v2
    return all_inp

def overwrite_rv_inputs(all_inputs, rv_inputs, rv1, calib_vars, sc_params):
    tmp = copy.deepcopy(all_inputs)
    # level 0 variables
    ix0 = calib_vars.index[calib_vars.level0 == True]
    for i, rv in enumerate(rv_inputs):
        calib_i = calib_vars.loc[ix0[i]]
        if isinstance(calib_i.key3, str):
            tmp[calib_i.key1][calib_i.key2][calib_i.key3] = rv
        else:
            tmp[calib_i.key1][calib_i.key2] = rv
    
    # level 1 variables
    ix1 = calib_vars.index[calib_vars.level0 == False]
    for i, rv in enumerate(rv1):
        calib_i = calib_vars.loc[ix1[i]]
        if isinstance(calib_i.key3, str):
            tmp[calib_i.key1][calib_i.key2][calib_i.key3] = rv
        else:
            tmp[calib_i.key1][calib_i.key2] = rv

    # scenario parameters
    for k1, data in sc_params.items():
        for k2, v in data.items():
            tmp[k1][k2] = v

    return tmp

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