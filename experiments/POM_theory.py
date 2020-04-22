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
import scipy.stats
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
    N_samples = 5000
    N_level1 = 20 # also do LHS for level 1 params
    ncores = 40
    nreps = 10 # to account for simulation-level variability (climate and prices)
    exp_name = 'pom_theory'
    inputs = {
        'model' : {'n_agents' : 200, 'T' : 30, 'exp_name' : exp_name},
        'decisions' : {}
    }
    scenarios = {'baseline' : {'decisions' : {'framework' : 'imposed', 'imposed_action' : {'conservation' : False, 'fertilizer' : False}}},
        'conservation' : {'decisions' : {'framework' : 'imposed', 'imposed_action' : {'conservation' : True, 'fertilizer' : False}}},
        'fertilizer' : {'decisions' : {'framework' : 'imposed', 'imposed_action' : {'conservation' : False, 'fertilizer' : True}}},
        'util' : {'decisions' : {'framework' : 'util_max'}}
        }
    fit_threshold = 0.8

    # define the variables for calibration
    calib_vars = define_calib_vars(exp_name)
    # generate set of RVs
    rvs, rvs1 = hypercube_sample(N_samples, N_level1, calib_vars)
    rvs1 = np.array([[4000,0.6],[4000,0.4],[1000,0.6],[1000,0.4]]) # (som, rain mu)

    # # run the model and calculate fitting metrics
    fits = run_model(rvs, rvs1, inputs, calib_vars, scenarios, ncores, nreps, load=True)
    
    # # process the fit data
    process_fits(fits, rvs, calib_vars, inputs, nreps, exp_name, fit_threshold)
    # plot_ex_post(exp_name, N_samples, nreps, rvs, calib_vars, fit_threshold)

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
        # ['climate','rain_sd',False,0.1,0.3,False,False],
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
    t_frac = 0.5
    ag_frac = 0.1
    # extract model data
    income = {}
    som = {}
    wealth = {}
    for sc_name, mods_sc in mods.items():
        income[sc_name] = []
        som[sc_name] = []
        wealth[sc_name] = []
        for mod_r in mods_sc:
            income[sc_name].append(mod_r.agents.income)
            som[sc_name].append(mod_r.land.organic)
            wealth[sc_name].append(mod_r.agents.wealth)
        income[sc_name] = np.mean(np.array(income[sc_name]), axis=0) # (t, agent) averaged over reps
        som[sc_name] = np.mean(np.array(som[sc_name]), axis=0) # (t, agent) averaged over reps
        wealth[sc_name] = np.mean(np.array(wealth[sc_name]), axis=0) # (t, agent) averaged over reps

    #### 1. fertilizer-SOM-poverty
    ### (a) fertilizer lock-in
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
    
    oneA = np.sum(a*b*c*d) > ag_frac

    ### (b) fertilizer for poverty alleviation 
    # A.  At t>>0, SOM increases under exclusive inorganic fertilizer application.
    a = som['fertilizer'][-1] > som['fertilizer'][0]
    # print(np.mean(a))
    # B.  At t>>0, poverty is reduced under fertilizer application.
    b = wealth['fertilizer'][-1] > wealth['baseline'][-1]

    oneB = np.sum(a*b) > ag_frac

    #### 2. conservation-SOM-poverty
    two_all = (wealth['conservation'][1] < wealth['baseline'][1]) * (wealth['conservation'][-1] > wealth['baseline'][-1])
    two = np.sum(two_all) > ag_frac

    ######## ADOPTION ########
    #### 3. fertilizer adoption
    threes = np.full(nreps, False)
    fours = np.full(nreps, False)
    for r in range(nreps):
        # extract single model data
        m = mods['util'][r]
        feas = m.agents.option_feasibility[1:]
        opts = m.agents.decision_options
        util = m.agents.exp_util[1:]
        fert_ix = np.where(np.array([((oi['fertilizer']==True)*(oi['conservation']==False)) for oi in opts])==1)[0][0]
        fert_choice = np.where(np.array([((oi['fertilizer']==True)) for oi in opts])==1)[0]
        cons_ix = np.where(np.array([((oi['fertilizer']==False)*(oi['conservation']==True)) for oi in opts])==1)[0][0]
        cons_choice = np.where(np.array([((oi['conservation']==True)) for oi in opts])==1)[0]

        # a)  Fertilizer improves utility, but some agents can't choose it (i.e., have to choose baseline) because of inadequate cash
        # a)  For >10% of the agents, fertilizer improves utility but most of the time (>50%) they can't choose it (i.e., have to choose baseline) because of inadequate cash
        # ^^ justification for this??? maybe say these were arbitrary, but we wanted the process to activate a non-neglibible amount of the time (ie be frequently present in the model)
        fert_util = util[:,fert_ix,:] > util[:,0,:]
        a_all = fert_util * ~feas[:,fert_ix] * (m.agents.choice_ixs[1:]==0) # dimension: (T-1, agents)
        a = np.mean((np.mean(a_all, axis=0) > t_frac)) > ag_frac

        # b)  Less risk averse agents are significantly more likely to choose fertilizer
        # calculate the probability of choosing fertilizer for each agent
        p_fert = np.mean(np.in1d(m.agents.choice_ixs, fert_choice).reshape(m.agents.choice_ixs.shape), axis=0) # dimension: agent
        risk_tol = m.agents.risk_tolerance
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.scatter(risk_tol, p_fert)
        # fig.savefig('fert_risk.png')
        if np.max(p_fert) > 0:
            corr = scipy.stats.pearsonr(risk_tol, p_fert)
            b = (corr[0] > 0) * (corr[1] < 0.05) # significant, positive correlation
        else:
            b = False # if all-zero array

        threes[r] = a*b

        #### 4. conservation adoption
        # a)    Conservation option improves utility, but households can’t choose it because of short-term negative effects
        cons_util = util[:,cons_ix,:] > util[:,0,:]
        a_all = cons_util * ~feas[:,cons_ix] * (m.agents.choice_ixs[1:]==0) # dimension: (T-1,agents)
        a = np.mean((np.mean(a_all, axis=0) > t_frac)) > ag_frac

        # b)  Under the same asset endowments, households with higher time discounting rates don't choose conservation but those with lower time discounting rates do
        p_cons = np.mean(np.in1d(m.agents.choice_ixs, cons_choice).reshape(m.agents.choice_ixs.shape), axis=0) # dimension: agent
        disc = m.agents.discount_rate
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.scatter(disc, p_cons)
        # fig.savefig('conservation_vs_discount_rate.png')
        if np.max(p_cons) > 0:
            corr = scipy.stats.pearsonr(disc, p_cons)
            b = (corr[0] > 0) * (corr[1] < 0.05) # significant, positive correlation
        else:
            b = False # if all-zero array

        fours[r] = a*b

    all_fits = [oneA,oneB,two,np.mean(threes),np.mean(fours)]
    # code.interact(local=dict(globals(), **locals()))
    return all_fits

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
    outname = 'fits_raw_{}n0_{}n1_{}reps.npz'.format(rvs.shape[0], rvs1.shape[0], nreps)       
    if os.path.isfile(outdir + outname) and load:
        return np.load(outdir+outname, allow_pickle=True)['data']

    # run the model
    N = rvs.shape[0]
    sim_chunks = chunkIt(np.arange(N), ncores)

    if ncores > 1:
        fits_par = Parallel(n_jobs=ncores)(delayed(run_chunk_sims)(sim_chunks[i], rvs, rvs1, inp_all, calib_vars, scenarios, nreps) for i in range(len(sim_chunks)))
        # convert to list of list of list
        fits = [item for sublist in fits_par for item in sublist]
    else:
        fits = run_chunk_sims(sim_chunks[0], rvs, rvs1, inp_all, calib_vars, scenarios, nreps) # result is [level0][level1][pattern]

    # fits_all = fits
    fits_all_np = np.array(fits)
    # write
    np.savez_compressed(outdir + outname, data=fits_all_np)
    # fits_all_np = np.array(pd.DataFrame.from_dict(fits, orient='index'))

    return fits_all_np

def run_chunk_sims(ixs, rvs, rvs1, inp_all, calib_vars, scenarios, nreps):
    '''
    run the "ixs" rows of rvs simulations
    '''
    inp_all = copy.deepcopy(inp_all) # just in case there are parallel issues with ids
    fits = OrderedDict()
    fits = []

    for ix in (tqdm(ixs) if (0 in ixs) else ixs): # pbar will be rough
        # loop over the level 1 variables
        fits_i = []
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
            # fits['{}_{}'.format(ix, j)] = fitting_metrics(mods, nreps)
            fits_i.append(fitting_metrics(mods, nreps))

        fits.append(fits_i)

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
    fit_flat = fits.flatten().reshape((fits.shape[0]*fits.shape[1],fits.shape[2])) # shape: (model config, fits)
    

    ## 1. frequency that each pattern is fit AND histogram
    fig, axs = plt.subplots(1,2,figsize=(8,5))
    xs = np.arange(fit_flat.shape[1])
    axs[0].bar(x = xs, height=np.mean(fit_flat, axis=0))
    axs[0].set_ylabel('Prob(pattern satisfied)')
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(['1a','1b','2','3','4'])
    axs[0].grid(False)
    axs[0].set_xlabel('Pattern')
    sums = np.sum(fit_flat, axis=1)
    axs[1].hist(sums)
    axs[1].set_xlabel('Number of patterns matched')
    axs[1].grid(False)
    axs[1].set_xticks(xs+1)
    axs[1].set_xticklabels((xs+1).astype(int))
    nmax = len(np.where(sums==max(sums))[0])
    axs[1].text(max(sums), nmax+1, '{} patterns\n{} model(s)'.format(max(sums), nmax), horizontalalignment='center')
    axs[1].set_xlim([0, fits.shape[2]])
    fig.savefig(outdir + 'histogram.png')


    ## 2. parameter sets (level0) + how many patterns they match
    # sum_1a = np.sum(fits == np.array([1,0,1,1,1]), axis=(2))
    # num_all_1a = np.sum(sum_1a==5, axis=1)
    # mean_1a = np.mean(sum_1a, axis=1)
    # sum_1b = np.sum(fits == np.array([0,1,1,1,1]), axis=(2))
    # num_all_1b = np.sum(sum_1b==5, axis=1)

    ## 2. finding how many desirable patterns each parameter configuration matches
    fit_a = np.sum(np.sum(fits[:,:,[0,2]] == np.array([1,1]), axis=2)==2, axis=1) # (N) <-- how many times does each config fit 1a+2
    fit_b = np.sum(np.sum(fits[:,:,[1,2]] == np.array([1,1]), axis=2)==2, axis=1) # (N)
    fit_both = (fit_a>0) * (fit_b>0) # fits both 1a and 1b at some point
    # now find how many of patterns 3 and 4 these models fit
    sum_34 = np.sum(fits[:,:,[3,4]], axis=2) * fit_both[:,None] # (N,n2)
    mean_sum_34 = np.round(np.mean(sum_34, axis=1),2)
    sel_sums = mean_sum_34[fit_both]
    fig, ax = plt.subplots()
    ax.hist(sel_sums)
    ax.set_xlim([0,2])
    mx = sel_sums.max()
    nmax = (sel_sums==mx).sum()
    ax.text(mx, nmax+1, '{}/2, {}mod(s)'.format(mx, nmax))
    ax.set_xlabel('Fit to patterns 3 and 4')
    fig.savefig(outdir+'patterns_34_fitting.png')
    # select the best fitting model(s)
    max_ix = np.where(mean_sum_34==mx)[0]
    ok_ix = np.where(mean_sum_34 >= fit_threshold*mx)[0] # within threshold of the maximum

    ## 3. plot their parameter values
    vals_best = rvs[max_ix]
    ix0 = np.array(calib_vars['level0'] == True)
    vals_sc = (vals_best - np.array(calib_vars['min_val'])[None,ix0]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,ix0]
    fig, ax = plt.subplots(figsize=(8,5))
    # plot everything within the threshold of this quality
    vals_ok = rvs[ok_ix]
    vals_ok_sc = (vals_ok - np.array(calib_vars['min_val'])[None,ix0]) / np.array(calib_vars['max_val']-calib_vars['min_val'])[None,ix0]
    ax.plot(np.transpose(vals_ok_sc), alpha=0.5, lw=0.5, color='k', label='_nolegend_')
    ax.plot(np.transpose(vals_sc), color='b', label='best')
    ax.set_xticks(np.arange(calib_vars.loc[ix0].shape[0]))
    ax.set_xticklabels(calib_vars.key2[ix0], rotation=90)
    ax.set_ylabel('Value (scaled)')
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend()
    fig.savefig(outdir + 'param_vals.png', dpi=500)

    ## 4. save their values
    inp_all = base_inputs.compile()
    inp_all = overwrite_inputs(inp_all, inputs)
    fits_mod = []
    for v, vi in enumerate(max_ix):
        inp_all = overwrite_rv_inputs(inp_all, rvs[vi], [], calib_vars, {})
        inp_all['model']['exp_name'] = '{}/{}_{}reps/{}/'.format(exp_name, N, nreps, v)
        m = model.Model(inp_all)
        for t in range(m.T):
            m.step()
        # plt_single.main(m)
        # fits_mod.append(fitting_metrics(m, nreps))

        # save the inputs -- as csv and pickle
        df = pd.DataFrame.from_dict({(i,j): inp_all[i][j] 
                                   for i in inp_all.keys() 
                                   for j in inp_all[i].keys()},
                               orient='index')
        df.to_csv(outdir + 'input_params_{}.csv'.format(v))
        with open(outdir + 'input_params_{}.pkl'.format(v), 'wb') as f:
            pickle.dump(inp_all, f)
    
    # code.interact(local=dict(globals(), **locals()))


    # run and plot one of them
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