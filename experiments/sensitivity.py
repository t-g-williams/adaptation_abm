'''
sensitivity analysis using a random forest
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import model.model as mod
import model.base_inputs as inp
import calibration.POM as POM
from . import multi_scenario_comparison as msc
from . import climate_shock_multi as shock
import code
import tqdm
import numpy as np
import pyDOE
import pickle
import pandas as pd
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence

def main():
    exp_name = '2019_10_9'
    N_vars = 10 # number of random variable sets to generate
    N_reps = 100 # number of times to repeat model for each variable set
    ncores = 10

    ### 1. load the POM variables
    pom_nvars = 100000
    pom_nreps = 10
    f = '../outputs/{}/POM/{}_{}reps/input_params_0.pkl'.format(exp_name, pom_nvars, pom_nreps)
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    inp_base['model']['T'] = 100
    inp_base['model']['n_agents'] = 200
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'

    ### 2. sample: generate random perturbed variable sets
    perturb_perc = 20
    sens_vars = {
        'agents' : ['wealth_init_mean','cash_req_mean','livestock_cost'],
        'land' : ['organic_N_min_init','max_organic_N','fast_mineralization_rate',
            'slow_mineralization_rate','loss_max','loss_min','max_yield',
            'rain_crit','rain_cropfail_low_SOM','random_effect_sd',
            'crop_CN_conversion','residue_CN_conversion',
            'wealth_N_conversion','livestock_frac_crops','livestock_residue_factor'],
        'climate' : ['rain_mu','rain_sd']
    }
    params, keys, names = hypercube_sample(N_vars, sens_vars, inp_base, perturb_perc)

    ### 3. run the policy analysis
    T_shock = [30]
    T_res = [10]
    shock_mag = [0.1]
    Ys = calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, shock_mag)

    ### 4. run the random forest
    var_imp, pdp_data, fit = random_forest(Ys, params, names, keys)

    ### 5. plot results
    plot_rf_results(var_imp, pdp_data, fit, exp_name)

def hypercube_sample(N, sens_vars, inp_base, perturb_perc):
    '''
    create latin hypercube sample
    '''
    # create list of vars
    keys = []
    val_names = []
    vals = []
    for k, v in sens_vars.items():
        val_names += v
        keys += [k] * len(v)
        for vi in v:
            vals += [inp_base[k][vi]]

    # generate uniform vars
    np.random.seed(0)
    rvs_unif = pyDOE.lhs(len(vals), N)

    # scale
    rvs = np.array(vals)[None,:] * (1 + (rvs_unif-0.5) * 2*perturb_perc/100)

    return rvs, keys, val_names

def calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, shock_mag):
    '''
    for each set of parameters, run the model and calculate the quantity of interest (qoi)
    the QoI is the probability that cover crops is preferable to insurance
    with respect to the effects on wealth of a shock at T_shock, assessed over T_res years
    '''
    exp_name = exp_name + '/sensitivity'
    outdir = '../outputs/{}'.format(exp_name)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    adap_scenarios = {
        # 'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    param_chunks = POM.chunkIt(np.arange(params.shape[0]), ncores)
    results = Parallel(n_jobs=ncores)(delayed(chunk_QoI)(chunk, exp_name, N_reps, params, keys, names,
        inp_base, adap_scenarios, shock_mag, T_shock, T_res) for chunk in param_chunks)

    # calculate mean over agent types for simplicity
    tmp = np.array(np.concatenate(results))
    QoIs = np.mean(tmp, axis=1)
    return QoIs

def chunk_QoI(chunk, exp_name, N_reps, params, keys, names, inp_base, adap_scenarios, shock_mag, T_shock, T_res):
    '''
    Calculate the QoI for a chunk of parameter sets
    '''
    ncores = 1 # don't parallelize within this
    QoIs = []
    for c in (tqdm(chunk) if (0 in chunk) else chunk): # pbar will be rough
        # set the inputs
        inputs = array_to_dict(params[c], keys, names, inp_base)
        # run the sims
        results = shock.run_shock_sims(exp_name, N_reps, inputs, adap_scenarios, shock_mag, T_shock, ncores, T_res,
            save=False, flat_reps=False)
        # calculate the QOIs
        bools = results['cover_crop'] > results['insurance']
        probs = np.array(bools.mean(axis=0))
        QoIs.append(probs)

    return QoIs

def array_to_dict(params, keys, names, inp_base):
    d = copy.deepcopy(inp_base)
    for i, k in enumerate(keys):
        d[k][names[i]] = params[i]
    return d

def random_forest(y, X, varz, keys):
    # fit gradient boosted forest
    gb = GradientBoostingRegressor(random_state=0).fit(X, y)

    # extract importances
    var_imp = pd.DataFrame({'importance' : gb.feature_importances_, 'key' : keys, 
        'variable' : varz}).sort_values(['key','importance'], ascending=False)

    # var_imp = pd.DataFrame(gb.feature_importances_, index = varz, 
    #         columns=['importance']).sort_values('importance', ascending=False)

    # partial dependence
    pdp_data = {}
    for xi, var in enumerate(varz):
        pdp_data[varz[xi]] = partial_dependence(gb, features=[xi], X=X)

    # predictive accuracy (in-sample)
    fit = gb.score(X,y)

    return var_imp, pdp_data, fit

def plot_rf_results(var_imp, pdp_data, fit, exp_name):
    ## have the plot function here for now

    ## A. variable importance
    # format the data for plotting
    var_imp['ix'] = np.arange(var_imp.shape[0])
    colors = {'land' : 'b', 'climate' : 'r', 'agents' : 'k'}
    var_imp['color'] = np.nan
    for index, row in var_imp.iterrows():
        var_imp.loc[index, 'color'] = colors[row.key]
    # create the figure
    fig, ax = plt.subplots(figsize=(5,9))
    xs = np.array(var_imp['ix'])
    ax.barh(xs, np.array(var_imp.importance), color=var_imp.color)
    ax.set_yticks(xs)
    ax.set_yticklabels(np.array(var_imp['variable']))
    ax.set_xlabel('Variable importance')
    fig.savefig('../outputs/{}/sensitivity/variable_importance.png'.format(exp_name))
        
    ## B. partial dependence plots
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.plot(pdp_data['livestock_frac_crops'][0][0], pdp_data['livestock_frac_crops'][1][0])
    fig.savefig('../outputs/{}/sensitivity/partial_dependence.png'.format(exp_name))

    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()