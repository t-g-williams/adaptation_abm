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
from . import analysis as shock
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
import logging
import logging.config
logging.config.fileConfig('logger.conf', defaults={'logfilename' : 'logs/{}.log'.format(os.path.basename(__file__)[:-3])})
logger = logging.getLogger('sLogger')

def main():
    exp_name = '2019_10_15_4'
    N_vars = 1000 # number of random variable sets to generate
    N_reps = 100 # number of times to repeat model for each variable set
    ncores = 40
    pom_nvars = 100000
    pom_nreps = 10
    n_mods = 3 # number of successful POM models
    perturb_perc = 30
    load = True
    nboot_rf = 100
    sens_vars = {
        'agents' : ['wealth_init_mean','cash_req_mean','livestock_cost'],
        'land' : ['organic_N_min_init','max_organic_N','fast_mineralization_rate',
            'slow_mineralization_rate','loss_max','loss_min','max_yield',
            'rain_crit','rain_cropfail_low_SOM','random_effect_sd',
            'crop_CN_conversion','residue_CN_conversion',
            'wealth_N_conversion','livestock_frac_crops','livestock_residue_factor'],
        'climate' : ['rain_mu','rain_sd']
    }

    for m in range(n_mods):
        logger.info('model {}........'.format(m))
        ### 1. load the POM variables
        f = '../outputs/{}/POM/{}_{}reps/input_params_{}.pkl'.format(exp_name, pom_nvars, pom_nreps, m)
        inp_base = pickle.load(open(f, 'rb'))
        # manually specify some variables (common to all scenarios)
        inp_base['model']['n_agents'] = 200
        inp_base['model']['exp_name'] = exp_name
        inp_base['agents']['adap_type'] = 'always'

        ### 2. sample: generate random perturbed variable sets
        params, keys, names = hypercube_sample(N_vars, sens_vars, inp_base, perturb_perc)

        ### 3. run the policy analysis
        T_shock = [10] # measured after the burn-in
        T_res = [5]
        shock_mag = [0.1]
        inp_base['model']['T'] = T_shock[0] + T_res[0] + inp_base['adaptation']['burnin_period']
        Ys = calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, shock_mag, m, load)

        ### 4. run the random forest
        var_imp_df, var_imp_list, pdp_data = bootstrap_random_forest(Ys, params, names, keys, nboot_rf)

        ### 5. plot results
        plot_rf_results(var_imp_df, var_imp_list, pdp_data, Ys.mean(), exp_name, m)

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

def calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, shock_mag, mod_number, load):
    '''
    for each set of parameters, run the model and calculate the quantity of interest (qoi)
    the QoI is the probability that cover crops is preferable to insurance
    with respect to the effects on wealth of a shock at T_shock, assessed over T_res years
    '''
    exp_name_sens = exp_name + '/model_' + str(mod_number) + '/sensitivity'
    outdir = '../outputs/{}'.format(exp_name_sens)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    adap_scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    param_chunks = POM.chunkIt(np.arange(params.shape[0]), ncores)
    results = Parallel(n_jobs=ncores)(delayed(chunk_QoI)(chunk, exp_name_sens, N_reps, params, keys, names,
        inp_base, adap_scenarios, shock_mag, T_shock, T_res, load) for chunk in param_chunks)

    # calculate mean over agent types for simplicity
    tmp = np.array(np.concatenate(results))
    QoIs = np.mean(tmp, axis=1)
    return QoIs

def chunk_QoI(chunk, exp_name, N_reps, params, keys, names, inp_base, adap_scenarios, shock_mag, T_shock, T_res, load):
    '''
    Calculate the QoI for a chunk of parameter sets
    '''
    ncores = 1 # don't parallelize within this
    QoIs = []
    for c in (tqdm(chunk) if (0 in chunk) else chunk): # pbar will be rough
        # set the inputs
        inputs = array_to_dict(params[c], keys, names, inp_base)
        # run the sims
        outcomes = ['income'] # NOTE: SHOULD PUT THIS INPUT HIGHER UP
        exp_name_c = exp_name + '/' + str(c)
        results, results_baseline = shock.run_shock_sims(exp_name_c, N_reps, inputs, adap_scenarios, shock_mag, T_shock, ncores, T_res,
            outcomes, load=load, flat_reps=False)
        # calculate the QOIs
        bools = results_baseline['cover_crop'].loc[('income')] > results_baseline['insurance'].loc[('income')]
        probs = np.array(bools.mean(axis=0))
        QoIs.append(probs)

    return QoIs

def array_to_dict(params, keys, names, inp_base):
    d = copy.deepcopy(inp_base)
    for i, k in enumerate(keys):
        d[k][names[i]] = params[i]
    return d

def bootstrap_random_forest(y, X, varz, keys, nboot):
    '''
    bootstrap the data and fit random forest to each
    then calculate variable importance for each
    '''

    var_imps = {}
    pdp_datas = {}
    for v in varz:
        pdp_datas[v] = {'x' : [], 'y' : []}
        var_imps[v] = []

    for b in range(nboot):
        # generate bootstrap and run
        boot_ix = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        var_imp, pdp_data, fit = random_forest(y[boot_ix], X[boot_ix], varz, keys)
        # append to overall lists
        if b == 0:
            var_imps_df = var_imp
        else:
            # take mean importance
            var_imps_df['importance'] = pd.merge(var_imps_df, var_imp['importance'], left_index=True, right_index=True, how='outer').sum(axis=1)
        for v in varz:
            pdp_datas[v]['x'].append(pdp_data[v][1][0])
            pdp_datas[v]['y'].append(pdp_data[v][0][0])
            var_imps[v].append(var_imp['importance'][var_imp['variable']==v].values[0])

    # combine results
    var_imps_df['importance'] /= nboot

    return var_imps_df, var_imps, pdp_datas

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

def plot_rf_results(var_imp_df, var_imp_list, pdp_data, mean_val, exp_name, mod_number):
    ## have the plot function here for now
    plot_dir = '../outputs/{}/model_{}/plots/'.format(exp_name, mod_number)
    ## A. variable importance
    # format the data for plotting
    var_imp_df = var_imp_df.sort_values(['key','importance'], ascending=False)
    var_imp_df['ix'] = np.arange(var_imp_df.shape[0])
    colors = {'land' : 'b', 'climate' : 'r', 'agents' : 'k'}
    var_imp_df['color'] = np.nan
    for index, row in var_imp_df.iterrows():
        var_imp_df.loc[index, 'color'] = colors[row.key]
    # format the error bars
    var_imp_df = var_imp_df.assign(upr=np.nan, lwr=np.nan)
    for k, v in var_imp_list.items():
        var_imp_df.loc[var_imp_df['variable']==k, 'upr'] = np.percentile(v, q=[95])
        var_imp_df.loc[var_imp_df['variable']==k, 'lwr'] = np.percentile(v, q=[5])
    print('up to here!!')
    code.interact(local=dict(globals(), **locals()))

    # create the figure
    fig, ax = plt.subplots(figsize=(5,9))
    xs = np.array(var_imp['ix'])
    ax.barh(xs, np.array(var_imp.importance), color=var_imp.color)
    ax.set_yticks(xs)
    ax.set_yticklabels(np.array(var_imp['variable']))
    ax.set_xlabel('Variable importance')
    fig.savefig(plot_dir + 'variable_importance.png')
        
    ## B. partial dependence plots
    # land
    fig = plt.figure(figsize=(15,12))
    N = 11
    axs = []
    for n in range(N):
        axs.append(fig.add_subplot(4,3,n+1))

    ## land
    n_land = 6
    for i in range(n_land):
        var = var_imp[var_imp.key=='land'].iloc[i]['variable']
        axs[i].plot(pdp_data[var][1][0], pdp_data[var][0][0]+mean_val)
        axs[i].set_xlabel(var)

    # agents
    for j in range(3):
        var = var_imp[var_imp.key=='agents'].iloc[j]['variable']
        axs[i+j+1].plot(pdp_data[var][1][0], pdp_data[var][0][0]+mean_val)
        axs[i+j+1].set_xlabel(var)

    # climate
    for k in range(2):
        var = var_imp[var_imp.key=='climate'].iloc[k]['variable']
        axs[i+j+k+2].plot(pdp_data[var][1][0], pdp_data[var][0][0]+mean_val)
        axs[i+j+k+2].set_xlabel(var)

    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.set_ylim([0,1])
        if a % 3 == 0:
            ax.set_ylabel('P(CC>ins)')
        else:
            ax.set_yticklabels([])

    axs[0].set_title('LAND')
    axs[6].set_title('AGENTS')
    axs[9].set_title('CLIMATE')

    fig.savefig(plot_dir + 'partial_dependence.png')

    # code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()