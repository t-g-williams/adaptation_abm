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
# from . import analysis_poverty as msc
from . import analysis_shock as shock
import code
import tqdm
import numpy as np
import pyDOE
import pickle
import string
import pandas as pd
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
import logging
import logging.config


def main():
    exp_name = 'es_r1_sims'
    N_vars = 10000 # number of random variable sets to generate
    N_reps = 100 # number of times to repeat model for each variable set
    ncores = 40
    pom_nvars = 100000
    pom_nreps = 10
    n_mods = 1 # number of successful POM models --> default to 1
    perturb_perc = 30 # magnitude of the perturbation
    load = True # load previously-generated results
    nboot_rf = 100 # number of bootstraps for random forest
    sens_vars = {
        'agents' : ['wealth_init_mean','cash_req_mean','livestock_cost'],#,'land_area_multiplier'],
        'land' : ['organic_N_min_init','max_organic_N','fast_mineralization_rate',
            'slow_mineralization_rate','loss_max','loss_min','max_yield',
            'rain_crit','rain_cropfail_low_SOM','random_effect_sd',
            'crop_CN_conversion','residue_CN_conversion',
            'wealth_N_conversion','livestock_frac_crops','livestock_residue_factor'],
        'climate' : ['rain_mu','rain_sd']
    }
    sens_vars_clean = {
        'wealth_init_mean' : 'Initial wealth',
        'cash_req_mean' : 'Consumption requirement',
        'livestock_cost' : 'Livestock cost',
        'organic_N_min_init' : r'$SOM_{init}$',
        'max_organic_N' : r'$SOM_{max}$',
        'fast_mineralization_rate' : 'SOM mineralization rate',
        'slow_mineralization_rate' : 'Added OM mineralization rate',
        'loss_max' : 'xx',
        'loss_min' : 'xx',
        'max_yield' : 'Maximum yield',
        'rain_crit' : 'xx',
        'rain_cropfail_low_SOM' : 'xx',
        'random_effect_sd' : 'xx',
        'crop_CN_conversion' : 'xx',
        'residue_CN_conversion' : 'xx',
        'wealth_N_conversion' : 'Livestock N\nconversion factor',
        'livestock_frac_crops' : 'Livestock %\nresidue requirement',
        'livestock_residue_factor' : 'xx',
        'rain_mu' : 'Climate mean',
        'rain_sd' : 'Climate std. dev.'
    }
    # specify which land vars to plot? if False, plot the most important
    land_vars = ['livestock_frac_crops','wealth_N_conversion','organic_N_min_init','max_yield']

    for m in range(n_mods):
        logger.info('model {}........'.format(m))
        ### 1. load the POM variables
        f = '../outputs/{}/POM/{}_{}reps/input_params_{}.pkl'.format(exp_name, pom_nvars, pom_nreps, m)
        inp_base = pickle.load(open(f, 'rb'))
        # manually specify some variables (common to all scenarios)
        inp_base['model']['n_agents'] = 201
        inp_base['model']['exp_name'] = exp_name
        inp_base['agents']['adap_type'] = 'always'
        inp_base['agents']['land_area_multiplier'] = 1 # not in POM experiment

        ### 2. sample: generate random perturbed variable sets
        params, keys, names = hypercube_sample(N_vars, sens_vars, inp_base, perturb_perc)

        ### 3. run the policy analysis
        T_shock = [20] # measured after the burn-in
        T_res = [5]
        T_dev = 50 # for development resilience
        shock_mag = [0.1]
        inp_base['model']['T'] = T_shock[0] + T_res[0] + inp_base['adaptation']['burnin_period']
        Ys_dev = calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, T_dev, shock_mag, m, load, 'development')
        Ys_climate = calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, T_dev, shock_mag, m, load, 'climate')

        ### 4. run the random forest
        boot_dev = bootstrap_random_forest(Ys_dev, params, names, keys, nboot_rf, load, exp_name, m, 'development')
        boot_climate = bootstrap_random_forest(Ys_climate, params, names, keys, nboot_rf, load, exp_name, m, 'climate')
        # code.interact(local=dict(globals(), **locals()))
        ### 5. plot results
        # plot_rf_results(boot_climate, Ys_climate.mean(), exp_name, m, 'climate')
        # plot_rf_results(boot_dev, Ys_dev.mean(), exp_name, m, 'development')
        plot_rf_results(boot_climate, boot_dev, [Ys_climate.mean(), Ys_dev.mean()], ['shock absorption', 'poverty reduction'], 
            exp_name, m, inp_base, sens_vars_clean, land_vars=land_vars)

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

def calculate_QoI(exp_name, params, keys, names, inp_base, N_reps, ncores, T_shock, T_res, T_dev, shock_mag, mod_number, load, res_type):
    '''
    for each set of parameters, run the model and calculate the quantity of interest (qoi)
    the QoI is the probability that cover crops is preferable to insurance
    with respect to the effects on wealth of a shock at T_shock, assessed over T_res years
    '''
    exp_name_sens = exp_name + '/model_' + str(mod_number) + '/sensitivity'
    ext = '' if __name__ == '__main__' else '../'
    outdir = '{}../outputs/{}'.format(ext, exp_name_sens)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    adap_scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    param_chunks = POM.chunkIt(np.arange(params.shape[0]), ncores)
    if res_type == 'climate':
        results = Parallel(n_jobs=ncores)(delayed(chunk_QoI)(chunk, exp_name_sens, N_reps, params, keys, names,
            inp_base, adap_scenarios, shock_mag, T_shock, T_res, load) for chunk in param_chunks)
        # calculate mean over agent types for simplicity
        tmp = np.array(np.concatenate(results))
        QoIs = np.mean(tmp, axis=1)
    elif res_type == 'development':
        results = Parallel(n_jobs=ncores)(delayed(chunk_QoI_dev)(chunk, exp_name_sens, N_reps, params, keys, names,
            inp_base, adap_scenarios, shock_mag, T_dev, load) for chunk in param_chunks)
        tmp = np.array(np.concatenate(results))
        QoIs = tmp[:,1] # MIDDLE AGENTS ONLY!!!!

    return QoIs

def chunk_QoI_dev(chunk, exp_name, N_reps, params, keys, names, inp_base, adap_scenarios, shock_mag, T_dev, load):
    '''
    Calculate the QoI for a chunk of parameter sets
    for the development resilience outcome
    '''
    ncores = 1 # don't parallelize within this
    QoIs = []
    for c in (tqdm(chunk) if (0 in chunk) else chunk): # pbar will be rough
        # set the inputs
        inputs = array_to_dict(params[c], keys, names, inp_base)
        exp_name_c = exp_name + '/' + str(c)
        # run the sims
        QoIs.append(shock.run_dev_res_sims(exp_name_c, N_reps, inputs, adap_scenarios, ncores, T_dev, load=load))

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
        # note that for P(CC>ins) we want to measure cc<ins because these are income LOSSES!!
        bools = results_baseline['cover_crop'].loc[('income')] < results_baseline['insurance'].loc[('income')]
        probs = np.array(bools.mean(axis=0))
        QoIs.append(probs)

    return QoIs

def array_to_dict(params, keys, names, inp_base):
    d = copy.deepcopy(inp_base)
    for i, k in enumerate(keys):
        d[k][names[i]] = params[i]
    return d

def bootstrap_random_forest(y, X, varz, keys, nboot, load, exp_name, mod_number, res_type):
    '''
    bootstrap the data and fit random forest to each
    then calculate variable importance for each
    '''
    ext = '' if __name__ == '__main__' else '../'
    outname = '{}../outputs/{}/model_{}/sensitivity/bootstrap_random_forest_{}.pkl'.format(ext, exp_name, mod_number, res_type)
    if load and os.path.isfile(outname):
        tmp = pickle.load(open(outname, 'rb'))
        return tmp

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
    # write
    combined = {'var_imps_df' : var_imps_df, 'var_imps' : var_imps, 'pdp_datas' : pdp_datas}
    with open(outname, 'wb') as f:
        pickle.dump(combined, f, pickle.HIGHEST_PROTOCOL)

    return combined

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

def plot_rf_results(d_climate, d_dev, mean_vals, res_types, exp_name, mod_number, inp_base, sens_vars_clean, land_vars=False):
    plot_dir = '../outputs/{}/model_{}/plots/sensitivity/'.format(exp_name, mod_number)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    ### A: variable importance
    ## format the data
    tmp = d_climate['var_imps_df'].copy()
    tmp.columns.values[tmp.columns=='importance'] = 'importance_climate'
    tmp = tmp.merge(d_dev['var_imps_df'], on=('key','variable'))
    tmp.columns.values[tmp.columns=='importance'] = 'importance_development'
    # sort
    var_imp_df = tmp.sort_values(['key','importance_climate'], ascending=False)
    var_imp_df['ix'] = np.arange(var_imp_df.shape[0])
    # colors
    colors = {'land' : 'b', 'climate' : 'r', 'agents' : 'k'}
    var_imp_df['color'] = np.nan
    for index, row in var_imp_df.iterrows():
        var_imp_df.loc[index, 'color'] = colors[row.key]
    # # error bars
    var_imp_df = var_imp_df.assign(upr_climate=np.nan, upr_dev=np.nan, lwr_climate=np.nan, lwr_dev=np.nan)
    for k, v in d_climate['var_imps'].items():
        var_imp_df.loc[var_imp_df['variable']==k, 'upr_climate'] = np.percentile(v, q=[95])
        var_imp_df.loc[var_imp_df['variable']==k, 'lwr_climate'] = np.percentile(v, q=[5])
    for k, v in d_dev['var_imps'].items():
        var_imp_df.loc[var_imp_df['variable']==k, 'upr_dev'] = np.percentile(v, q=[95])
        var_imp_df.loc[var_imp_df['variable']==k, 'lwr_dev'] = np.percentile(v, q=[5])

    # create the figure
    fig, ax = plt.subplots(figsize=(9,9))
    bps = []
    xs = np.array(var_imp_df['ix'])
    for i, di in enumerate([d_climate,d_dev]):
        y_box = []
        for v in var_imp_df['variable']:
            y_box.append(di['var_imps'][v])
        y_box = np.array(y_box).transpose()
        bps.append(ax.boxplot(y_box, positions=xs+0.85+0.3*i, vert=False, patch_artist=True, 
                              showfliers=False, widths=0.25))#color=var_imp_df.color)
    ax.set_yticks(xs+1)
    ax.set_yticklabels(np.array(var_imp_df['variable']))
    ax.set_xlabel('Variable importance')
    ax.grid(False, axis='x')
    # fill with colors
    # colors = ['pink', 'lightblue', 'lightgreen']
    colors = ['0.5','1']
    for bi, bp in enumerate(bps):
        for patch in bp['boxes']:
            patch.set_facecolor(colors[bi])
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='k')
            
    ax.axhline(y=15.5, color='k',lw=2)
    ax.axhline(y=17.5, color='k',lw=2)
    xmx = ax.get_xlim()[1]
    fsz = 18
    ax.text(xmx*0.97, 15.25, 'LAND', ha='right',va='top', fontsize=fsz)
    ax.text(xmx*0.97, 17.25, 'CLIMATE', ha='right',va='top', fontsize=fsz)
    ax.text(xmx*0.97, 20.25, 'AGENTS', ha='right',va='top', fontsize=fsz)
    ylim = ax.get_ylim()

    ax.fill_between([0,xmx],[15.5,15.5],[0,0], color='k',alpha=0.05)
    # ax.fill_between([0,xmx],[15.5,15.5],[17.5,17.5], color='green',alpha=0.2)
    ax.fill_between([0,xmx],[17.5,17.5],[21,21], color='k',alpha=0.05)
    ax.set_xlim([0,xmx])
    ax.set_ylim(ylim)
    # ax.legend(bps, ['A', 'B'], loc='upper right')
    ax.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], res_types, loc='center', 
              bbox_to_anchor=[0.5,-0.1], ncol=2, frameon=False)
    fig.savefig(plot_dir + 'variable_importance_boxplot_combined.png', dpi=200)

    ####### PDPs #######
    fig, axs = plt.subplots(3,5, figsize=(15,10), sharey=True, gridspec_kw={'height_ratios':[1,1,0.01]})
    axs[1,4].remove()
    [axi.remove() for axi in axs[2,:]]
    axs_flat = list(axs.flatten())
    del axs_flat[9]

    clrs = ['k','k']
    lss = ['-','--']
    alpha=0.3

    ## land
    n_land = 4
    for i in range(n_land):
        if isinstance(land_vars,bool):
            var = var_imp_df[var_imp_df.key=='land'].iloc[i]['variable']   
        else:
            var = land_vars[i]
        for o, obj in enumerate([d_climate,d_dev]):
            ax = axs[1,i]
            pdp_data = obj['pdp_datas']
            xs = np.array(pdp_data[var]['x']).mean(axis=0) # note: this might fail if low Nreps since the PDP has < 100 values
            ys = np.percentile(np.array(pdp_data[var]['y']), q=[2.5,50,97.5], axis=0)
            ax.fill_between(xs, ys[0]+mean_vals[o], ys[2]+mean_vals[o], color=clrs[o], alpha=alpha)
            ax.plot(xs, ys[1]+mean_vals[o], color=clrs[o], label=res_types[o], ls=lss[o])
            ax.set_xlabel(sens_vars_clean[var])
            # plot the default value
            ax.text(0.5,0,'x',transform=ax.transAxes, va='center', ha='center')

    # agents
    ag_varz = ['cash_req_mean','livestock_cost','wealth_init_mean'] # impose the order
    for j in range(3):
        # var = var_imp_df[var_imp_df.key=='agents'].iloc[j]['variable']
        var = ag_varz[j]
        for o, obj in enumerate([d_climate,d_dev]):
            ax = axs[0,j]
            pdp_data = obj['pdp_datas']
            xs = np.array(pdp_data[var]['x']).mean(axis=0)
            ys = np.percentile(np.array(pdp_data[var]['y']), q=[2.5,50,97.5], axis=0)
            ax.fill_between(xs, ys[0]+mean_vals[o], ys[2]+mean_vals[o], color=clrs[o], alpha=alpha)
            ax.plot(xs, ys[1]+mean_vals[o], color=clrs[o], label=res_types[o], ls=lss[o])
            ax.set_xlabel(sens_vars_clean[var])
            # plot the default value
            ax.text(0.5,0,'x',transform=ax.transAxes, va='center', ha='center')

    # climate
    for k in range(2):
        var = var_imp_df[var_imp_df.key=='climate'].iloc[k]['variable']
        for o, obj in enumerate([d_climate,d_dev]):
            ax = axs[0,k+3]
            pdp_data = obj['pdp_datas']
            xs = np.array(pdp_data[var]['x']).mean(axis=0)
            ys = np.percentile(np.array(pdp_data[var]['y']), q=[2.5,50,97.5], axis=0)
            ax.fill_between(xs, ys[0]+mean_vals[o], ys[2]+mean_vals[o], color=clrs[o], alpha=alpha)
            ax.plot(xs, ys[1]+mean_vals[o], color=clrs[o], ls=lss[o])
            ax.set_xlabel(sens_vars_clean[var])
            # plot the default value
            ax.text(0.5,0,'x',transform=ax.transAxes, va='center', ha='center')

    labs = 5
    for a, ax in enumerate(axs_flat):
        ax.grid(False)
        ax.text(0.01, 0.99, string.ascii_uppercase[a], fontsize=16, transform=ax.transAxes, ha='left', va='top')
        # code.interact(local=dict(globals(), **locals()))
        try:
            ax.tick_params(axis='x', which='major', pad=10)
        except:
            pass
        if a % 5 == 0:
            ax.set_ylabel(r'P(CC$\succ$ins)')

    axs[0,0].text(0, 1.1, '\nAGENTS', transform=axs[0,0].transAxes, fontsize=22)
    axs[0,3].text(0, 1.1, 'CLIMATE', transform=axs[0,3].transAxes, fontsize=22)
    axs[1,0].text(0, 1.1, 'LAND', transform=axs[1,0].transAxes, fontsize=22)

    lg = fig.legend(res_types, bbox_to_anchor=[0.5,0.01], ncol=2, loc='center', frameon=False)
    fig.savefig(plot_dir + 'partial_dependence_combined.png', bbox_extra_artists =(lg,), dpi=200) #  bbox_inches='tight', 

if __name__ == '__main__':
    logging.config.fileConfig('logger.conf', defaults={'logfilename' : 'logs/{}.log'.format(os.path.basename(__file__)[:-3])})
    logger = logging.getLogger('sLogger')
    main()