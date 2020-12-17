'''
assess synergies/tradeoffs over time
'''
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import xarray
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import model.model as mod
import model.base_inputs as inp
import calibration.POM as POM
import plot.poverty as plt_pov
from experiments import analysis_shock
from experiments import analysis_poverty
import seaborn as sns
import copy
import sys
import code
import tqdm
import numpy as np
import string
import pandas as pd
import pickle
import copy
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import logging
import logging.config

import warnings
warnings.simplefilter("ignore", UserWarning) # ImageGrid spits some annoying harmless warnings w.r.t. tight_layout

def main():
    exp_name_POM = 'es_r2' # for reading POM outputs
    exp_name_base = 'es_r2' # for writing outputs
    solution_number = 0 # the id number of the POM solutions
    ncores = 40 # number of cores for parallelization
    load = True # load pre-saved outputs?
    nreps = 300 # for the simulation

    exp_name = '{}/model_{}'.format(exp_name_base, solution_number)
    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    pom_nvars = 100000
    pom_nreps = 10
    f = '../outputs/{}/POM/{}_{}reps/input_params_{}.pkl'.format(exp_name_POM, pom_nvars, pom_nreps, solution_number)
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    inp_base['model']['n_agents'] = 200
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always' # agents always choose the adaptation option
    inp_base['model']['shock'] = False
    T_dev = 50 # time period for development resilience simulations

    #### adaptation scenarios
    adap_scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
        'both' : {'model' : {'adaptation_option' : 'both'}},
    }

    ### SHOCK RESILIENCE
    # [results, results_baseline, results_dev] = assess_synergies(exp_name, inp_base, adap_scenarios, load, ncores, nreps, T_dev)
    # plot_synergy(exp_name, adap_scenarios, inp_base, results, results_baseline, results_dev)

    #### POVERTY REDUCTION
    inp_base['model']['T'] = T_dev + inp_base['adaptation']['burnin_period']
    mods = analysis_poverty.multi_mod_run(nreps, inp_base, adap_scenarios, ncores)
    plt_pov.main(mods, nreps, inp_base, adap_scenarios, exp_name, T_dev, dir_ext='synergy/')

def assess_synergies(exp_name, inp_base, adap_scenarios, load, ncores, nreps, T_dev):
    '''
    compare the strategies over the dimensions of shock (t_shock, t_res)
    '''
    shock_mags = [0.1,0.2]#,0.3]
    shock_times = np.arange(2,51,step=2) # measured after the burn-in period (T_shock in paper)
    T_res = np.arange(1,15) # how many years to calculate effects over
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    outcomes = ['wealth','income']

    #### RUN THE MODELS ####
    t1 = time.time()
    exp_name_syn = exp_name + '/synergy'
    results, results_baseline = analysis_shock.run_shock_sims(exp_name_syn, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
    results_dev = analysis_shock.run_dev_res_sims(exp_name_syn, nreps, inp_base, adap_scenarios, ncores, T_dev, load=load, return_all=True)
    t2 = time.time()
    print('{} seconds'.format(t2-t1))
    return results, results_baseline, results_dev

def plot_synergy(exp_name, adap_scenarios, inp_base, shock, base, dev):
    '''
    for shock resilience:
    plot synergy -- i.e., "both" > "cover_crop" + "insurance"
    dev = a dict. each entry is of size [land, replication]
    '''
    savedir = '../outputs/{}/plots/synergy/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    #### SHOCK RESILIENCE
    # subset the dataframes to what we care about (NOTE: this might not work first time round... re-run the code and it does (only works when importing CSV))
    both_all = base['both'].loc[('income','0_2'),'1.5']
    cc_all = base['cover_crop'].loc[('income','0_2'),'1.5']
    ins_all = base['insurance'].loc[('income','0_2'),'1.5']
    none_all = base['baseline'].loc[('income','0_2'),'1.5']
    both = both_all.groupby(level=[0,1]).mean() # take the mean over the replications
    cc = cc_all.groupby(level=[0,1]).mean() # take the mean over the replications
    ins = ins_all.groupby(level=[0,1]).mean() # take the mean over the replications
    none = none_all.groupby(level=[0,1]).mean() # take the mean over the replications

    #### plot the effects separately
    titls = ['cover_crop','insurance','both']
    plot_vals = [cc-none,ins-none,both-none]
    mm = np.abs(np.array(plot_vals)).max()
    fig = plt.figure(figsize=(6*3, 5))
    axs = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.5, add_all=True, label_mode='L',
        cbar_mode='single',cbar_location='right', aspect=False)
    for a, ax in enumerate(axs):
        hm = (-plot_vals[a]).to_xarray().plot(ax=ax, cmap='bwr',add_colorbar=False, vmin=-mm,vmax=mm)# plot the -ve b/c -ve values are good
        ax.set_title(titls[a])
        ax.set_xlabel(r'Time of shock ($T_{shock}$)')

    axs[0].set_ylabel(r'Assessment period ($T_{assess}$)')
    cax = axs.cbar_axes[0]
    cbar = cax.colorbar(hm)
    axis = cax.axis[cax.orientation]
    axis.label.set_text("Average income benefit (birr/yr)")
    fig.savefig(savedir + 'shock_resilience_grid.png', dpi=200,bbox_inches='tight')

    ### synergies
    fig = plt.figure(figsize=(7, 5))
    ax = ImageGrid(fig, 111, nrows_ncols=(1,1), axes_pad=0.5, add_all=True, label_mode='L',
        cbar_mode='single',cbar_location='right', aspect=False)
    plt_vals_prob = (both_all<=(cc_all+ins_all-none_all)).groupby(level=[0,1]).mean() # <= b/c -ve is beneficial
    plt_vals = -(both-(cc+ins-none))
    # mm = np.abs(np.array(plt_vals)).max()
    hm = (plt_vals_prob).to_xarray().plot(ax=ax[0], cmap='bwr',add_colorbar=False, vmin=0,vmax=1) # plot the -ve b/c -ve values are good
    # code.interact(local=dict(globals(), **locals()))  
    cax = ax.cbar_axes[0]
    cbar = cax.colorbar(hm)
    axis = cax.axis[cax.orientation]
    # axis.label.set_text("Synergy (E[birr/year])")
    axis.label.set_text("P(synergy)")
    ax[0].set_ylabel(r'Assessment period ($T_{assess}$)')
    ax[0].set_xlabel(r'Time of shock ($T_{shock}$)')
    fig.savefig(savedir + 'synergy_grid.png', dpi=200,bbox_inches='tight')
    plt.close('all')
    # sys.exit()

    ## are benefits of both greater than the single option?
    fig = plt.figure(figsize=(6*2, 5))
    axs = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5, add_all=True, label_mode='L',
        cbar_mode='single',cbar_location='right', aspect=False)
    plt_vals = [-(both-cc),-(both-ins)]
    plt_vals_prob = [((-(both_all-cc_all))>0).groupby(level=[0,1]).mean(),((-(both_all-ins_all))>0).groupby(level=[0,1]).mean()]
    # mm = np.abs(np.array(plt_vals)).max()


    # make a color map
    from matplotlib.colors import LinearSegmentedColormap
    cmaps = [LinearSegmentedColormap.from_list('cm_cc', [(1, 0, 0), (1, 1, 1), (0, 1, 0)]), # R -> W -> G
                LinearSegmentedColormap.from_list('cm_ins', [(0, 0, 1), (1, 1, 1), (0, 1, 0)])] # B -> W -> G
    cmaps = [LinearSegmentedColormap.from_list('cm_cc', [(0, 0, 0), (1, 1, 1), (0, 0.5, 0)]), # k -> W -> G
                LinearSegmentedColormap.from_list('cm_ins', [(0, 0, 0), (1, 1, 1), (0, 0.5, 0)])] # k -> W -> G  
    # cmaps = ['bwr','bwr'] # default ones....
    # cmaps = [sns.diverging_palette(145, 300, s=60, as_cmap=True),sns.diverging_palette(145, 300, s=60, as_cmap=True)]
    titls = [r"P(both$\succ$cover_crop)", r"P(both$\succ$insurance)"]
    for a, ax in enumerate(axs):
        # hm = (plt_vals[a]).to_xarray().plot(ax=ax, cmap='bwr',add_colorbar=False, vmin=-mm,vmax=mm)# plot the -ve b/c -ve values are good
        hm = (plt_vals_prob[a]).to_xarray().plot(ax=ax, cmap=cmaps[a],add_colorbar=False, vmin=0,vmax=1)# plot the -ve b/c -ve values are good
        ax.set_xlabel(r'Time of shock ($T_{shock}$)')
        # ax.set_title(titls[a])
        ax.text(-0.05, 1.03, '{}: {}'.format(string.ascii_uppercase[a], titls[a]), fontsize=24, transform=ax.transAxes, ha='left',va='bottom')
    axs[0].set_ylabel(r'Assessment period ($T_{assess}$)')
    cax = axs.cbar_axes[0]
    cbar = cax.colorbar(hm)
    axis = cax.axis[cax.orientation]
    axis.label.set_text("Additional average income benefit (birr/yr)")
    axis.label.set_text("Probability")
    fig.savefig(savedir + 'synergy_grid_comparison_prob.png', dpi=200,bbox_inches='tight')
    # sys.exit() 




if __name__ == '__main__':
    logging.config.fileConfig('logger.conf', defaults={'logfilename' : 'logs/{}.log'.format(os.path.basename(__file__)[:-3])})
    logger = logging.getLogger('sLogger')
    main()