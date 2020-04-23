'''
explore effects of outgrower over time
'''
import sys
import model.model as mod_code
import plot.single as plt_single
import copy
import os
import numpy as np
from collections import OrderedDict
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import model.base_inputs as base_inputs
from joblib import Parallel, delayed
import experiments.POM as POM
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import code

def main():
    exp_name = 'outgrower_explore_rndm'
    nreps = 40
    ncores = 40


    inputs = base_inputs.compile()
    inputs['model']['T'] = 100
    inputs['model']['n_agents'] = 100
    inputs['model']['seed'] = 0
    inputs['decisions']['framework'] = 'util_max'
    inputs['adaptation']['outgrower']['active'] = True

    # inputs['climate']['rain_sd'] = 0
    # inputs['land']['random_effect_sd'] = 0
    inputs['agents']['land_area_init'] = [0.25,0.5,1,2]

    inputs['adaptation']['outgrower']['fixed_price'] = True
    inputs['adaptation']['outgrower']['land_rqmt_type'] = 'fraction'

    setups = {
        '100_percent' : {'adaptation' : {'outgrower' : {'land_rqmt_type' : 'fraction', 'land_rqmt_amt' : 1}}},
        '50_percent' : {'adaptation' : {'outgrower' : {'land_rqmt_type' : 'fraction', 'land_rqmt_amt' : 0.5}}},
        '1_ha' : {'adaptation' : {'outgrower' : {'land_rqmt_type' : 'ha', 'land_rqmt_amt' : 1}}},
        '0_5_ha' : {'adaptation' : {'outgrower' : {'land_rqmt_type' : 'ha', 'land_rqmt_amt' : 0.5}}},
    }

    for name, outg_params in setups.items():
        print('{}....'.format(name))
        inp_setup = change_inputs(copy.deepcopy(inputs), outg_params)
        mods = run_models(inp_setup, nreps, ncores)
        # plotting
        outdir = '../outputs/{}'.format(exp_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        plot_absolute(mods, outdir, name, nreps)
        plot_effects(mods, outdir, name, nreps)
        plot_effects_equity(mods, outdir, name, nreps)
        plot_decisions(mods, outdir, name, nreps)
    # code.interact(local=dict(globals(), **locals()))

def run_models(inputs, nreps, ncores):
    #### RUN THE MODEL ####
    scenarios = {
        'baseline' : {'adaptation' : {'outgrower' : {'active':False}}},
        'outgrower' : {'adaptation' : {'outgrower' : {'active':True}}},
    }
    cols = {'baseline' : 'k', 'outgrower' : 'y'}

    mods = {}
    for name, vals in scenarios.items():
        # change the params
        params = change_inputs(copy.deepcopy(inputs), vals)
        # initialize and run model
        mods[name] = Parallel(n_jobs=ncores)(delayed(single_model)(r, params) for r in range(nreps))

    return mods

def single_model(r, params):
    params['model']['seed'] = r
    m = mod_code.Model(params)
    for t in range(m.T):
        m.step()
    return m

def change_inputs(params, change_dict):
    for k, v in change_dict.items():
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                for k3, v3 in v2.items():
                    params[k][k2][k3] = v3
            else:
                params[k][k2] = v2
    return params

def plot_absolute(mods, outdir, scenario, nreps):
    # format results
    fig, ax_all = plt.subplots(6,2,figsize=(12,12), sharex=True, sharey='row')

    for s, sim in enumerate(['baseline','outgrower']):
        axs = ax_all[:,s]
        objs = OrderedDict({'wealth':[],'income':[],'SOM':[],'conservation':[],'fertilizer':[],'outgrower':[]})
        land = []
        for r in range(nreps):
            objs['wealth'].append(mods[sim][r].agents.wealth)
            objs['income'].append(mods[sim][r].agents.income)
            objs['SOM'].append(mods[sim][r].land.organic)
            objs['conservation'].append(mods[sim][r].agents.choices['conservation'])
            objs['fertilizer'].append(mods[sim][r].agents.choices['fertilizer'])
            objs['outgrower'].append(mods[sim][r].agents.choices['outgrower'])
            land.append(mods[sim][r].agents.land_area)

        land = np.mean(np.array(land), axis=0)
        areas = np.unique(land)

        for k, v in objs.items():
            objs[k] = np.mean(np.array(objs[k]), axis=0)   

        ylabs = list(objs.keys())
        for o, name in enumerate(ylabs):
            for a, area in enumerate(areas):
                ags = land == area
                axs[o].plot(np.mean(objs[name][:,ags], axis=1), color=str(1-area/max(areas)), lw=2, label='{}ha'.format(area))
                # axs[o].plot(np.mean(objs[name][:,ags], axis=1), color='b', lw=0.3)
                # axs[o].plot(np.mean(objs[name], axis=1), color='k', lw=3)

            axs[o].grid(False)
            if s == 0:
                axs[o].set_ylabel(name)
            axs[o].axhline(0, color='k', lw=0.8)
            
        axs[0].legend()
        axs[-1].set_xlabel('year')
        axs[0].set_title(sim)
    
    fig.tight_layout()
    fig.savefig('{}/absolute_{}.png'.format(outdir, scenario))

def plot_effects(mods, outdir, scenario, nreps):
    # format results
    fig, axs = plt.subplots(4,1,figsize=(5,8), sharex=True)
    scs = ['outgrower','baseline']

    objs = OrderedDict({'wealth':[],'income':[],'SOM':[], 'outgrower':[]})
    for r in range(nreps):
        objs['wealth'].append(mods[scs[0]][r].agents.wealth - mods[scs[1]][r].agents.wealth)
        objs['income'].append(mods[scs[0]][r].agents.income - mods[scs[1]][r].agents.income)
        objs['SOM'].append(mods[scs[0]][r].land.organic - mods[scs[1]][r].land.organic)
        objs['outgrower'].append(mods[scs[0]][r].agents.choices['outgrower'])

    for k, v in objs.items():
        objs[k] = np.mean(np.array(objs[k]), axis=0)
        

    ylabs = list(objs.keys())
    for o, name in enumerate(ylabs):
        axs[o].plot(objs[name], color='b', lw=0.3)
        axs[o].plot(np.mean(objs[name], axis=1), color='k', lw=3)

        axs[o].grid(False)
        if name == 'outgrower':
            axs[o].set_ylabel('P(outgrower)')
        else:
            axs[o].set_ylabel(r'$\Delta\ {}$'.format(name))
        axs[o].axhline(0, color='k', lw=0.8)
        
    axs[-1].set_xlabel('year')
    fig.tight_layout()
    fig.savefig('{}/all_effects_{}.png'.format(outdir, scenario))

def plot_effects_equity(mods, outdir, scenario, nreps):
    ## EQUITY OF EFFECTS
    fig, axs = plt.subplots(4,1,figsize=(7,8), sharex=True)
    scs = ['outgrower','baseline']

    objs = OrderedDict({'wealth':[],'income':[],'SOM':[], 'land':[],'outgrower':[]})
    for r in range(nreps):
        objs['wealth'].append(mods[scs[0]][r].agents.wealth - mods[scs[1]][r].agents.wealth)
        objs['income'].append(mods[scs[0]][r].agents.income - mods[scs[1]][r].agents.income)
        objs['SOM'].append(mods[scs[0]][r].land.organic - mods[scs[1]][r].land.organic)
        objs['land'].append(mods[scs[0]][r].agents.land_area)
        objs['outgrower'].append(mods[scs[0]][r].agents.choices['outgrower'])

    for k, v in objs.items():
        objs[k] = np.mean(np.array(objs[k]), axis=0)
        
    areas = np.unique(objs['land'])
        
    ylabs = ['wealth','income','SOM','outgrower']
    for o, name in enumerate(ylabs):
        for a, area in enumerate(areas):
            ags = objs['land'] == area
            axs[o].plot(np.mean(objs[name][:,ags], axis=1), color=str(1-area/max(areas)), lw=2, label='{}ha'.format(area))

        axs[o].grid(False)
        if name =='outgrower':
            axs[o].set_ylabel(r'$P(outgrower)$')
        else:
            axs[o].set_ylabel(r'$E[\Delta\ {}]$'.format(name))
        axs[o].axhline(0, color='k', lw=0.5, ls=':')
        axs[o].legend()
        
    axs[-1].set_xlabel('year')
    fig.tight_layout()
    fig.savefig('{}/equity_effects_{}.png'.format(outdir, scenario))

def plot_decisions(mods, outdir, scenario, nreps):
    ## DECISIONS OVER TIME
    fig, axs = plt.subplots(3,1,figsize=(7,8), sharex=True)
    scs = ['outgrower','baseline']

    objs = OrderedDict({'outgrower':[],'cons':[],'cons_base':[],'fert':[],'fert_base':[],'land':[]})
    for r in range(nreps):
        objs['outgrower'].append(mods[scs[0]][r].agents.choices['outgrower'])
        objs['cons'].append(mods[scs[0]][r].agents.choices['conservation'])
        objs['cons_base'].append(mods[scs[1]][r].agents.choices['conservation'])
        objs['fert'].append(mods[scs[0]][r].agents.choices['fertilizer'])
        objs['fert_base'].append(mods[scs[1]][r].agents.choices['fertilizer'])
        objs['land'].append(mods[scs[0]][r].agents.land_area)

    for k, v in objs.items():
        objs[k] = np.mean(np.array(objs[k]), axis=0)
        
    areas = np.unique(objs['land'])

    for a, area in enumerate(areas):
        ags = objs['land'] == area
        axs[0].plot(np.mean(objs['outgrower'][1:,ags], axis=1), color=str(1-area/max(areas)), lw=2, 
                    ls='-', label='{}ha'.format(area))
        axs[1].plot(np.mean(objs['cons'][1:,ags], axis=1), color=str(1-area/max(areas)), lw=2, 
                    ls='-', label='{}ha'.format(area))
        axs[1].plot(np.mean(objs['cons_base'][1:,ags], axis=1), color=str(1-area/max(areas)), lw=2, 
                    ls=':', label='_nolegend_')
        axs[2].plot(np.mean(objs['fert'][1:,ags], axis=1), color=str(1-area/max(areas)), lw=2, 
                    ls='-', label='{}ha'.format(area))
        axs[2].plot(np.mean(objs['fert_base'][1:,ags], axis=1), color=str(1-area/max(areas)), lw=2, 
                    ls=':', label='_nolegend_')
        
    ylabs = ['outgrower','conservation','fertilizer']
    for o, name in enumerate(ylabs):
        axs[o].grid(False)
        axs[o].set_ylabel(r'$P({})$'.format(name))
        axs[o].axhline(0, color='k', lw=0.5, ls=':')
        axs[o].legend()
        
    axs[-1].set_xlabel('year')
    fig.tight_layout()
    fig.savefig('{}/decisions_{}.png'.format(outdir, scenario))

if __name__ == '__main__':
    main()