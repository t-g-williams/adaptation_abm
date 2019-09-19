'''
explore the effects of climate shocks under different model conditions
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.shock as shock_plot
import plot.single_run as plt
import calibration.POM as POM
import imp
import copy
import code
import tqdm
import numpy as np
import pandas as pd
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main():
    nreps = 1000
    exp_name = 'climate_shocks'
    ncores = 40

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    f = '../outputs/POM/constrained livestock frac/input_params_0.pkl'
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    T = 120
    inp_base['model']['T'] = T
    inp_base['model']['n_agents'] = 100
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'
    # remove in future -- only cause not in POM experiment
    inp_base['adaptation']['insurance']['cost_factor'] = 1 
    inp_base['model']['shock'] = False
    inp_base['climate']['shock_years'] = [15]
    inp_base['climate']['shock_rain'] = 0.1
    

    #### adaptation scenarios
    adap_scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    #### shock scenarios
    shock_mags = [0.1, 0.2, 0.3]
    shock_times = [10,20,30,40,50]
    T = 30 # how many years to calculate effects over

    #### RUN THE MODELS ####
    results = run_shock_sims(nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T)

    #### PLOT ####
    shock_plot.main(results, shock_mags, shock_times, exp_name)

def run_shock_sims(nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T):
    '''
    loop over the adaptation and shock scenarios
    '''
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
    scenario_results = {}

    for scenario, scenario_params in adap_scenarios.items():
        # change the params for the scenario
        params = copy.copy(inp_base)
        for k, v in scenario_params.items():
            for k2, v2 in v.items():
                params[k][k2] = v2
        
        # run baseline sims
        tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params) for i in range(len(rep_chunks)))
        base = extract_arrays(tmp)

        # run each of the shock sims
        nplots = params['agents']['n_plots_init']
        idx = pd.MultiIndex.from_product([shock_mags,shock_times], names=('mag','time'))
        diffs_pd = pd.DataFrame(index=idx, columns=nplots)
        for shock_yr in shock_times:
            for shock_mag in shock_mags:
                # add the shock conditions
                params_shock = copy.copy(params)
                params_shock['model']['shock'] = True
                params_shock['climate']['shock_years'] = [shock_yr]
                params_shock['climate']['shock_rain'] = shock_mag

                # run the model under these conditions
                tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params_shock) for i in range(len(rep_chunks)))
                
                # calculate the resilience factors
                tmp = extract_arrays(tmp)
                inc_diffs = base['income'] - tmp['income']
                # sum over the required years
                diff_sums = np.mean(inc_diffs[:,shock_yr:(shock_yr+T),:], axis=1)
                # loop over the agent types
                for n, nplot in enumerate(nplots):
                    ags = tmp['n_plots'] == nplot
                    # calculate the mean over agents and replications
                    diffs_pd.loc[(shock_mag, shock_yr), nplot] = np.mean(diff_sums[ags])

        scenario_results[scenario] = diffs_pd
    
    return scenario_results
    # code.interact(local=dict(globals(), **locals()))

def extract_arrays(tmp):
    return {
        'n_plots' : np.array([oi for tmp_i in tmp for oi in tmp_i['n_plots']]),
        'income' : np.array([oi for tmp_i in tmp for oi in tmp_i['income']]),   
    }

def run_chunk_reps(reps, params):
    '''
    run a chunk of replications
    '''
    params = copy.copy(params)
    ms = {'n_plots' : [], 'income' : []}
    # with tqdm(reps, disable = not True) as pbar:
    for r in reps:
        params['model']['seed'] = r # set the seed
        
        # initialize and run model
        m = mod.Model(params)
        for t in range(m.T):
            m.step()
        # append to list
        ms['n_plots'].append(m.agents.n_plots)
        ms['income'].append(m.agents.income.astype(int))
        # pbar.update()

    return ms

if __name__ == '__main__':
    main()