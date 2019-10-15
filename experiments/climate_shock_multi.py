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
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main():
    nreps = 105
    exp_name = '2019_10_15_4'
    ncores = 40
    load = False
    flat_reps = False

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    pom_nvars = 100000
    pom_nreps = 10
    f = '../outputs/{}/POM/{}_{}reps/input_params_0.pkl'.format(exp_name, pom_nvars, pom_nreps)
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    inp_base['model']['n_agents'] = 200
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'
    inp_base['model']['shock'] = False

    #### adaptation scenarios
    adap_scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    #### shock scenarios
    shock_mags = [0.1]#, 0.3]
    shock_times = np.arange(2,51,step=2) # measured after the burn-in period
    T_res = np.arange(1,16) # how many years to calculate effects over
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1

    t1 = time.time()
    for baseline_resilience in [True,False]: # note: this currently runs all sims twice so is inefficient
        for outcome in ['wealth','income']:
            #### RUN THE MODELS ####
            results = run_shock_sims(exp_name, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, baseline_resilience, outcome, load=load, flat_reps=flat_reps)
            #### PLOT ####
            shock_plot.main(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome)
    t2 = time.time()
    print('{} seconds'.format(t2-t1))

def run_shock_sims(exp_name, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, baseline_resilience, outcome, load=True, flat_reps=True):
    '''
    loop over the adaptation and shock scenarios
    '''
    outdir1 = '../outputs/{}/shocks/'.format(exp_name)
    outdir = '../outputs/{}/shocks/raw'.format(exp_name)
    if not os.path.isdir(outdir1):
        os.mkdir(outdir1)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    T_burn = inp_base['adaptation']['burnin_period']
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
    scenario_results = {}

    for scenario, scenario_params in adap_scenarios.items():
        ext = '_baseline' if baseline_resilience else ''
        savename = '{}/{}reps_{}{}_{}.csv'.format(outdir, nreps, scenario, ext, outcome)
        # load if results already saved
        if load and os.path.exists(savename):
            ixs = [0,1,2] if flat_reps else [0,1,2,3]
            scenario_results[scenario] = pd.read_csv(savename, index_col=ixs)
            continue

        # change the params for the scenario
        params = copy.copy(inp_base)
        for k, v in scenario_params.items():
            for k2, v2 in v.items():
                params[k][k2] = v2
        
        # run baseline sims
        if ncores > 1:
            tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params) for i in range(len(rep_chunks)))
        else:
            tmp = []
            for i in rep_chunks:
                tmp.append(run_chunk_reps(i, params))
        base = extract_arrays(tmp)

        # for resilience assessment: subtract values from the no_intervention scenario?
        if baseline_resilience:
            if scenario == 'baseline': # use "baseline" scenario's values as no_shock
                no_shock = base
        else:
            no_shock = base # use this scenario's values as no_shock

        ## run each of the shock sims
        land_area = params['agents']['land_area_init']
        
        # create a dataframe
        if flat_reps:
            idx = pd.MultiIndex.from_product([shock_mags,T_res,shock_times], names=('mag','assess_pd','time'))
        else:
            idx = pd.MultiIndex.from_product([shock_mags,T_res,shock_times,np.arange(nreps)], names=('mag','assess_pd','time','rep'))
        diffs_pd = pd.DataFrame(index=idx, columns=land_area, dtype=float).sort_index()

        for shock_yr in shock_times:
            for shock_mag in shock_mags:
                mag_str = str(shock_mag).replace('.','_')
                # add the shock conditions
                params_shock = copy.copy(params)
                params_shock['model']['shock'] = True
                params_shock['climate']['shock_years'] = [shock_yr+T_burn] # shock time is measured after the burn-in period
                params_shock['climate']['shock_rain'] = shock_mag

                # run the model under these conditions
                if ncores > 1:
                    tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params_shock) for i in range(len(rep_chunks)))
                else:
                    tmp = []
                    for i in rep_chunks:
                        tmp.append(run_chunk_reps(i, params))

                # calculate the resilience factors
                tmp = extract_arrays(tmp)
                inc_diffs = no_shock[outcome] - tmp[outcome] # measure of damage: (no_shock) - (shock) --> +ve means there is damage (ie wealth/income higher in no shock)
                # sum over the required years
                for T in T_res:
                    xtra = 1 if outcome == 'wealth' else 0 # wealth is at END of year so one higher index than income
                    diff_sums = np.mean(inc_diffs[:,(shock_yr+T_burn+xtra):(shock_yr+T+T_burn+xtra),:], axis=1)
                    # loop over the agent types
                    for n, area in enumerate(land_area):
                        ags = tmp['land_area'] == area
                        # calculate the mean over agents and replications
                        if flat_reps:
                            diffs_pd.loc[(shock_mag, T, shock_yr), area] = np.mean(diff_sums[ags])
                        else:
                            means = [np.mean(diff_sums[r,ags[r]]) for r in range(nreps)]
                            diffs_pd.loc[(shock_mag, T, shock_yr), area] = means

        scenario_results[scenario] = diffs_pd
        diffs_pd.to_csv(savename) # write to csv

    return scenario_results

def extract_arrays(tmp):
    return {
        'land_area' : np.array([oi for tmp_i in tmp for oi in tmp_i['land_area']]),
        'income' : np.array([oi for tmp_i in tmp for oi in tmp_i['income']]),   
        'wealth' : np.array([oi for tmp_i in tmp for oi in tmp_i['wealth']]),   
    }

def run_chunk_reps(reps, params):
    '''
    run a chunk of replications
    '''
    params = copy.copy(params)
    ms = {'land_area' : [], 'income' : [], 'wealth' : []}
    # with tqdm(reps, disable = not True) as pbar:
    for r in reps:
        params['model']['seed'] = r # set the seed
        
        # initialize and run model
        m = mod.Model(params)
        for t in range(m.T):
            m.step()
        # append to list
        ms['land_area'].append(m.agents.land_area)
        ms['income'].append(m.agents.income.astype(int))
        ms['wealth'].append(m.agents.wealth.astype(int))
        # pbar.update()

    return ms

if __name__ == '__main__':
    main()