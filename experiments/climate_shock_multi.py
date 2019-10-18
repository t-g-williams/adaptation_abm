'''
explore the effects of climate shocks under different model conditions
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.shock as shock_plot
import plot.single_run as plt
import calibration.POM as POM
import copy
import sys
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
    exp_name = '2019_10_15_4'
    ncores = 40
    load = True
    # load = False

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

    ## A: resilience as function of T_res, T_shock
    # assess_resilience(exp_name, inp_base, adap_scenarios, load, ncores)

    ## B: vary shock magnitude
    # vary_magnitude(exp_name, inp_base, adap_scenarios, load, ncores)

    ## C: effect of policy design
    policy_design(exp_name, inp_base, adap_scenarios, load, ncores)

def policy_design(exp_name, inp_base, adap_scenarios, load, ncores):
    '''
    explore the effect of the policy parameters on the response
    use a single shock magnitude
    '''
    ## shock settings
    nreps = 100
    shock_mags = [0.1]
    shock_times = np.arange(2,31,step=2) # measured after the burn-in period
    T_res = np.arange(1,16) # how many years to calculate effects over
    outcomes = ['wealth','income']
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    ## parameter settings
    cc_N_fix = np.linspace(40,200,10).astype(int)
    cc_cost_factor = np.round(np.linspace(0.25,4,10), 2)
    ins_percentile = np.round(np.linspace(0.01, 0.3,10), 3)
    ins_cost_factor = np.round(np.linspace(0.25,4,10), 2)

    ## set up outputs
    res_cc = {}
    res_ins = {}
    for o in outcomes:
        res_cc[o] = []
        res_ins[o] = []

    #### cover crops ####
    for cc_N in cc_N_fix:
        for cc_cost in cc_cost_factor:
            exp_name_pol = exp_name + '/policy_design/cover_crop/{}N_{}cost'.format(cc_N, str(cc_cost).replace('.','_'))

            # change the inputs
            inp_tmp = copy.deepcopy(inp_base)
            inp_tmp['adaptation']['cover_crop']['N_fixation_min'] = cc_N # NOTE: the "max" value is currently inactive
            inp_tmp['adaptation']['cover_crop']['cost_factor'] = cc_cost

            # run the models
            results, results_baseline = run_shock_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
            # note: just work with the results_baseline for now

            for o in outcomes:
                # pre-run P(cc>ins) calculations to reduce size in memory
                bools = results_baseline['cover_crop'].loc[(o)] < results_baseline['insurance'].loc[(o)]
                probs = bools.astype(int).groupby(level=[0,1,2]).mean() # take the mean over the replications. use astype(int) in case they are all true or false
                # add the policy info into the results
                probs['cc_N_fix'] = cc_N
                probs['cc_cost'] = cc_cost
                # add to the master list
                res_cc[o].append(probs.set_index(['cc_N_fix', 'cc_cost'], append=True))
        
    # join all dataframes together
    for k,v in res_cc.items():
        res_cc[k] = pd.concat(v)

    #### insurance ####
    for ins_perc in ins_percentile:
        for ins_cost in ins_cost_factor:
            exp_name_pol = exp_name + '/policy_design/insurance/{}perc_{}cost'.format(ins_perc, str(ins_cost).replace('.','_'))

            # change the inputs
            inp_tmp = copy.deepcopy(inp_base)
            inp_tmp['adaptation']['insurance']['climate_percentile'] = ins_perc
            inp_tmp['adaptation']['insurance']['cost_factor'] = ins_cost

            # run the models
            results, results_baseline = run_shock_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
            # note: just work with the results_baseline for now

            for o in outcomes:
                # pre-run P(cc>ins) calculations to reduce size in memory
                bools = results_baseline['cover_crop'].loc[(o)] < results_baseline['insurance'].loc[(o)]
                probs = bools.astype(int).groupby(level=[0,1,2]).mean() # take the mean over the replications. use astype(int) in case they are all true or false
                # add the policy info into the results
                probs['ins_perc'] = ins_perc
                probs['ins_cost'] = ins_cost
                # add to the master list
                res_ins[o].append(probs.set_index(['ins_perc', 'ins_cost'], append=True))
        
    # join all dataframes together
    for k,v in res_ins.items():
        res_ins[k] = pd.concat(v)

    # plot
    shock_plot.policy_design(res_cc, res_ins, shock_mags, shock_times, T_res, exp_name)
    code.interact(local=dict(globals(), **locals()))

def vary_magnitude(exp_name, inp_base, adap_scenarios, load, ncores):
    '''
    explore the effect of varying the shock magnitude for a fixed time of shock
    (informed from part A)
    '''
    #### shock scenarios
    nreps = 100
    shock_mags = np.round(np.linspace(0,0.5,10), 3)
    shock_times = [5,10,20] # keep this fixed in each plot
    T_res = np.arange(1,15) # how many years to calculate effects over
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    outcomes = ['wealth','income']

    #### RUN THE MODELS ####
    t1 = time.time()
    exp_name_mag = exp_name + '/shock_magnitude'
    results, results_baseline = run_shock_sims(exp_name_mag, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
    t2 = time.time()
    print('{} seconds'.format(t2-t1))
    #### PLOT ####
    shock_plot.shock_mag_grid_plot(results, shock_mags, shock_times, T_res, exp_name, False, outcomes)
    shock_plot.shock_mag_grid_plot(results_baseline, shock_mags, shock_times, T_res, exp_name, True, outcomes)

def assess_resilience(exp_name, inp_base, adap_scenarios, load, ncores):
    '''
    compare the strategies over the dimensions of shock (t_shock, t_res)
    '''
    nreps = 100
    shock_mags = [0.1,0.2,0.3]
    shock_times = np.arange(2,51,step=2) # measured after the burn-in period
    T_res = np.arange(1,15) # how many years to calculate effects over
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    outcomes = ['wealth','income']

    #### RUN THE MODELS ####
    t1 = time.time()
    exp_name_res = exp_name + '/resilience'
    results, results_baseline = run_shock_sims(exp_name_res, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
    t2 = time.time()
    print('{} seconds'.format(t2-t1))
    #### PLOT ####
    shock_plot.resilience(results, shock_mags, shock_times, T_res, exp_name, False, outcomes)
    shock_plot.resilience(results_baseline, shock_mags, shock_times, T_res, exp_name, True, outcomes)

def run_shock_sims(exp_name, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=True, flat_reps=True):
    '''
    loop over the adaptation and shock scenarios
    '''
    outdir = '../outputs/{}/raw'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    T_burn = inp_base['adaptation']['burnin_period']
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
    results = {}
    results_baseline = {}

    for scenario, scenario_params in adap_scenarios.items():
        # different savename for the baseline and non-baseline
        savename = '{}/{}reps_{}.csv'.format(outdir, nreps, scenario)
        savename2 = '{}/{}reps_{}_baseline.csv'.format(outdir, nreps, scenario)
        # load if results already saved
        if load and os.path.exists(savename):
            ixs = [0,1,2,3] if flat_reps else [0,1,2,3,4]
            results[scenario] = pd.read_csv(savename, index_col=ixs)
            results_baseline[scenario] = pd.read_csv(savename2, index_col=ixs)
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
        no_shock = extract_arrays(tmp)

        # for resilience assessment: subtract values from the no_intervention scenario?
        if scenario == 'baseline': # use "baseline" scenario's values as no_shock
            no_shock_base = no_shock

        ## run each of the shock sims
        land_area = params['agents']['land_area_init']
        
        # create a dataframe
        mags_str = np.array([str(si).replace('.','_') for si in shock_mags])
        if flat_reps:
            idx = pd.MultiIndex.from_product([outcomes,mags_str,T_res,shock_times], names=('outcome','mag','assess_pd','time'))
        else:
            idx = pd.MultiIndex.from_product([outcomes,mags_str,T_res,shock_times,np.arange(nreps)], names=('outcome','mag','assess_pd','time','rep'))
        df = pd.DataFrame(index=idx, columns=land_area, dtype=float).sort_index()
        diffs_pd = pd.DataFrame(index=idx, columns=land_area, dtype=float).sort_index()
        diffs_pd_base = diffs_pd.copy()

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
                for outcome in outcomes:
                    inc_diffs = no_shock[outcome] - tmp[outcome] # measure of damage: (no_shock) - (shock) --> +ve means there is damage (ie wealth/income higher in no shock)
                    inc_diffs_base = no_shock_base[outcome] - tmp[outcome]
                    # sum over the required years
                    for T in T_res:
                        xtra = 1 if outcome == 'wealth' else 0 # wealth is at END of year so one higher index than income
                        diff_sums = np.mean(inc_diffs[:,(shock_yr+T_burn+xtra):(shock_yr+T+T_burn+xtra),:], axis=1)
                        diff_sums_base = np.mean(inc_diffs_base[:,(shock_yr+T_burn+xtra):(shock_yr+T+T_burn+xtra),:], axis=1)
                        # loop over the agent types
                        for n, area in enumerate(land_area):
                            ags = tmp['land_area'] == area
                            # calculate the mean over agents and replications
                            if flat_reps:
                                means = np.mean(diff_sums[ags])
                                means_base = np.mean(diff_sums_base[ags])
                            else:
                                means = [np.mean(diff_sums[r,ags[r]]) for r in range(nreps)]
                                means_base = [np.mean(diff_sums_base[r,ags[r]]) for r in range(nreps)]
                            diffs_pd.loc[(outcome, mag_str, T, shock_yr), area] = means
                            diffs_pd_base.loc[(outcome, mag_str, T, shock_yr), area] = means_base

        results[scenario] = diffs_pd
        results_baseline[scenario] = diffs_pd_base
        diffs_pd.to_csv(savename) # write to csv
        diffs_pd_base.to_csv(savename2)

    return results, results_baseline

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