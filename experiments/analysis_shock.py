'''
explore the effects of climate shocks under different model conditions
to run this in commandline:
    1. navigate to the main code folder
    2. type: python -m experiments.shock_analysis
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.shock as shock_plot
import plot.single_run as plt
import calibration.POM as POM
from experiments import convergence
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
import logging
import logging.config
logger = logging.getLogger('sLogger')


def main():
    exp_name_POM = 'es_r1_sims' # for reading POM outputs
    exp_name_base = 'es_r1_sims' # for writing outputs
    solution_numbers = [0] # the id numbers of the POM solutions
    ncores = 10 # number of cores for parallelization
    load = True # load pre-saved outputs?
    nreps = 300 # for the simulation

    for solution_number in solution_numbers:
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
        # inp_base['agents']['land_area_multiplier'] = 1
        # inp_base['adaptation']['cover_crop']['climate_dependence'] = False
        # print('50% residue lost!!!!')

        #### adaptation scenarios
        adap_scenarios = {
            'baseline' : {'model' : {'adaptation_option' : 'none'}},
            'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
            'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
        }

        # 0: convergence analysis
        # nreps_req = convergence.convergence_analysis(exp_name, inp_base, adap_scenarios, ncores)

        # ## A: resilience as function of T_res, T_shock
        assess_resilience(exp_name, inp_base, adap_scenarios, load, ncores, nreps)

        # # ## B: vary shock magnitude
        # vary_magnitude(exp_name, inp_base, adap_scenarios, load, ncores, nreps)

        # # # ## C: effect of policy design
        # policy_design(exp_name, inp_base, adap_scenarios, load, ncores, nreps)

def policy_design(exp_name, inp_base, adap_scenarios, load, ncores, nreps):
    '''
    explore the effect of the policy parameters on the response
    use a single shock magnitude
    '''
    ## shock settings
    shock_mags = [0.2] # code only designed to have one value in this list
    shock_times = [10] # np.arange(2,31,step=2) # measured after the burn-in period
    T_res = [1,3,5,7,9]# np.arange(1,16) # how many years to calculate effects over
    T_dev = 20 # time period for development resilience simulations
    outcomes = ['wealth','income']
    inp_base['model']['T'] = shock_times[-1] + T_res[-1] + inp_base['adaptation']['burnin_period'] + 1
    ## parameter settings
    res = 20 # resolution
    cc_N_fix = np.linspace(40,200,res).astype(int)
    cc_cost_factor = np.round(np.linspace(0.1,4,res), 2)
    ins_percentile = np.round(np.linspace(0.01, 0.3,res), 3)
    ins_cost_factor = np.round(np.linspace(0.1,4,res), 2)

    ## set up outputs
    res_cc = {}
    res_ins = {}
    for o in outcomes:
        for obj in [res_cc, res_ins]:
            obj[o] = []
    dev_cc = pd.DataFrame(index=pd.MultiIndex.from_product([cc_N_fix, cc_cost_factor], names=('N_fix','cost_factor')), columns=inp_base['agents']['land_area_init'])
    dev_ins = pd.DataFrame(index=pd.MultiIndex.from_product([ins_percentile, ins_cost_factor], names=('percentile','cost_factor')), columns=inp_base['agents']['land_area_init'])

    #### cover crops ####
    logger.info('LEGUME COVER ......')
    all_cc_outname = '../outputs/'+exp_name+'/policy_design/cover_crop/{}/combined_{}res_dev{}.pkl'.format(shock_mags[0], res, T_dev)
    if os.path.isfile(all_cc_outname):
        tmp = pickle.load(open(all_cc_outname, 'rb'))
        res_cc = tmp['res_cc']
        dev_cc = tmp['dev_cc']
    else:
        for ci, cc_N in enumerate(cc_N_fix):
            logger.info('  N = {} / {}'.format(ci+1,len(cc_N_fix)))
            for cc, cc_cost in enumerate(cc_cost_factor):
                # logger.info('    cost = {} / {}'.format(cc+1,len(cc_cost_factor)))
                exp_name_pol = exp_name + '/policy_design/cover_crop/{}/{}N_{}cost'.format(shock_mags[0], cc_N, str(cc_cost).replace('.','_'))

                # change the inputs
                inp_tmp = copy.deepcopy(inp_base)
                inp_tmp['adaptation']['cover_crop']['N_fixation_min'] = cc_N # NOTE: the "max" value is currently inactive
                inp_tmp['adaptation']['cover_crop']['cost_factor'] = cc_cost

                # run the models

                dev_cc.loc[(cc_N, cc_cost)] = run_dev_res_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, ncores, T_dev, load=load)
                results, results_baseline = run_shock_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
                # note: just work with the results_baseline for now

                for o in outcomes:
                    # pre-run P(cc>ins) calculations to reduce size in memory
                    bools = results_baseline['cover_crop'].loc[(o)] < results_baseline['insurance'].loc[(o)]
                    probs = bools.astype(int).groupby(level=[0,1,2]).mean() # take the mean over the replications. use astype(int) in case they are all true or false
                    # add the policy info into the results
                    probs['cc_N_fix'] = cc_N
                    probs['cc_cost'] = cc_cost
                    probs.columns = probs.columns.astype(str) # some issues with area not being saved as a string
                    # add to the master list
                    res_cc[o].append(probs.set_index(['cc_N_fix', 'cc_cost'], append=True))
            
        # join all dataframes together
        for k,v in res_cc.items():
            res_cc[k] = pd.concat(v)
        # write
        with open(all_cc_outname, 'wb') as f:
            dict_out = {}
            dict_out['res_cc'] = res_cc
            dict_out['dev_cc'] = dev_cc
            pickle.dump(dict_out, f, pickle.HIGHEST_PROTOCOL)

    #### insurance ####
    logger.info('INSURANCE ......')
    all_ins_outname = '../outputs/'+exp_name+'/policy_design/insurance/{}/combined_{}res_dev{}.pkl'.format(shock_mags[0], res, T_dev)
    if os.path.isfile(all_ins_outname):
        tmp = pickle.load(open(all_ins_outname, 'rb'))
        res_ins = tmp['res_ins']
        dev_ins = tmp['dev_ins']
    else:
        for i, ins_perc in enumerate(ins_percentile):
            logger.info('  perc = {} / {}'.format(i+1, len(ins_percentile)))
            for j, ins_cost in enumerate(ins_cost_factor):
                # logger.info('    cost = {} / {}'.format(j+1, len(ins_cost_factor)))
                exp_name_pol = exp_name + '/policy_design/insurance/{}/{}perc_{}cost'.format(shock_mags[0], ins_perc, str(ins_cost).replace('.','_'))

                # change the inputs
                inp_tmp = copy.deepcopy(inp_base)
                inp_tmp['adaptation']['insurance']['climate_percentile'] = ins_perc
                inp_tmp['adaptation']['insurance']['cost_factor'] = ins_cost

                # run the models
                dev_ins.loc[(ins_perc, ins_cost)] = run_dev_res_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, ncores, T_dev, load=load)
                results, results_baseline = run_shock_sims(exp_name_pol, nreps, inp_tmp, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
                # note: just work with the results_baseline for now

                for o in outcomes:
                    # pre-run P(cc>ins) calculations to reduce size in memory
                    bools = results_baseline['cover_crop'].loc[(o)] < results_baseline['insurance'].loc[(o)]
                    probs = bools.astype(int).groupby(level=[0,1,2]).mean() # take the mean over the replications. use astype(int) in case they are all true or false
                    # add the policy info into the results
                    probs['ins_perc'] = ins_perc
                    probs['ins_cost'] = ins_cost
                    probs.columns = probs.columns.astype(str) # some issues with area not being saved as a string
                    # add to the master list
                    res_ins[o].append(probs.set_index(['ins_perc', 'ins_cost'], append=True))
            
        # join all dataframes together
        for k,v in res_ins.items():
            res_ins[k] = pd.concat(v)
        # write
        with open(all_ins_outname,'wb') as f:
            dict_out = {}
            dict_out['res_ins'] = res_ins
            dict_out['dev_ins'] = dev_ins
            pickle.dump(dict_out, f, pickle.HIGHEST_PROTOCOL)

    # plot  
    shock_plot.policy_design_both_res_types(dev_cc, dev_ins, res_cc, res_ins, shock_mags, shock_times, T_res, T_dev, exp_name)
    shock_plot.policy_design_dev_res(dev_cc, dev_ins, shock_mags, exp_name)
    shock_plot.policy_design_all_combined(res_cc, res_ins, shock_mags, shock_times, T_res, exp_name)
    shock_plot.policy_design_single(res_cc, res_ins, shock_mags, shock_times, T_res, exp_name)
    # shock_plot.policy_design_all(res_cc, res_ins, shock_mags, shock_times, T_res, exp_name)
    # code.interact(local=dict(globals(), **locals()))

def vary_magnitude(exp_name, inp_base, adap_scenarios, load, ncores, nreps):
    '''
    explore the effect of varying the shock magnitude for a fixed time of shock
    (informed from part A)
    '''
    #### shock scenarios
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

def assess_resilience(exp_name, inp_base, adap_scenarios, load, ncores, nreps):
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
    exp_name_res = exp_name + '/resilience'
    results, results_baseline = run_shock_sims(exp_name_res, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=load, flat_reps=False)
    t2 = time.time()
    print('{} seconds'.format(t2-t1))
    #### PLOT ####
    shock_plot.resilience(results, shock_mags, shock_times, T_res, exp_name, False, outcomes) # baseline_resilience=False --> this means the shock effects are calculated relative to {policy,no_shock} 
    shock_plot.resilience(results_baseline, shock_mags, shock_times, T_res, exp_name, True, outcomes) # here baseline_resilience=True, so measured relative to {baseline, no_shock}

def run_dev_res_sims(exp_name, nreps, inp_base, adap_scenarios, ncores, T_dev, load=True, return_all=False):
    '''
    loop over the adaptation and shock scenarios
    '''
    outdir = '../outputs/{}'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    T_burn = inp_base['adaptation']['burnin_period']
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
    results = {}

    savename = '{}/{}reps_dev_{}yrs.csv'.format(outdir, nreps, T_dev)
    # load if results already saved
    if load and os.path.exists(savename):
        results = pickle.load(open(savename, 'rb'))
        if return_all:
            return results
        else:
            return np.mean(results['cover_crop'] > results['insurance'], axis=1)
    
    for scenario, scenario_params in adap_scenarios.items():
        # different savename for the baseline and non-baseline

        # change the params for the scenario
        params = copy.deepcopy(inp_base)
        params['model']['T'] = T_dev + T_burn
        for k, v in scenario_params.items():
            for k2, v2 in v.items():
                params[k][k2] = v2
        
        # run baseline sims
        tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params) for i in range(len(rep_chunks)))
        arrays = extract_arrays(tmp)

        # calculate the relevant output metric
        # probability that wealth > 0  at the end for each agent type
        res = []
        for land in inp_base['agents']['land_area_init']:
            vals = arrays['wealth'][:,-1,:]
            masked = np.ma.masked_array(vals, mask=~(arrays['land_area']==land))
            res.append(np.array(np.mean(masked>0, axis=1)))
        results[scenario] = np.array(res)
        
    with open(savename, 'wb') as f:
        pickle.dump(results, f)
    
    if return_all:
        return results
    else:
        output = np.mean(results['cover_crop'] > results['insurance'], axis=1)
        return output

def run_shock_sims(exp_name, nreps, inp_base, adap_scenarios, shock_mags, shock_times, ncores, T_res, outcomes, load=True, flat_reps=True):
    '''
    loop over the adaptation and shock scenarios
    '''
    outdir = '../outputs/{}'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    T_burn = inp_base['adaptation']['burnin_period']
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
    results = {}
    results_baseline = {}

    for scenario, scenario_params in adap_scenarios.items():
        logger.info(scenario)
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
        params = copy.deepcopy(inp_base)
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
            no_shock_base = no_shock # note: this fails if some scenarios have been run already -- need to re-run all

        
        # create dataframes to store outputs
        # note: the "_base" extension means that it is the difference measured from the {policy,shock} with respect to the baseline (i.e., {no_policy,no_shock} simulation)
        # whereas the other way measures the {policy,shock} relative to the {policy,no_shock}
        land_area = params['agents']['land_area_init']
        mags_str = np.array([str(si).replace('.','_') for si in shock_mags])
        if flat_reps:
            idx = pd.MultiIndex.from_product([outcomes,mags_str,T_res,shock_times], names=('outcome','mag','assess_pd','time'))
        else:
            idx = pd.MultiIndex.from_product([outcomes,mags_str,T_res,shock_times,np.arange(nreps)], names=('outcome','mag','assess_pd','time','rep'))
        df = pd.DataFrame(index=idx, columns=land_area, dtype=float).sort_index()
        diffs_pd = pd.DataFrame(index=idx, columns=land_area, dtype=float).sort_index()
        diffs_pd_base = diffs_pd.copy()

        ## run each of the shock sims
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
        # change the land area back to the std values
        m.agents.land_area = np.round(m.agents.land_area / m.agents.land_area_multiplier, 1)
        # append to list
        ms['land_area'].append(m.agents.land_area)
        ms['income'].append(m.agents.income.astype(int))
        ms['wealth'].append(m.agents.wealth.astype(int))
        # pbar.update()

    return ms

if __name__ == '__main__':
    logging.config.fileConfig('logger.conf', defaults={'logfilename' : 'logs/{}.log'.format(os.path.basename(__file__)[:-3])})
    logger = logging.getLogger('sLogger')
    main()