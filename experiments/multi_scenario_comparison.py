'''
multi-replication scenario analysis
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.multi_scenario as plt_multi
import plot.single_run as plt
import calibration.POM as POM
import imp
import copy
import code
import tqdm
import numpy as np
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main():
    nreps = 1000
    exp_name = 'MS_1000yrs_shock'
    ncores = 40

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    f = '../outputs/POM/constrained livestock frac/input_params_0.pkl'
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    T = 1000
    inp_base['model']['T'] = T
    inp_base['model']['n_agents'] = 100
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'
    inp_base['adaptation']['insurance']['cost_factor'] = 1 # remove in future -- only cause not in POM experiment
    # inp_base['agents']['n_plots_init'] = [15,16,17,18,19]

    #### adaptation scenarios
    scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }
    shock_years = []

    #### shock scenarios
    scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none', 'shock' : False}},
        'shock' : {'model' : {'adaptation_option' : 'none', 'shock' : True}, 'climate' : {'shock_years' : [15], 'shock_rain' : 0.1}},
    }
    shock_years = [15]

    #### RUN THE MODELS ####
    mods = multi_mod_run(nreps, inp_base, scenarios, ncores)

    #### PLOT ####
    plt_multi.main(mods, nreps, inp_base, scenarios, exp_name, T, shock_years)

def multi_mod_run(nreps, inp_base, scenarios, ncores):
    mods = {}
    for name, vals in scenarios.items():
        # change the params
        params = copy.copy(inp_base)
        for k, v in vals.items():
            for k2, v2 in v.items():
                params[k][k2] = v2

        rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
        tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params) for i in range(len(rep_chunks)))

        # format the outputs
        mods[name] = {
            'wealth' : np.array([oi for tmp_i in tmp for oi in tmp_i['wealth']]).astype(int),
            'organic' : [oi.astype(int) for tmp_i in tmp for oi in tmp_i['organic']], # each rep is different length
            'n_plots' : np.array([oi for tmp_i in tmp for oi in tmp_i['n_plots']]),
            'coping' : np.array([oi for tmp_i in tmp for oi in tmp_i['coping']]),
            'owners' : [oi for tmp_i in tmp for oi in tmp_i['owners']],
            'income' : np.array([oi for tmp_i in tmp for oi in tmp_i['income']]).astype(int),
        }

    return mods

def run_chunk_reps(reps, params):
    '''
    run a chunk of replications
    '''
    params = copy.copy(params)
    ms = {'wealth' : [], 'organic' : [], 'coping' : [], 'n_plots' : [], 'owners' : [],
        'income' : []}
    # with tqdm(reps, disable = not True) as pbar:
    for r in reps:
        params['model']['seed'] = r # set the seed
        
        # initialize and run model
        m = mod.Model(params)
        for t in range(m.T):
            m.step()
        # append to list
        ms['wealth'].append(m.agents.wealth)
        ms['organic'].append(m.land.organic)
        ms['n_plots'].append(m.agents.n_plots)
        ms['coping'].append(m.agents.coping_rqd)
        ms['owners'].append(m.land.owner)
        ms['income'].append(m.agents.income)
        # pbar.update()

    return ms

if __name__ == '__main__':
    main()