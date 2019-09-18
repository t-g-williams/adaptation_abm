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
import numpy as np
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main():
    nreps = 2
    exp_name = 'multi_scenario_compare3'
    ncores = 1

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    f = '../outputs/POM/constrained livestock frac/input_params_0.pkl'
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    T = 100
    inp_base['model']['T'] = T
    inp_base['model']['n_agents'] = 1000
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'
    inp_base['adaptation']['insurance']['cost_factor'] = 1 # remove in future -- only cause not in POM experiment

    # define some scenarios
    scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    #### RUN THE MODELS ####
    mods = multi_mod_run(nreps, inp_base, scenarios, ncores)

    #### PLOT ####
    plt_multi.main(mods, nreps, inp_base, scenarios, exp_name, T)

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
            'wealth' : np.array([oi for tmp_i in tmp for oi in tmp_i['wealth']]),
            'organic' : [oi for tmp_i in tmp for oi in tmp_i['organic']], # each rep is different length
            'n_plots' : np.array([oi for tmp_i in tmp for oi in tmp_i['n_plots']]),
            'coping' : np.array([oi for tmp_i in tmp for oi in tmp_i['coping']]),
            'owners' : [oi for tmp_i in tmp for oi in tmp_i['owners']],
        }

    return mods

def run_chunk_reps(reps, params):
    '''
    run a chunk of replications
    '''
    params = copy.copy(params)
    ms = {'wealth' : [], 'organic' : [], 'coping' : [], 'n_plots' : [], 'owners' : []}
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

    return ms

if __name__ == '__main__':
    main()