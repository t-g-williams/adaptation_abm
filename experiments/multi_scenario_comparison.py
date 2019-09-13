'''
multi-replication scenario analysis
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.multi_scenario as plt_multi
import plot.single_run as plt
import imp
import numpy as np
import pickle
import copy

def main():
    nreps = 1000
    exp_name = 'multi_scenario_compare'

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    f = '../outputs/POM/10000_10reps/input_params.pkl'
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    T = 100
    inp_base['model']['T'] = T
    inp_base['model']['n_agents'] = 200
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'affording'

    # define some scenarios
    scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }

    #### RUN THE MODELS ####
    mods = multi_mod_run(nreps, inp_base, scenarios)

    #### PLOT ####
    plt_multi.main(mods, nreps, inp_base, scenarios, exp_name, T)

def multi_mod_run(nreps, inp_base, scenarios):
    mods = {}
    for name, vals in scenarios.items():
        ms = {'wealth' : [], 'organic' : [], 'coping' : [], 'n_plots' : []}
        for r in range(nreps):
            if r % 100 == 0:
                print('rep {} / {}'.format(r, nreps))
            # change the params
            params = copy.copy(inp_base)
            for k, v in vals.items():
                for k2, v2 in v.items():
                    params[k][k2] = v2
            inp_base['model']['seed'] = r # set the seed
            
            # initialize and run model
            m = mod.Model(params)
            for t in range(m.T):
                m.step()
            # append to list
            ms.append(m)
            ms['wealth'].append(m.agents.wealth)
            ms['organic'].append(m.land.organic)
            ms['n_plots'].append(m.agents.n_plots)
            ms['coping'].append(m.agents.coping_rqd)
        
        for k, v in ms.items():
            ms[k] = np.array(v)
        mods[name] = ms

    return mods

if __name__ == '__main__':
    main()