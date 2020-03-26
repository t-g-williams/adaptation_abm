'''
add LSLA to the model
compare the effect relative to the baseline
'''

from model.model import Model
import model.base_inputs as base_inputs
import plot.single as plot_single
import plot.single_baseline_comparison as plot_compare
import code
import time
import pickle
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict 
from joblib import Parallel, delayed
import copy
from . import POM

def main():
    #### define LSLA parameters ####
    exp_name = 'trajectories_const_farmland'
    pom_exp = 'pom_100000_10rep'

    nrep_abm = 20
    ncores = 20

    inputs_base = base_inputs.compile()
    inputs_base['model']['T'] = 30
    inputs_base['model']['n_agents'] = 200
    inputs_base['model']['exp_name'] = exp_name

    ##### different LSLA scenarios
    ext1  = ''
    scenarios1 = {
    'baseline' : {'model' : {'lsla_simulation' : False}},
    'displ. to farms' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0.153,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'displ. to commons' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0.153,
                'LUC' : 'farm',
                'encroachment' : 'commons',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'LSLA in commons' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0.153,
                'LUC' : 'commons',
                'encroachment' : 'commons',
        }},
    'outgrower' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'outgrower' : True,
                'fert_amt' : 50,
                'irrig' : True,
                'no_fallow' : True,
        }},
    }

    #### outgrower scenarios
    ext2  = 'outgrower_'
    scenarios2 = {
    'baseline' : {'model' : {'lsla_simulation' : False}},
    'irrig + high fertilizer' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'outgrower' : True,
                'fert_amt' : 100,
                'irrig' : True,
                'no_fallow' : True,
        }},
    'irrig + low fertilizer' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'outgrower' : True,
                'fert_amt' : 20,
                'irrig' : True,
                'no_fallow' : True,
        }},
    'NO irrig + high fertilizer' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'outgrower' : True,
                'fert_amt' : 100,
                'irrig' : False,
                'no_fallow' : True,
        }},
    'NO irrig + low fertilizer' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'outgrower' : True,
                'fert_amt' : 20,
                'irrig' : False,
                'no_fallow' : True,
        }},
    }    

    # load inputs
    f = '../outputs/{}/{}/selected_params.pkl'.format(exp_name, pom_exp)
    inputs_pom = pickle.load(open(f, 'rb'))
    calib_vars = pd.read_csv('../outputs/{}/{}/calib_vars.csv'.format(exp_name, pom_exp), index_col=0)
    # flatten the params
    inp_dict = OrderedDict()
    for cat, params in inputs_pom.items():
        for mi in range(params.shape[0]):
            inp_dict['{}_{}'.format(cat, mi)] = params[mi]
    mod_names = list(inp_dict.keys())
    
    #### RUN THE MODELS ####
    exts = [ext1, ext2]
    all_scenarios = [scenarios1, scenarios2]
    for e, ext in enumerate(exts):
        print('{}....'.format(ext))
        scenarios = all_scenarios[e]

        all_mods = OrderedDict()
        for name_sc, vals in scenarios.items():
            print('    {}.....'.format(name_sc))
            # change the params
            inputs_sc = copy.deepcopy(inputs_base)
            for k, v in vals.items():
                for k2, v2 in v.items():
                    inputs_sc[k][k2] = v2

            # run the model for each model in each category
            mod_list = Parallel(n_jobs=ncores)(delayed(run_multirep_model)(i, inp_dict, inputs_sc, nrep_abm, calib_vars) for i in range(len(inp_dict)))
            mod_dict = OrderedDict()
            for mi, rep_mods in enumerate(mod_list):
                mod_dict[mod_names[mi]] = rep_mods
            all_mods[name_sc] = mod_dict

        # write to file
        fname = '../outputs/{}/{}lsla_all_models.pkl'.format(exp_name, ext)
        with open(fname,'wb') as f:
            pickle.dump(all_mods, f)

def run_multirep_model(i, param_dict, inputs_base, nreps, calib_vars):
    '''
    for a single cat/model, run nreps
    '''
    params_mod = param_dict[list(param_dict.keys())[i]]
    param = POM.overwrite_rv_inputs(inputs_base, params_mod, calib_vars.key1, calib_vars.key2)
    mod_data = {'has_livestock' : [], 'sufficient_income' : [], 'som' : [], 'grass' : []}
    for r in range(nreps):
        param['model']['seed'] = r
        m = Model(param) # ordered dict so OK
        for t in range(m.T):
            m.step()

        # indicators in the POM for trajectories
        mod_data['has_livestock'].append(m.agents.livestock>0)
        mod_data['sufficient_income'].append(~m.agents.cons_red_rqd)
        mod_data['som'].append(m.land.organic/m.land.organic_N_min_init) # fraction relative to initial
        mod_data['grass'].append(m.rangeland.R/m.rangeland.R_max/m.rangeland.R0_frac) # fraction relative to initial
    
    return mod_data

if __name__ == '__main__':
    main()