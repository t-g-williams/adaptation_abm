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
import sys
import copy

def main():
    #### define LSLA parameters ####
    exp_name = '2020_2_12_10'
    pom_nreps = '10000_10reps'

    nreps = 100

    # load inputs
    f = '../outputs/{}/POM/{}/input_params_0.pkl'.format(exp_name, pom_nreps)
    inputs_pom = pickle.load(open(f, 'rb'))
    # params not in POM
    inputs = base_inputs.compile()
    for k, v in inputs_pom.items():
        for k2, v2 in v.items():
            inputs[k][k2] = v2

    inputs['model']['T'] = 50
    inputs['model']['n_agents'] = 400

    inputs['land']['max_yield'] = {0 : 6590, 1 : 6590}

    # other inputs
    # inputs['rangeland']['R0_frac'] = 0.6

    ##### different LSLA scenarios
    ext  = ''
    scenarios = {
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
    ext  = 'outgrower_'
    scenarios = {
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

    #### scenarios -- safeguarding livelihoods ####
    ext = 'farm_disp_'
    scenarios = {
        'baseline' : {'model' : {'lsla_simulation' : False}},
        'displ. to farms + employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0.153,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
            'LSLA + cover crop' : {'model' : {'lsla_simulation' : True, 
                'adaptation_option' : 'cover_crop'},
            'adaptation' : {'burnin_period' : 15},
             'LSLA' : {
                'employment' : 2,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
            'LSLA + insurance' : {'model' : {'lsla_simulation' : True, 
                'adaptation_option' : 'insurance'},
            'adaptation' : {'burnin_period' : 15},
             'LSLA' : {
                'employment' : 2,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    }

    ## loop over the scenarios
    mods = {}
    for name, vals in scenarios.items():
        print(name)
        mods[name] = []
        # change the params
        params = copy.deepcopy(inputs)
        for k, v in vals.items():
            for k2, v2 in v.items():
                params[k][k2] = v2
        params['model']['exp_name'] = exp_name + '/' + name

        for r in range(nreps):
            # initialize and run model
            params['model']['seed'] = r
            m = Model(params)
            for t in range(m.T):
                m.step()
            mods[name].append(m)

        # plot -- just one replication
        # plot_single.main(m)

    print('plotting all...')
    plot_compare.main(mods, exp_name, ext, relative=True)
    plot_compare.main(mods, exp_name, ext, relative=False)
    # code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()