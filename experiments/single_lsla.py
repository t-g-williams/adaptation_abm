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

    # load inputs
    f = '../outputs/{}/POM/{}/input_params_0.pkl'.format(exp_name, pom_nreps)
    inputs_pom = pickle.load(open(f, 'rb'))
    # params not in POM
    inputs = base_inputs.compile()
    for k, v in inputs_pom.items():
        for k2, v2 in v.items():
            inputs[k][k2] = v2

    inputs['model']['T'] = 30
    inputs['model']['n_agents'] = 400

    # other inputs
    # inputs['rangeland']['R0_frac'] = 0.6

    # specify the scenarios
    scenarios = {
    'baseline' : {'model' : {'lsla_simulation' : False}},
    'displ. to farms + employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 2,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'displ. to farms + NO employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0,
                'LUC' : 'farm',
                'encroachment' : 'farm',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'displ. to commons + employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 2,
                'LUC' : 'farm',
                'encroachment' : 'commons',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'displ. to commons + NO employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0,
                'LUC' : 'farm',
                'encroachment' : 'commons',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'LSLA in commons + employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 2,
                'LUC' : 'commons',
                'encroachment' : 'commons',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    'LSLA in commons + NO employment' : {'model' : {'lsla_simulation' : True},
             'LSLA' : {
                'employment' : 0,
                'LUC' : 'commons',
                'encroachment' : 'commons',
                'frac_retain' : 0.5,
                'land_distribution_type' : 'amt_lost', # amt_lost 'equal_hh'  "equal_pp"
                'land_taking_type' : 'equalizing', # random or equalizing
        }},
    }

    ## loop over the scenarios
    mods = {}
    for name, vals in scenarios.items():
        # change the params
        params = copy.copy(inputs)
        for k, v in vals.items():
            for k2, v2 in v.items():
                params[k][k2] = v2
        params['model']['exp_name'] = exp_name + '/' + name
        # initialize and run model
        m = Model(params)
        for t in range(m.T):
            m.step()
        mods[name] = m

        # plot
        plot_single.main(m)

    print('plotting all...')
    plot_compare.main(mods, exp_name, relative=True)
    plot_compare.main(mods, exp_name, relative=False)
    # code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()