from model.model import Model
import model.base_inputs as inputs
import plot.single_run as plt
import code
import time
import pickle
import numpy as np
import sys

# compile the inputs
inp_base = inputs.compile()

#### OR ####

# load from POM experiment
# f = '../outputs/2019_10_10/POM/100000_10reps/input_params_0.pkl'
f = '../outputs/2019_10_15_4/POM/100000_10reps/input_params_0.pkl'
f = '../outputs/es_r1_fertilizer/POM/100000_10reps/input_params_0.pkl'
inp = pickle.load(open(f, 'rb'))

## change any params, e.g.
inp['model']['T'] = 100
inp['model']['n_agents'] = 100
inp['model']['exp_name'] = 'test'
inp['agents']['land_area_multiplier'] = 1
# new parameters added for beliefs / decision-making
# inp['agents']['risk_tolerance'] = inp_base['agents']['risk_tolerance']
# inp['agents']['fert_kg'] = inp_base['agents']['fert_kg']
# inp['agents']['fertilizer_cost'] = inp_base['agents']['fertilizer_cost']
# inp['agents']['nsim_utility'] = inp_base['agents']['nsim_utility']
# inp['agents']['fertilizer'] = inp_base['agents']['fertilizer'] # binary switch
# inp['agents']['fert_cash_constrained'] = inp_base['agents']['fert_cash_constrained'] # binary switch
# inp['agents']['fert_use_savings'] = inp_base['agents']['fert_use_savings'] # binary switch

# initialize the model
print('running model....')
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

# plot
code.interact(local=dict(globals(), **locals()))
print('plotting....')
plt.main(m)