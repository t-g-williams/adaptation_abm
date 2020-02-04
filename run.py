from model.model import Model
import model.base_inputs as base_inputs
import plot.single_run as plt
import code
import time
import pickle
import numpy as np
import sys

st1 = time.time()
# compile the inputs
inp_base = base_inputs.compile()
inp = inp_base
inputs = inp

#### OR ####

# load from POM experiment
# f = '../outputs/2019_10_15_4/POM/100000_10reps/input_params_0.pkl'
# inp = pickle.load(open(f, 'rb'))
# inp['climate']['rain_mu'] = 0.5
# inp['agents']['land_area_multiplier'] = 1
# inp['rangeland'] = inp_base['rangeland']


## change any params
inputs['model']['T'] = 50
inputs['model']['n_agents'] = 600
# inputs['agents']['jobs_availability'] *= 20
# inputs['agents']['living_cost_pp'] *= 5
# inputs['rangeland']['range_farm_ratio'] = 0.5
# inputs['rangeland']['gr2'] = 0.1
# inputs['agents']['savings_acct'] = True
# inputs['rangeland']['rangeland_dynamics'] = True

# initialize the model
m = Model(inputs)
# run the model
for t in range(m.T):
    m.step()

st2 = time.time()
print(st2-st1)
# sys.exit()
# plot
code.interact(local=dict(globals(), **locals()))
plt.main(m)

st3 = time.time()
# print(st3-st2)