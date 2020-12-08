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
f = '../outputs/es_r2/POM/100000_10reps/input_params_0.pkl'
inp = pickle.load(open(f, 'rb'))
# code.interact(local=dict(globals(), **locals()))

## change any params, e.g.
inp['model']['T'] = 100
inp['model']['n_agents'] = 100
inp['model']['exp_name'] = 'test'
inp['agents']['land_area_multiplier'] = 1
inp['agents']['fodder_constraint'] = True
inp['climate']['shock_as_percentile'] = False
inp['adaptation']['cover_crop']['climate_dependence'] = True

# initialize the model
print('running model....')
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

# plot
print('plotting....')
plt.main(m)

## temporary: replacing inp_base with inp_POM
for k, v in inp.items():
    for k2, v2 in v.items():
        inp_base[k][k2] = v2
fdir = '../outputs/es_r22/POM/100000_10reps/'
import os
os.makedirs(fdir)
with open(fdir + 'input_params_0.pkl', 'wb') as f:
    pickle.dump(inp_base, f)