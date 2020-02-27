from model.model import Model
import model.base_inputs as inputs
import plot.single_run as plt
import code
import time
import pickle
import numpy as np
import sys

# compile the inputs
inp = inputs.compile()

#### OR ####

# load from POM experiment
f = '../outputs/2019_10_10/POM/100000_10reps/input_params_0.pkl'
inp = pickle.load(open(f, 'rb'))

## change any params, e.g.
inp['model']['T'] = 60
inp['model']['n_agents'] = 30
inp['model']['exp_name'] = 'test'
inp['agents']['land_area_multiplier'] = 1

# initialize the model
print('running model....')
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

# plot
print('plotting....')
plt.main(m)
# code.interact(local=dict(globals(), **locals()))