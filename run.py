from model.model import Model
import model.base_inputs as inputs
import plot.single_run as plt
import code
import time
import pickle
import numpy as np
import sys

st1 = time.time()
# compile the inputs
inp = inputs.compile()

#### OR ####

# load from POM experiment
f = '../outputs/2019_10_10/POM/100000_10reps/input_params_0.pkl'
inp = pickle.load(open(f, 'rb'))


## change any params
inp['model']['adaptation_option'] = 'insurance'
inp['model']['shock'] = False
inp['model']['T'] = 60
inp['model']['n_agents'] = 3
# inp['climate']['shock_years'] = [15]
# inp['climate']['shock_rain'] = 0.15
inp['model']['exp_name'] = 'test'
inp['climate']['rain_mu'] = 0.5

# initialize the model
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

st2 = time.time()
print(st2-st1)
# sys.exit()
# plot
plt.main(m)
code.interact(local=dict(globals(), **locals()))

st3 = time.time()
# print(st3-st2)