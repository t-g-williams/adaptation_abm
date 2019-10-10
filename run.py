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
f = '../outputs/2019_10_9/POM/100000_10reps/input_params_0.pkl'
inp = pickle.load(open(f, 'rb'))


## change any params
inp['model']['adaptation_option'] = 'none'
inp['model']['shock'] = True
inp['model']['T'] = 30
inp['model']['n_agents'] = 3
inp['climate']['shock_years'] = [15]
inp['climate']['shock_rain'] = 0.15

# initialize the model
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

st2 = time.time()
print(st2-st1)
code.interact(local=dict(globals(), **locals()))
sys.exit()
# plot
plt.main(m)

st3 = time.time()
# print(st3-st2)