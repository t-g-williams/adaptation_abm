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
# f = '../outputs/POM/different max livestock/input_params_0.pkl'
# inp = pickle.load(open(f, 'rb'))

## change any params
inp['model']['adaptation_option'] = 'none'
# inp['model']['shock'] = False
inp['model']['T'] = 1000
inp['model']['n_agents'] = 1000

# initialize the model
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

st2 = time.time()
print(st2-st1)
sys.exit()
code.interact(local=dict(globals(), **locals()))
# plot
plt.main(m)

st3 = time.time()
# print(st3-st2)