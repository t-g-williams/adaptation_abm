from model.model import Model
import model.base_inputs as base_inputs
import plot.single as plt
import code
import time
import pickle
import numpy as np
import sys

st1 = time.time()

f = '../outputs/2020_2_5_10/POM/200000_20reps/input_params_0.pkl'
f = '../outputs/2020_2_12_8/POM/10000_10reps/input_params_0.pkl'
inputs_pom = pickle.load(open(f, 'rb'))

inputs = base_inputs.compile()
for k, v in inputs_pom.items():
    for k2, v2 in v.items():
        inputs[k][k2] = v2

## change any params
inputs['model']['T'] = 20
inputs['model']['n_agents'] = 21
# inputs['agents']['read_from_file'] = False
# inputs['livestock']['consumption'] = 500


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