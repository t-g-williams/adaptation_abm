from model.model import Model
import model.base_inputs as base_inputs
import plot.single as plt
import code
import time
import pickle
import numpy as np
import pandas as pd
import sys

st1 = time.time()


## POM - trajectories
exp_name = 'trajectories_test/pom_10000_10rep'
f = '../outputs/{}/selected_params.pkl'.format(exp_name)
inp_all = pickle.load(open(f, 'rb'))
calib_vars = pd.read_csv('../outputs/{}/calib_vars.csv'.format(exp_name), index_col=0)

se = '0' # s+e+, s+e-, s-e+, s-e-
inputs = base_inputs.compile()
for i in range(inp_all[se].shape[1]):
    inputs[calib_vars.loc[i, 'key1']][calib_vars.loc[i,'key2']] = inp_all[se][0,i]


# f = '../outputs/2020_2_5_10/POM/200000_20reps/input_params_0.pkl'
# # f = '../outputs/2020_2_12_11/POM/100000_10reps/input_params_0.pkl'
# f = '../outputs/2020_2_12_10/POM/10000_10reps/input_params_0.pkl'
# inputs_pom = pickle.load(open(f, 'rb'))

# inputs = base_inputs.compile()
# for k, v in inputs_pom.items():
#     for k2, v2 in v.items():
#         inputs[k][k2] = v2

## change any params
inputs['model']['T'] = 30
inputs['model']['n_agents'] = 200
# inputs['agents']['read_from_file'] = False
# inputs['livestock']['consumption'] = 5
# inputs['rangeland']['gr2'] = 0.9

# temporary
inputs['land']['max_yield'] = {0 : 6590, 1 : 6590}
inputs['model']['lsla_simulation'] = True
inputs['LSLA']['outgrower'] = True
inputs['livestock']['N_production'] = 0
inputs['land']['fast_mineralization_rate'] = 0.5
inputs['land']['residue_CN_conversion'] = 200
inputs['land']['loss_max'] = 0.5
inputs['land']['max_organic_N'] = 5000

inputs['market']['fertilizer_cost'] = 0
inputs['land']['random_effect_sd'] = 0
# inputs['land']['fallow_N_add'] = 100000
inputs['livestock']['N_production'] = 0
inputs['land']['fast_mineralization_rate'] = 0.5
inputs['land']['loss_max'] = 0.5
inputs['climate']['rain_sd'] = 0


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