'''
add LSLA to the model
'''

from model.model import Model
import model.base_inputs as base_inputs
import code
import time
import pickle
import numpy as np
import sys
def main():

    #### define LSLA parameters ####
    lsla_params = {
        'tstart' : 5, # e.g. 5 means start of 6th year of simulation
        'employment' : 2, # jobs/ha taken
        'LUC' : 'farm', # 'farm' or 'commons'' or ?'none'?
        'encroachment' : 'farm', # where do displaced HHs encroach on? 'farm' or 'commons'
        'frac_retain' : 0.5, # fraction of land that was originally taken that HHs retain (on average)
        'land_distribution_type' : 'amt_lost', # current_holdings: proportional to current holdings, 'equal_hh' : equal per hh, "equal_pp" : equal per person
        'land_taking_type' : 'random', # random or equalizing
        }



    #### load inputs ####
    # inputs = base_inputs.compile()
    f = '../outputs/2020_2_10/POM/200000_20reps/input_params_0.pkl'
    inputs = pickle.load(open(f, 'rb'))

    #### initialize the base model ####
    m = Model(inputs)
    # run the model
    for t in range(m.T):
        m.step()

    st2 = time.time()
    print(st2-st1)
    # sys.exit()
    # plot
    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()