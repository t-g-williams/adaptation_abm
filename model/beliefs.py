'''
functions for forming and updating beliefs
'''
import numpy as np
import code
import copy

class Beliefs():
    def __init__(self, agents, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['beliefs']
        for key, val in self.inputs.items():
            setattr(self, key, val)

        N = agents.N
        T = agents.T

        # initialize belief arrays
        self.alpha = {}
        self.beta = {}
        self.mu = {}
        self.n = {}
        for q, quant in enumerate(self.quantities):
            # empty matrices
            self.alpha[quant] = np.full([T+1,N], np.nan)
            self.beta[quant] = np.full([T+1,N], np.nan)
            self.mu[quant] = np.full([T+1,N], np.nan)
            self.n[quant] = np.full([T+1,N], -99)
            # populate with initial values
            self.alpha[quant][0] = self.alpha0[q]
            self.beta[quant][0] = self.beta0[q]
            self.mu[quant][0] = self.mu0[q]
            self.n[quant][0] = self.n0[q]