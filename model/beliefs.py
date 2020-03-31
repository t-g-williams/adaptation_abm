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
        self.var = {}
        self.n = {}
        for q, quant in enumerate(self.quantities):
            # empty matrices
            self.alpha[quant] = np.full([T+1,N], np.nan)
            self.beta[quant] = np.full([T+1,N], np.nan)
            self.mu[quant] = np.full([T+1,N], np.nan)
            self.var[quant] = np.full([T+1,N], np.nan)
            self.n[quant] = np.full([T+1,N], -99)
            # populate with initial values
            self.alpha[quant][0] = self.alpha0[q]
            self.beta[quant][0] = self.beta0[q]
            self.mu[quant][0] = self.mu0[q]
            self.n[quant][0] = self.n0[q]
            self.var[quant][0] = self.beta[quant][0] / self.alpha[quant][0] # expected variance, if tau ~ Ga(alpha,beta)

    def update(self, agents, land):
        '''
        update all beliefs at the end of the year
        all beliefs are represented using the normal - inverse gamma conjugate prior
        https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        
        - we assume the quantity x is normally distributed: x ~ N(mu, tau)
        - the beliefs for mu and tau are distributions:
                  mu ~ N(mu0, n0*tau)
                  tau ~ Ga(a, b)
        - we only are concerned in the decision-making with the _expected_ value of these distributions
                  i.e., E[mu] = mu0, E[tau] = a/b (so E[sigma2] = b/a since tau = 1/sigma2)
        '''
        t = agents.t[0]
        data_objs = {  # [numerator, denominator]
            'ag_trad' : [agents.farm_income['trad'][t], land.ha_farmed['trad'][t] * agents.ag_labor_rqmt['trad']], # [$, ha * ppl/ha = ppl]
            'ag_int' : [agents.farm_income['int'][t], land.ha_farmed['int'][t] * agents.ag_labor_rqmt['int']],
            'ag_div' : [agents.farm_income['div'][t], land.ha_farmed['div'][t] * agents.ag_labor_rqmt['div']],
            'non_farm' : [agents.salary_income[t], agents.salary_tot_consider_amt[t]], # [$, ppl]
        }

        # loop over the different beliefs
        for quant, obs_data in data_objs.items():
            ixs = obs_data[1] > 0 # ixs that have observed this period (i.e., the denominator is > 0)
            # get the observation (x)
            x = obs_data[0][ixs] / obs_data[1][ixs] # x = the observation
            n = 1 # i.e., a single observation
            # priors
            mu0 = self.mu[quant][t,ixs]
            n0 = self.n[quant][t,ixs]
            a = self.alpha[quant][t,ixs]
            b = self.beta[quant][t,ixs]
            # updating:
            self.mu[quant][t+1,ixs] = n/(n+n0)*x + n0/(n+n0)*mu0
            self.n[quant][t+1,ixs] = n0 + n
            self.alpha[quant][t+1,ixs] = a + n/2
            self.beta[quant][t+1,ixs] = b + 1/2 * (x-mu0)**2 * (1 + n*n0 / (2*(n+n0))) # note: this formula is simplified for a single observation
            self.var[quant][t+1,ixs] = self.beta[quant][t+1,ixs] / self.alpha[quant][t+1,ixs]
            # update beliefs for those with no observations
            objs = [self.mu, self.n, self.alpha, self.beta, self.var]
            for o in objs:
                o[quant][t+1,~ixs] = o[quant][t,~ixs]