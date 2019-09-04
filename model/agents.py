import numpy as np
import code

class Agents():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['agents']
        for key, val in self.inputs.items():
            setattr(self, key, val)

        self.N = self.all_inputs['model']['n_agents']
        self.T = self.all_inputs['model']['T']
        self.id = np.arange(self.N)

        # generate land ownership
        self.n_plots = self.init_farm_size()
        self.crop_production = np.full([self.T, self.N], np.nan)

        # wealth (cash holdings)
        # this represents the START of the year
        self.wealth = np.full([self.T+1, self.N], np.nan)
        self.wealth[0] = np.random.normal(self.wealth_init_mean, self.wealth_init_sd, self.N)
        self.wealth[0][self.wealth[0]<0] = 0 # fix any -ve values
        # cash requirements
        self.cash_req = np.random.normal(self.cash_req_mean, self.cash_req_sd, self.N)
        # coping measures
        self.coping_rqd = np.full([self.T, self.N], False)
        self.cant_cope = np.full([self.T, self.N], False)

    def init_farm_size(self):
        '''
        initialize agent-level farm size (number of pixels)
        '''
        if self.land_heterogeneity:
            # use poisson distribution
            return np.random.poisson(self.land_mean, self.N)
        else:
            # all the same size
            return np.full(self.land_mean, self.N)

    def coping_measures(self):
        '''
        calculate end-of-year income balance
        and simulate coping measures
        '''
        t = self.t[0]
        
        # income = crop_sales - cash_req
        income = self.crop_sell_price*self.crop_production[t] - self.cash_req

        ## coping measures
        # assume those with -ve income were required to engage in coping measure
        self.coping_rqd[t, income < 0] = True
        # add (or subtract) agent income to their wealth
        # this proxies the effect of buying (+ve income) or selling (-ve income) livestock
        self.wealth[t+1] = self.wealth[t] + income
        # record agents with -ve wealth (not able to cope)
        self.cant_cope[t, self.wealth[t+1] < 0] = True