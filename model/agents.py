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
        # money
        self.income = np.full([self.T+1, self.N], np.nan)
        self.cash_req = np.random.normal(self.cash_req_mean, self.cash_req_sd, self.N)
        # coping measures
        self.coping_rqd = np.full([self.T, self.N], False)
        self.cant_cope = np.full([self.T, self.N], False)
        # adaptation option decisions
        self.adapt = np.full([self.T+1, self.N], False)

    def init_farm_size(self):
        '''
        initialize agent-level farm size (number of pixels)
        '''
        if self.land_heterogeneity:
            # use poisson distribution
            return np.random.poisson(self.land_mean, self.N)
        else:
            # all the same size
            return np.full(self.N, self.land_mean).astype(int)

    def calculate_income(self, land, climate, adap_properties):
        '''
        calculate end-of-year income
        '''
        t = self.t[0]
        # costs and payouts for adaptation option
        adap_costs = np.full(self.N, 0.)
        payouts = np.full(self.N, 0.)
        if adap_properties['type'] == 'insurance':
            # costs
            adap_costs[self.adapt[t]] = adap_properties['cost'] * land.area * self.n_plots[self.adapt[t]]
            # payouts
            if climate.rain[t] < adap_properties['magnitude']:
                payouts[self.adapt[t]] = adap_properties['payout'] * self.n_plots[self.adapt[t]] * land.area
        elif adap_properties['type'] == 'cover_crop':
            adap_costs[self.adapt[t]] = adap_properties['cost'] * land.area * self.n_plots[self.adapt[t]]

        # income = crop_sales + payouts - cash_req - adap_costs
        self.income[t] = self.crop_sell_price*self.crop_production[t] + payouts - self.cash_req - adap_costs

    def coping_measures(self):
        '''
        calculate end-of-year income balance
        and simulate coping measures
        '''
        t = self.t[0]
        # assume those with -ve income are required to engage in coping measure
        self.coping_rqd[t, self.income[t] < 0] = True
        # add (or subtract) agent income to their wealth
        # this proxies the effect of buying (+ve income) or selling (-ve income) livestock
        self.wealth[t+1] = self.wealth[t] + self.income[t]
        # record agents with -ve wealth (not able to cope)
        self.cant_cope[t, self.wealth[t+1] < 0] = True

    def adaptation(self, land, adap_properties):
        '''
        simulate adaption decision-making
        '''
        if adap_properties['adap']:
            t = self.t[0]
            if self.adap_type == 'coping':
                # agents engage in the adaptation option next period
                # if they had to cope this period
                self.adapt[t+1, self.coping_rqd[t]] = True
            elif self.adap_type == 'switching':
                # agents SWITCH adaptation types if they had to cope in this period
                self.adapt[t+1, ~self.coping_rqd[t]] = self.adapt[t, ~self.coping_rqd[t]]
                self.adapt[t+1, self.coping_rqd[t]] = ~self.adapt[t, self.coping_rqd[t]]
            elif self.adap_type == 'affording':
                # agents adapt if they can afford it
                afford = self.wealth[t+1] >= (adap_properties['cost'] * land.area * self.n_plots)
                self.adapt[t+1, afford] = True