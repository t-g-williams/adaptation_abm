'''
model class
'''
import numpy as np
from .agents import Agents
from .land import Land
from .climate import Climate
import code

class Model():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['model']
        for key, val in self.inputs.items():
            setattr(self, key, val)

        np.random.seed(self.seed)

        # initialize the sub-objects
        self.agents = Agents(inputs)
        self.land = Land(self.agents, inputs)
        self.climate = Climate(inputs)

        # set the time
        # save as list so it is mutable (stays same over all objects)
        self.t = self.agents.t = self.land.t = self.climate.t = [0] 
        
        # initialize adaptation options
        self.init_adaptation_options()

    def step(self):
        '''
        advance the simulation by one year
        '''
        self.land.update_soil(self.agents)
        self.land.crop_yields(self.agents, self.climate)
        self.agents.calculate_income(self.land, self.climate, self.adap_properties)
        self.agents.coping_measures()
        self.agents.adaptation(self.adap_properties)
        # increment the year
        self.t[0] += 1

    def init_adaptation_options(self):
        '''
        calculate any required parameters for the selected adaptation option(s)
        '''
        if self.adaptation_option == 'insurance':
            props = self.all_inputs['adaptation']['insurance']
            ## calculate the insurance cost and payout
            ## make it fair in expectation
            ## NOTE : this doesn't incorporate soil quality reductions! so it's not fair if you don't have good soil quality
            # calculate expected crop yield
            rains = np.random.normal(self.climate.rain_mu, self.climate.rain_sd, 1000)
            rain_facs = np.full(rains.shape, np.nan)
            for r, rain in enumerate(rains):
                rain_facs[r] = self.land.calculate_rainfall_factor(rain, virtual=True)
            exp_yield = np.mean(rain_facs) * self.land.max_yield
            exp_crop_income = exp_yield * self.agents.crop_sell_price # birr/ha
            payout = exp_crop_income * props['payout_magnitude'] # birr/ha
            cost = payout * props['climate_percentile'] # birr/ha
            magnitude = np.percentile(rains, props['climate_percentile']*100)
            # save these
            self.adap_properties = {
                'type' : 'insurance',
                'cost' : cost,
                'payout' : payout,
                'magnitude' : magnitude
            }
        else:
            self.adap_properties = {
                'type' : 'None'
            }
