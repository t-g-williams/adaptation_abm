'''
model class
'''
import numpy as np
from .agents import Agents
from .land import Land
from .climate import Climate

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
        self.t = self.agents.t = self.land.t = [0] 

    def step(self):
        '''
        advance the simulation by one year
        '''
        self.land.update_soil(self.agents)
        self.land.crop_yields(self.agents, self.climate)
        self.agents.coping_measures()
        # increment the year
        self.t[0] += 1