import numpy as np
import scipy.stats

class Climate():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['climate']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # create the entire sequence of climate realizations
        self.rain = np.random.normal(self.rain_mu, self.rain_sd, self.T)
        self.rain[self.rain < 0] = 0
        self.rain[self.rain > 1] = 1

        # add in shock if necessary
        if inputs['model']['shock']:
            if self.shock_as_percentile:
                # find the absolute rain value corresponding to this percentile
                val = scipy.stats.norm.ppf(self.shock_rain, self.rain_mu, self.rain_sd)
                val = max(min(val,1),0) # bound to [0,1]
                self.rain[self.shock_years] = val
            else:
                # the shock magnitude is given as an absolute value
                self.rain[self.shock_years] = self.shock_rain