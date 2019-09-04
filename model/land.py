import numpy as np
import code
import copy

class Land():
    def __init__(self, agents, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['land']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # how many total plots?
        self.n_plots = sum(agents.n_plots)
        self.owner = np.repeat(agents.id, self.n_plots)

        ##### soil properties #####
        # represents the START of the year
        self.SOM = np.full([self.T+1, self.n_plots], np.nan)
        self.init_SOM()
        # inorganic represents the END of the year (i.e. for crop yield)
        self.inorganic = np.full([self.T, self.n_plots], np.nan)

        ##### crop yields #####
        self.yields = np.full([self.T, self.n_plots], np.nan)
        self.yields_unconstrained = np.full([self.T, self.n_plots], np.nan)
        self.nutrient_factors = np.full([self.T, self.n_plots], np.nan)
        self.rf_factors = np.full([self.T, self.n_plots], np.nan)

    def init_SOM(self):
        '''
        iniitalize the soil organic matter levels
        '''
        # sample from uniform distribution
        self.SOM[0] = np.random.uniform(self.SOM_min_init, self.SOM_max_init, self.n_plots)

    def update_soil(self, agents):
        '''
        simulate the evolution of the land throughout the year
        '''
        ### initialize -- assume inorganic is reset each year
        inorganic = np.full(self.n_plots, 0.)
        organic = copy.copy(self.SOM[self.t[0]])

        ### agent inputs
        organic += self.livestock_SOM_input(agents)

        ### inorganic losses: loss of inorganic is a linear function of SOM
        losses = inorganic * (self.loss_min + (1-organic) * (self.loss_max - self.loss_min))
        inorganic -= losses

        ### mineralization: assume a linear decay model
        mineralization = self.mineralization_rate * organic
        inorganic += mineralization
        organic -= mineralization

        ### save final values
        self.inorganic[self.t[0]] = inorganic
        self.SOM[self.t[0]+1] = organic

    def livestock_SOM_input(self, agents):
        '''
        use agents' wealth as a _proxy_ for livestock ownership
        assuming that livestock manure applied to field = f(wealth)
        '''
        # assume a linear conversion function
        # agents' wealth is split equally between their fields
        wealth_per_field = agents.wealth[self.t[0]] / agents.n_plots
        wealth_per_field = np.repeat(wealth_per_field, agents.n_plots) # change shape
        return wealth_per_field * self.wealth_SOM_conversion

    def crop_yields(self, agents, climate):
        '''
        calculate crop yields
        assume yield = (MAX * climate_reduction +/- error) * nutrient reduction
        '''
        t = self.t[0]
        # rainfall effect
        self.rf_factors[t] = self.calculate_rainfall_factor(climate.rain[t])
        # random effect
        errors = np.random.normal(1, self.random_effect_sd, self.n_plots)
        errors[errors < 0] = 0
        # nutrient unconstrained yield
        self.yields_unconstrained[t] = self.max_yield * self.rf_factors[t] * errors
        # factor in nutrient contraints
        max_with_nutrients = self.inorganic[t] / self.nitrogen_fraction
        self.yields[t] = np.minimum(self.yields_unconstrained[t], max_with_nutrients)
        self.nutrient_factors[t] = self.yields[t] / self.yields_unconstrained[t]
        # attribute to agents.
        agents.crop_production[t] = self.land_to_agent(self.yields[t], agents.n_plots, mode='sum')

    def calculate_rainfall_factor(self, rain):
        '''
        convert the rainfall value (in 0,1) to a yield reduction factor
        '''
        if rain > self.rain_crit:
            return np.full(self.n_plots, 1) # no effect
        else:
            # organic matter reduces rainfall sensitivity
            # first, calculate value with perfect SOM
            a = self.rain_cropfail_high_SOM
            b = self.rain_cropfail_low_SOM
            c = self.rain_crit
            eff_max = (a-rain) / (a-c)
            # now, if SOM=0, how much is it reduced?
            # this is a function of the difference in the slopes of the two lines
            red_max = (c - rain) * (1/(c-b) - 1/(c-a))
            # now factor in the fields' actual SOM values
            rf_effects = eff_max - (1-self.SOM[self.t[0]]) * red_max
            return rf_effects

    def land_to_agent(self, vals, num_fields, mode='sum'):
        '''
        convert a land-level property to an agent-level property.
        vectorized
        '''
        cumsums = np.concatenate(([0], np.cumsum(vals)))
        ends = np.cumsum(num_fields) # this is the index that corresponds to the final element of each agent's sum
        starts = ends - num_fields # this is the index of the point BEFORE each agent's values starts
        ag_sums = (cumsums[ends] - cumsums[starts])

        if mode == 'sum':
            return ag_sums
        elif mode == 'average':
            return ag_sums / num_fields        