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
        self.n_plots = agents.N
        self.owner = agents.id

        ##### soil properties #####
        # represents the START of the year
        self.organic = np.full([self.T+1, self.n_plots], np.nan)
        self.init_organic()
        # inorganic represents the END of the year (i.e. for crop yield)
        self.inorganic = np.full([self.T, self.n_plots], np.nan)

        ##### crop yields #####
        self.yields = np.full([self.T, self.n_plots], -9999) # kg
        self.yields_unconstrained = np.full([self.T, self.n_plots], -9999)# kg
        self.nutrient_factors = np.full([self.T, self.n_plots], np.nan)
        self.rf_factors = np.full([self.T, self.n_plots], np.nan)

        # other
        self.cover_crop_N_fixed = np.full([self.T, self.n_plots], np.nan)

    def init_organic(self):
        '''
        iniitalize the soil organic matter levels
        kgN / ha
        '''
        # sample from uniform distribution
        self.organic_N_max_init = self.organic_N_min_init
        self.organic[0] = np.random.uniform(self.organic_N_min_init, self.organic_N_max_init, self.n_plots)

    def update_soil(self, agents, adap_properties, climate):
        '''
        simulate the evolution of the land throughout the year
        '''
        ### initialize -- assume inorganic is reset each year
        inorganic = np.full(self.n_plots, 0.)
        organic = copy.copy(self.organic[self.t[0]])

        ### mineralization: assume a linear decay model
        # assume the stocks from last year mineralize straight away
        mineralization = self.slow_mineralization_rate * organic
        inorganic += mineralization
        organic -= mineralization

        ### agent inputs
        # inorganic += self.apply_fixed_fertilizer(agents) # kgN/ha ## NOT IN MODEL YET
        residue = self.crop_residue_input()
        livestock = self.livestock_SOM_input(agents) # kgN/ha
        cover_crop = self.cover_crop_input(agents, adap_properties, climate) # kgN/ha
        # these additions are split between organic and inorganic matter
        inorganic += self.fast_mineralization_rate * (residue + livestock + cover_crop)
        organic += (1-self.fast_mineralization_rate) * (residue + livestock + cover_crop)

        ### constrain to be within bounds
        organic[organic < 0] = 0
        organic[organic > self.max_organic_N] = self.max_organic_N

        ### inorganic losses: loss of inorganic is a linear function of SOM
        losses = inorganic * (self.loss_min + (self.max_organic_N-organic)/self.max_organic_N * (self.loss_max - self.loss_min))
        inorganic -= losses
        inorganic[inorganic < 0] = 0 # constrain

        ### save final values
        self.inorganic[self.t[0]] = inorganic # end of this year (for yields)
        self.organic[self.t[0]+1] = organic # start of next year

    def crop_residue_input(self):
        '''
        apply crop residues from the previous year to fields
        assume there's a conversion factor from the crop yields
        and convert back to "nitrogen"
        '''
        if self.t[0] > 0:
            return self.yields[self.t[0]-1] * self.residue_loss_factor * self.residue_multiplier / self.residue_CN_conversion # kgN/ha = kg crop/ha * __ * kgN/kgC 
        else:
            return np.full(self.n_plots, 0.)

    def livestock_SOM_input(self, agents):
        '''
        use agents' wealth as a _proxy_ for livestock ownership
        assuming that the amount of additional SOM available 
        (above the crop residues they have consumed)
        is given by the fraction of their consumption that is from crop residue
        '''
        # agents' wealth is split equally over their land. birr / ha
        wealth_per_ha = agents.wealth[self.t[0]] / agents.land_area
        wealth_per_ha = np.maximum(wealth_per_ha, 0) # assume ppl in debt have no livestock
        N_per_ha = wealth_per_ha * self.wealth_N_conversion * (1-self.livestock_frac_crops) # birr/ha * kgN/birr * __ = kgN/ha
        return N_per_ha

    def cover_crop_input(self, agents, adap_properties, climate):
        '''
        calculate the input from legume cover crops
        assume a linear model between the specified minimum and maximum amounts based on SOM content
        
        if climate dependence of cover crop benefits:
            use the same rainfall scaling factor as for crop yields
        '''
        inputs = np.full(self.n_plots, 0.)
        if adap_properties['type'] in ['cover_crop','both']:
            adap = agents.adapt[self.t[0]]
            fields = np.in1d(self.owner, agents.id[adap]) # identify the fields
            inputs_som_scaling = adap_properties['N_fixation_min'] + \
                (1-self.organic[self.t[0],fields] / self.max_organic_N) * (adap_properties['N_fixation_max']-adap_properties['N_fixation_min']) # kg/ha
            rf_factors = self.calculate_rainfall_factor(climate.rain[self.t[0]], cover_crop=True)[fields] if adap_properties['climate_dependence'] else 1
            self.cover_crop_N_fixed[self.t[0],fields] = (inputs_som_scaling * rf_factors)
            inputs[fields] += self.cover_crop_N_fixed[self.t[0],fields]

        return inputs

    def crop_yields(self, agents, climate):
        '''
        calculate crop yields
        assume yield = (MAX_VAL * climate_reduction +/- error) * nutrient_reduction
        '''
        t = self.t[0]
        # rainfall effect
        self.rf_factors[t] = self.calculate_rainfall_factor(climate.rain[t])
        # random effect
        errors = np.random.normal(1, self.random_effect_sd, self.n_plots)
        errors[errors < 0] = 0
        # nutrient unconstrained yield
        self.yields_unconstrained[t] = self.max_yield * self.rf_factors[t] # kg/ha
        # factor in nutrient contraints
        max_with_nutrients = self.inorganic[t] / (1/self.crop_CN_conversion+self.residue_multiplier/self.residue_CN_conversion) # kgN/ha / (kgN/kgC_yield) = kgC/ha ~= yield(perha
        self.yields[t] = np.minimum(self.yields_unconstrained[t], max_with_nutrients) * errors # kg/ha
        with np.errstate(invalid='ignore'):
            self.nutrient_factors[t] = self.yields[t] / self.yields_unconstrained[t]
            self.nutrient_factors[t] = np.minimum(self.nutrient_factors[t], 1)

        # attribute to agents
        agents.crop_production[t] = self.yields[t] * agents.land_area # kg

    def calculate_rainfall_factor(self, rain, virtual=False, cover_crop=False):
        '''
        convert the rainfall value (in 0,1) to a yield reduction factor
        '''
        if rain > self.rain_crit:
            if virtual:
                return 1
            else:
                return np.full(self.n_plots, 1) # no effect
        else:
            # organic matter reduces rainfall sensitivity
            # first, calculate value with maximum organic N
            a = self.rain_cropfail_high_SOM
            b = self.rain_cropfail_low_SOM
            c = self.rain_crit
            eff_max = (a-rain) / (a-c)
            # if this is a "virtual" calculation we don't account for the SOM
            if virtual:
                return max(eff_max, 0)
            # now, if SOM=0, how much is it reduced?
            # this is a function of the difference in the slopes of the two lines
            red_max = (c - rain) * (1/(c-b) - 1/(c-a))
            # now factor in the fields' actual SOM values
            if cover_crop:
                # assume the start-of-year value
                org = self.organic[self.t[0]]
            else:
                # assume the average of the start and end of the year
                org = np.mean(self.organic[[self.t[0], self.t[0]+1]], axis=0)
            
            rf_effects = eff_max - (1 - org/self.max_organic_N) * red_max
            return np.maximum(rf_effects, 0)

    def apply_fixed_fertilizer(self, agents):
        '''
        simulate application of a fixed amount of fertilizer to fields
        only for agents that are using this option
        '''
        fert_applied = np.full(self.n_plots, 0.)
        ag = agents.adapt['fertilizer_fixed'][agents.t[0]]
        if np.sum(ag) > 0:
            # add to the fields
            fields = np.in1d(self.owner, agents.id[ag]) # identify the fields
            amt = self.all_inputs['adaptation']['fertilizer_fixed']['application_rate']
            fert_applied[fields] = amt
            # add costs to agents
            agents.fert_costs[agents.t[0], ag] += amt * agents.fertilizer_cost * agents.land_area[ag]
        
        return fert_applied    