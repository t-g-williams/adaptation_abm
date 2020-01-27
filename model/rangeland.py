import numpy as np
import code

class Rangeland():
    def __init__(self, farm_area_ha, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['rangeland']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # calculate size (in ha)
        self.size = self.range_farm_ratio * farm_area_ha

        # initialize arrays
        # note: this represents the state at the START of the time step
        self.G = np.full(self.T+1, -99) # store as integer (kg/ha)
        self.R = np.full(self.T+1, -99) # kg/ha
        self.F_avail = np.full(self.T, -99)
        self.R[0] = self.R0_frac * self.R_max
        self.G[0] = self.R[0] * self.G_R_ratio # assume it starts saturated

    def update(self, climate, agents):
        '''
        simulate the rangeland dynamics
        This model is copied from Gunnar
        '''
        t = self.t[0]

        ## 1. green biomass growth
        self.G[t+1] = (1-self.G_mortality) * self.G[t] + \
                self.rain[t] * self.rain_use_eff * self.R[t]
        # green biomass constraints
        self.G[t+1] = min(self.G[t+1], self.R[t] * self.G_R_ratio)
        G_no_consumption = copy.deepcopy(self.G[t+1])

        ## 2. livestock reproduction
        # note: we model livestock as integer valued
        # each animal has the same probability of reproducing
        herds = agents.livestock[t]
        births = np.random.binomial(herds, self.birth_rate)
        herds += births

        ## 3. livestock consumption and destocking
        destocking_total = self.consumption(herds, land)
        self.apportion_destocking(herds, destocking_total)

        ## 4. reserve biomass growth
        self.R[t+1] = (1-self.R_mortality) * self.R[t] + \
                self.R_biomass_growth * (gr1 * (G_no_consumption - self.G[t+1]) + self.G[t+1]) * \
                (1 - self.R[t]/self.R_max)

    def consumption(self, land, herds):
        '''
        simulate the consumption of on-farm crop residues
        as well as green and reserve biomass
        '''
        # livestock consumption is achieved via a mix of on-farm residues and the communal rangeland
        residue_production = self.crop_production[t] * land.residue_multiplier * land.residue_loss_factor # kg total
        herds_on_residue = np.minimum(herds, residue_production / self.consumption) # kg / (kg/head) = head
        # demand for the rangeland
        rangeland_demand_intensity = np.sum(herds - herds_on_residue) * self.consumption / self.size # unit = kg/ha
        # how is this demand satisfied...?
        if self.G[t+1] > rangeland_demand_intensity:
            # green biomass satisfies demand
            self.G[t+1] -= rangeland_demand_intensity
            destocking_total = 0 # no destocking rqd
        else:
            # need to use reserve biomass
            reserve_demand = rangeland_demand_intensity - self.G[t+1]
            reserve_limit = gr2 * self.R[t]
            self.G[t+1] = 0
            if reserve_demand < reserve_limit:
                # reserve can supply the demand
                self.R[t] -= reserve_demand
                destocking_total = 0 # no destocking rqd
            else:
                # still remaining shortfall
                self.R[t] -= reserve_limit
                deficit = reserve_demand - reserve_limit
                destocking_total = int(deficit * self.size / self.consumption) # kg/ha * ha / (kg/head) = head

        return destocking_total

    def apportion_destocking(self, herds, total):
        '''
        apportion destocking randomly between agents
        each livestock has an equal probability of being destocked
        '''
        if total > 0:
            tot_ls = np.sum(herds)
            destock_ix = np.random.choice(np.arange(tot_ls), size=total, replace=False) # indexes of livestock
            owner_ix = np.repeat(np.arange(herds.shape), herds)
            owner_destocks = owner_ix[destock_ix]
            for owner in owner_destocks: # would be great to get rid of this for-loop!!
                herds[owner] -= 1
