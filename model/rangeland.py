import numpy as np
import code
import copy

class Rangeland():
    def __init__(self, farm_area_ha, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['rangeland']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # calculate size (in ha)
        self.size_ha = self.range_farm_ratio * farm_area_ha

        # initialize arrays
        # note: this represents the state at the START of the time step
        self.G = np.full(self.T+1, -99) # store as integer (kg/ha)
        self.R = np.full(self.T+1, -99) # kg/ha
        self.F_avail = np.full(self.T, -99)
        self.R[0] = self.R0_frac * self.R_max
        self.G[0] = self.R[0] * self.G_R_ratio # assume it starts saturated
        self.G_no_cons = np.full(self.T, -99) # store as integer (kg/ha)
        self.destocking_rqd = np.full(self.T, False)
        self.demand_intensity = np.full(self.T, np.nan)
        self.livestock_supported = np.full(self.T, -99)

    def update(self, climate, agents, land):
        '''
        simulate the rangeland dynamics
        This model is copied from Gunnar
        '''
        if not self.rangeland_dynamics:
            return

        t = self.t[0]

        ## 1. green biomass growth
        self.G[t+1] = (1-self.G_mortality) * self.G[t] + \
                climate.rain[t] * self.rain_use_eff * self.R[t]
        # green biomass constraints
        self.G[t+1] = min(self.G[t+1], self.R[t] * self.G_R_ratio)
        self.G_no_cons[t] = copy.deepcopy(self.G[t+1])

        ## 2. livestock reproduction
        # note: we model livestock as integer valued
        # each animal has the same probability of reproducing
        herds = agents.livestock[t]
        births = np.random.binomial(herds, self.all_inputs['livestock']['birth_rate'])
        herds += births
        agents.ls_reprod[t] = copy.deepcopy(herds)

        ## 3. livestock consumption and destocking
        destocking_total = self.consumption(agents, land, herds, t)
        self.apportion_destocking(destocking_total, agents.herds_on_rangeland[t])
        agents.livestock[t] = agents.herds_on_residue[t] + agents.herds_on_rangeland[t]
        agents.ls_destock[t] = copy.deepcopy(agents.livestock[t])

        ## 4. reserve biomass growth
        self.R[t+1] = (1-self.R_mortality) * self.R[t] + \
                self.R_biomass_growth * (self.gr1 * (self.G_no_cons[t] - self.G[t+1]) + self.G[t+1]) * \
                (1 - self.R[t]/self.R_max)

        if np.sum(agents.livestock[t]<0)>0:
            print('ERROR: negative livestock in rangeland.update()')
            code.interact(local=dict(globals(), **locals())) 

    def consumption(self, agents, land, herds, t):
        '''
        simulate the consumption of on-farm crop residues
        as well as green and reserve biomass
        '''
        # livestock consumption is achieved via a mix of on-farm residues and the communal rangeland
        agents.herds_on_residue[t] = np.floor(np.minimum(herds, land.residue_production / self.all_inputs['livestock']['consumption'])).astype(int) # kg / (kg/head) = head
        # agents.herds_on_residue[t] = np.minimum(herds, land.residue_production / self.all_inputs['livestock']['consumption'])
        # ^ take the floor of this. keep as integer
        # demand for the rangeland
        ls_consumption = self.all_inputs['livestock']['consumption']
        agents.herds_on_rangeland[t] = herds - agents.herds_on_residue[t]
        self.demand_intensity[t] = np.sum(agents.herds_on_rangeland[t]) * ls_consumption / self.size_ha # unit = kg/ha
        # how is this demand satisfied...?
        if self.G[t+1] > self.demand_intensity[t]:
            # green biomass satisfies demand
            self.G[t+1] -= self.demand_intensity[t]
            destocking_total = 0 # no destocking rqd
        else:
            # need to use reserve biomass
            reserve_demand = self.demand_intensity[t] - self.G[t+1]
            reserve_limit = self.gr2 * self.R[t]
            self.G[t+1] = 0
            if reserve_demand < reserve_limit:
                # reserve can supply the demand
                self.R[t] -= reserve_demand
                destocking_total = 0 # no destocking rqd
            else:
                # still remaining shortfall
                self.R[t] -= reserve_limit
                deficit = reserve_demand - reserve_limit
                destocking_total = int(np.ceil(deficit * self.size_ha / ls_consumption)) # kg/ha * ha / (kg/head) = head

        self.destocking_rqd[t] = True if np.sum(destocking_total)>0 else False
        self.livestock_supported[t] = np.sum(agents.herds_on_rangeland[t])-destocking_total

        if np.sum(agents.herds_on_residue[t]<0)>0:
            print('ERROR: negative herds on residue')
            code.interact(local=dict(globals(), **locals())) 

        # code.interact(local=dict(globals(), **locals()))  
        return destocking_total

    def apportion_destocking(self, total, range_herds):
        '''
        apportion destocking randomly between agents
        each livestock has an equal probability of being destocked
        '''
        if total > 0:
            tot_ls = np.sum(range_herds)
            total = min(total, tot_ls)
            destock_ix = np.random.choice(np.arange(tot_ls), size=total, replace=False) # indexes of livestock
            owner_ix = np.repeat(np.arange(range_herds.shape[0]), range_herds)
            owner_destocks = owner_ix[destock_ix]
            for owner in owner_destocks: # would be great to get rid of this for-loop!!
                range_herds[owner] -= 1