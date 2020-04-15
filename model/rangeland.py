import numpy as np
import code
import copy
from collections import Counter

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
        val = -99 if self.integer_consumption else np.nan
        self.demand_intensity = np.full(self.T, val)
        self.livestock_supported = np.full(self.T, val)

    def update(self, climate, agents, land):
        '''
        simulate the rangeland dynamics
        This model is copied from Gunnar
        '''
        t = self.t[0]

        ## 1. livestock reproduction
        # note: we model livestock as integer valued
        # each animal has the same probability of reproducing
        # control stochasticity b/c we have variable number of livestock
        herds = agents.ls_obj
        birth_rate = self.all_inputs['livestock']['birth_rate']
        if birth_rate > 0:
            rndm_num = np.random.randint(1e6)
            births = np.random.binomial(herds, birth_rate)
            np.random.seed(rndm_num)
            herds += births
        agents.ls_reprod[t] = copy.deepcopy(herds)

        if not self.rangeland_dynamics:
            return

        ## 2. green biomass growth
        self.G[t+1] = (1-self.G_mortality) * self.G[t] + \
                climate.rain[t] * self.rain_use_eff * self.R[t]
        # green biomass constraints
        self.G[t+1] = min(self.G[t+1], self.R[t] * self.G_R_ratio)
        self.G_no_cons[t] = copy.deepcopy(self.G[t+1])

        ## 3. livestock consumption and destocking
        destocking_total = self.consumption(agents, land, herds, t)
        self.apportion_destocking(destocking_total, agents.herds_on_rangeland[t])
        # add together the livestock on residue and rangeland
        # if there's been a "partial destocking" required on the rangeland, the whole animal must be destocked
        # hence, take the floor of this sum
        agents.ls_obj = np.floor(agents.herds_on_residue[t] + agents.herds_on_rangeland[t]).astype(int)
        agents.ls_rangeland[t] = copy.deepcopy(agents.ls_obj)

        ## 4. reserve biomass growth
        self.R[t+1] = (1-self.R_mortality) * self.R[t] + \
                self.R_biomass_growth * (self.gr1 * (self.G_no_cons[t] - self.G[t+1]) + self.G[t+1]) * \
                (1 - self.R[t]/self.R_max)

        if np.sum(agents.ls_obj<0)>0:
            print('ERROR: negative livestock in rangeland.update()')
            code.interact(local=dict(globals(), **locals())) 

    def consumption(self, agents, land, herds, t):
        '''
        simulate the consumption of on-farm crop residues
        as well as green and reserve biomass
        '''
        ls_consumption = self.all_inputs['livestock']['consumption']
        # livestock consumption is achieved via a mix of on-farm residues and the communal rangeland
        agents.herds_on_residue[t] = np.minimum(herds, land.residue_production[t] / ls_consumption) # kg / (kg/head) = head
        if self.integer_consumption:
            agents.herds_on_residue[t] = np.floor(agents.herds_on_residue[t]).astype(int)
        # agents.herds_on_residue[t] = np.minimum(herds, land.residue_production[t] / self.all_inputs['livestock']['consumption'])
        # ^ take the floor of this. keep as integer
        # demand for the rangeland
        agents.herds_on_rangeland[t] = herds - agents.herds_on_residue[t]

        if self.size_ha == 0:
            # clause if there is NO rangeland left (due to LSLA)
            self.demand_intensity[t] = np.inf
            self.G[t+1] = 0
            self.R[t] = 0
            destocking_total = int(np.ceil(agents.herds_on_rangeland[t].sum()))
        else:
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
                    destocking_total = int(np.ceil(deficit * self.size_ha / ls_consumption)) # kg/ha * ha / (kg/head) = head (must be integer)

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
        NOTE: variable number of calls to np.random so control stochasticity
        NOTE2: agents receive no income for this destocking !!
        '''
        rand_int = np.random.randint(1e6)
        if total > 0:
            tot_ls = np.sum(range_herds)
            total = np.ceil(min(total, tot_ls)).astype(int) # ceiling -- partial livestock can be destocked also
            
            if self.integer_consumption:
                destock_ix = np.random.choice(np.arange(tot_ls), size=total, replace=False) # indexes of livestock
                owner_ix = np.repeat(np.arange(range_herds.shape[0]), range_herds)
                owner_destocks = owner_ix[destock_ix]
                destock_counts = np.array(list(Counter(owner_destocks).items()))
            else:
                # create a list of all of the livestock (even partial livestock)
                # partial livestock have a lower probability of being selected for destocking
                rounded = np.ceil(range_herds).astype(int)
                owner_ix = np.repeat(np.arange(rounded.shape[0]), rounded)
                partial_ls_amt = range_herds - np.floor(range_herds)
                partial_ixs = np.cumsum(rounded[rounded>0]) - 1 # only consider the agents that have livestock
                probs = np.full(owner_ix.shape, 1.) # probability of being selected for destocking
                # the partial ixs includes those with _whole_ numbers of livestock (e.g., 5.)
                # we want to keep these ixs as a probability of 1, so filter also on partial_ls_amt>0
                probs[partial_ixs[partial_ls_amt[rounded>0]>0]] = partial_ls_amt[(rounded>0) * (partial_ls_amt>0)]
                # select the owner ids of the livestock that need to be destocked
                destock_ixs = np.random.choice(owner_ix.shape[0], size=total, replace=False, p=probs/probs.sum())
                owner_destocks = owner_ix[destock_ixs]
                destock_amts = probs[destock_ixs] # some of the destocks can be partial
                c = Counter()
                for o, ownr_id in enumerate(owner_destocks):
                    c.update({ownr_id:destock_amts[o]})
                destock_counts = np.array(list(c.items()))

            range_herds[destock_counts[:,0].astype(int)] -= destock_counts[:,1]

        np.random.seed(rand_int)