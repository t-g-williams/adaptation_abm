import numpy as np
import scipy.stats as stat
import code
import copy
import sys

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
        self.land_area = self.init_farm_size()      
        self.crop_production = np.full([self.T, self.N], -9999)
        # household size
        self.hh_size = np.full(self.N, self.hh_size_init) # all same size for now

        # define agent types
        self.init_types()

        # savings and livestock
        # this represents the START of the year
        self.savings = np.full([self.T+1, self.N], -9999)
        self.savings[0] = np.random.normal(self.savings_init_mean, self.savings_init_sd, self.N)
        self.savings[0][self.savings[0]<0] = 0 # fix any -ve values
        self.livestock = np.full([self.T+1, self.N], -9999)
        self.livestock[0] = np.floor(self.livestock_init).astype(int) # constant amount for each agent (same as savings)
        self.wealth = np.full([self.T+1, self.N], -9999) # sum of livestock + savings
        self.wealth[0] = self.savings[0] + self.livestock[0]
        # money
        self.income = np.full([self.T, self.N], -9999)
        self.living_cost =  self.living_cost_pp * self.hh_size
        # coping measures
        self.neg_income = np.full([self.T, self.N], False)
        self.destocking_rqd = np.full([self.T, self.N], False)
        self.stress_ls_sell_rqd = np.full([self.T, self.N], False)
        self.cant_cope = np.full([self.T, self.N], False)
        # adaptation option decisions
        self.adapt = np.full([self.T+1, self.N], False)
        # agricultural decisions
        self.fallow = np.full([self.T+1, self.N], True)
        # other livestock values for record keeping
        self.ls_start = np.full([self.T, self.N], -9999)
        self.ls_reprod = np.full([self.T, self.N], -9999)
        self.ls_destock = np.full([self.T, self.N], -9999)
        self.ls_stress = np.full([self.T, self.N], -9999)
        self.ls_purchase = np.full([self.T, self.N], -9999)
        self.herds_on_rangeland = np.full([self.T, self.N], -9999)

    def init_farm_size(self):
        '''
        initialize agent-level farm size (ha)
        '''
        mult = self.land_area_multiplier
        if self.N == len(self.land_area_init):
            return np.array(self.land_area_init) * mult
        elif self.N % len(self.land_area_init) == 0:
            # equal number of each
            return np.repeat(self.land_area_init, self.N / len(self.land_area_init)) * mult
        else:
            return np.random.choice(self.land_area_init, size=self.N) * mult
        

    def init_types(self):
        '''
        assign each agent to a "type" based on its characteristics
        '''
        if self.types == False:
            self.type = np.repeat(0, self.N)
            return

        self.type = np.full(self.N, np.nan, dtype='object')
        for name, rqmt in self.types.items():
            qualifies = np.full(self.N, True)
            for ki, vi in rqmt.items():
                qualifies *= eval('self.{}'.format(ki)) == vi
            self.type[qualifies] = name

    def calculate_income(self, land, climate, adap_properties):
        '''
        calculate end-of-year income
        '''
        t = self.t[0]
        # costs and payouts for adaptation option
        adap_costs = np.full(self.N, 0.)
        self.insurance_payout_year = False
        if adap_properties['type'] == 'insurance':
            # costs
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]
            # payouts
            if climate.rain[t] < adap_properties['magnitude']:
                payouts = np.full(self.N, 0.)
                payouts[self.adapt[t]] = adap_properties['payout'] * self.land_area[self.adapt[t]]
                self.insurance_payout_year = True
        elif adap_properties['type'] == 'cover_crop':
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]

        ## livestock milk production
        ls_milk_money = self.livestock[t] * self.all_inputs['livestock']['income']

        # income = crop_sales - living_cost - adap_costs + livestock_milk
        self.income[t] = self.crop_sell_price*self.crop_production[t] - self.living_cost - adap_costs + ls_milk_money

        if self.insurance_payout_year:
            # assume that agents first use their payout to neutralize their income
            # and any left over, they use to buy fodder
            # which will increase their maximum wealth capacity
            self.remaining_payout = np.minimum(np.maximum(payouts+self.income[t], 0), payouts) # outer "minimum" is in case their income is +ve --> they can only use the payout for fodder
            self.income[t] += payouts.astype(int)

    def coping_measures(self, land, rangeland):
        '''
        calculate end-of-year income balance
        and simulate coping measures
        '''
        ## 0. calculate livestock limits
        ls_inp = self.all_inputs['livestock']
        t = self.t[0]
        ls_obj = copy.deepcopy(self.livestock[t])

        ## 1. ADD INCOME TO SAVINGS
        # this proxies using savings to counteract -ve income
        self.savings[t+1] = self.savings[t] + self.income[t]
        neg = self.income[t] < 0
        self.neg_income[t, neg] = True # record those with -ve income

        ## 2. CONSUMPTION SMOOTHING
        # if agents are in negative wealth we assume they can smooth their consumption
        # up to some extent
        cons_red_max = (self.living_cost[neg] * self.living_cost_reduction_max).astype(int)
        self.savings[t+1, neg] += np.minimum(-self.income[t,neg], cons_red_max)

        ## 3. STRESS DESTOCKING
        sell_rqmt = np.maximum(np.ceil(-self.savings[t+1]/ls_inp['cost']), 0).astype(int) # calculate amt rqd
        sell_amt = np.minimum(ls_obj, sell_rqmt) # restricted by available livestock
        ls_obj -= sell_amt # reduce the herdsize
        self.savings[t+1] += sell_amt * ls_inp['cost'] # add to income
        self.stress_ls_sell_rqd[t, sell_rqmt>0] = True # record
        self.ls_stress[t] = copy.deepcopy(ls_obj)

        ## 4. reset income for those that have zero still
        # assume that debts can't carry over
        self.cant_cope[t, self.savings[t+1]<0] = True # record
        self.savings[t+1, self.savings[t+1]<0] = 0

        # code.interact(local=dict(globals(), **locals()))
        return ls_obj

    def livestock_stocking(self, land, ls_obj, rangeland):
        '''
        calculate stocking/destocking
        stocking 
        - if the agent has extra money
        - WITH RANGELAND: if no destocking was rqd at regional level
        destocking
        - WITH RANGELAND: already calculated
        - WITHOUT RANGELAND ONLY: if not enough fodder total
        '''
        t = self.t[0]
        ls_inp = self.all_inputs['livestock']
        # how many can be purchased
        max_purchase = np.floor(self.savings[t+1] / ls_inp['cost'])
        # how many can be grazed on-farm
        # average the production over the past n years
        yrs = np.arange(0,t+1) if t<self.n_yr_smooth else np.arange(t-self.n_yr_smooth+1,t+1)
        max_on_farm = np.mean(self.crop_production[yrs], axis=0) * land.residue_multiplier * land.residue_loss_factor / \
                (ls_inp['consumption']) # TLU = kgCrop * kgDM/kgCrop / kgDM/TLU
        # how many can be grazed off-farm
        if rangeland.rangeland_dynamics:
            if t == 0:
                max_off_farm = 0
            else:
                # assume that agents do not increase livestock on rangeland
                # if there was destocking rqd at the regional level
                # OPTION 1 otherwise, they look back at the average amount they've had over the past few years
                # this leads to high rangeland quality long-run
                # max_off_farm = self.herds_on_rangeland[t] if rangeland.destocking_rqd[t] else np.floor(np.mean(self.herds_on_rangeland[yrs], axis=0)).astype(int)
                # OPTION 2 otherwise, they look at who had the most in the previous year (->tragedy of the commons?)
                # this leads to degradation long-run
                # max_off_farm = self.herds_on_rangeland[t] if rangeland.destocking_rqd[t] else max(self.herds_on_rangeland[t])
                # OPTION 3: look at the average value of other agents
                # this leads to high long-term rangeland quality
                # max_off_farm = self.herds_on_rangeland[t] if rangeland.destocking_rqd[t] else np.floor(np.mean(self.herds_on_rangeland[t])).astype(int)
                # OPTION 4: look back at the max amount they've had over the past few years
                # this leads to low rangeland quality long-run
                max_off_farm = self.herds_on_rangeland[t] if rangeland.destocking_rqd[t] else np.floor(np.max(self.herds_on_rangeland[yrs], axis=0)).astype(int)
                # # code.interact(local=dict(globals(), **locals()))
        else:
            # if 80% of livestock must be grazed on fodder, then the maximum wealth you can carry
            # is 20% of your current livestock herds + whatever you can sustain from your crop residues
            # i.e. it's assumed that some fraction of your livestock are fully independent of crop residue
            # rather than all livestock requiring this fraction of feed from fodder
            max_off_farm = (1-ls_inp['frac_crops']) * ls_obj

        # calculate the required change in livestock
        # if this is positive, fodder availability and cash allow for livestock purchase
        # if this is negative (ONLY POSSIBLE W/O RANGELAND) then this represents lack of fodder availability -> destocking
        ls_change = np.minimum(max_purchase, max_on_farm + max_off_farm - ls_obj)
        if self.insurance_payout_year:
            # assume that any leftover income from the insurance payout can be put towards livestock
            ls_change += (self.remaining_payout / ls_inp['cost']) 

        ls_change = np.floor(ls_change).astype(int)
        # ^^ if this is +ve this represents purchase. 
        # ^^ if it's -ve this represents rqd destocking due to fodder availability (only possible w/o rangeland)
        
        # attribute changes
        ls_obj += ls_change # attribute to livestock
        self.savings[t+1] -= ls_change * ls_inp['cost'] # attribute to savings
        self.destocking_rqd[t,ls_change<0] = True # record
        self.ls_purchase[t] = copy.deepcopy(ls_obj)

        # save for next time step
        self.livestock[t+1] = ls_obj # save
        self.wealth[t+1] = ls_obj*ls_inp['cost'] + self.savings[t+1]

    def adaptation(self, land, adap_properties):
        '''
        simulate adaption decision-making
        assume there is a burn-in period before any adaptation option comes into effect
        this is because there's some dependence on the initial condition / starting wealth value
        '''
        t = self.t[0]
        if adap_properties['adap'] and (t >= adap_properties['burnin_period']):
            if self.adap_type == 'coping':
                # agents engage in the adaptation option next period
                # if they had to cope this period
                self.adapt[t+1, self.neg_income[t]] = True
            elif self.adap_type == 'switching':
                # agents SWITCH adaptation types if they had to cope in this period
                self.adapt[t+1, ~self.neg_income[t]] = self.adapt[t, ~self.neg_income[t]]
                self.adapt[t+1, self.neg_income[t]] = ~self.adapt[t, self.neg_income[t]]
            elif self.adap_type == 'affording':
                # agents adapt if they can afford it
                afford = self.savings[t+1] >= (adap_properties['cost'] * self.land_area)
                self.adapt[t+1, afford] = True
            elif self.adap_type == 'always':
                # all agents adapt
                self.adapt[t+1] = True
            else:
                print('ERROR: unrecognized adaptation type')
                sys.exit()