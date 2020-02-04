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
        self.savings_post_cons_smooth = np.full([self.T, self.N], -9999)
        self.living_cost =  self.living_cost_pp * self.hh_size
        # coping measures
        self.cons_red_rqd = np.full([self.T, self.N], False)
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
        self.ls_num_lbr = np.full([self.T, self.N], -9999)
        self.ls_reprod = np.full([self.T, self.N], -9999)
        self.ls_destock = np.full([self.T, self.N], -9999)
        self.ls_stress = np.full([self.T, self.N], -9999)
        self.ls_purchase = np.full([self.T, self.N], -9999)
        self.herds_on_rangeland = np.full([self.T, self.N], -9999)
        # other
        self.n_yr_smooth = int(self.n_yr_smooth) # in case it's from POM
        self.ag_labor = np.full([self.T, self.N], np.nan)
        self.nonag_labor = np.full([self.T, self.N], np.nan)
        self.ls_labor = np.full([self.T, self.N], np.nan)

        ##### MARKET #####
        self.job_avail_total = np.floor(self.jobs_availability * self.N / self.job_increment) * self.job_increment

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

    def labor_allocation(self):
        '''
        start-of-year labor allocation to ag and non-ag activities
        '''
        t = self.t[0]
        self.ls_start[t] = copy.deepcopy(self.livestock[t])
        ## farm labor
        self.ag_labor[t] = np.minimum(self.ag_labor_rqmt*self.land_area, self.hh_size) # ppl = ppl/ha*ha
        farm_frac = np.minimum(self.hh_size / (self.ag_labor_rqmt * self.land_area), np.full(self.N,1)) # ppl / (ppl/ha*ha)
        ## livestock labor
        # destock if required (assume they are sold)
        max_ls = np.floor(np.minimum((self.hh_size - self.ag_labor[t]) / self.ls_labor_rqmt, self.livestock[t])).astype(int) # head = ppl / (ppl/head)
        destock_amt = np.ceil(np.maximum(self.livestock[t] - max_ls, 0)).astype(int)
        self.livestock[t] -= destock_amt
        self.savings[t] += destock_amt * self.all_inputs['livestock']['cost']
        self.ls_num_lbr[t] = copy.deepcopy(self.livestock[t])
        self.ls_labor[t] = self.livestock[t] * self.ls_labor_rqmt


        ## non-farm labor
        if t == 0:
            # only allocating labor as coping mechanism so don't do in first period
            self.nonag_labor[t] = 0
        else:
            # assume that people by default do what they did last year
            # ie ppl with jobs keep them by default
            nonag_lbr = self.nonag_labor[t-1]

            # DECREASES: exits from non-farm labor
            # this frees up space for other agents
            if t > 1:
                # agents decrease their non-farm labor when
                # their average income over the past N years (smoothing) allows
                # for expenditure beyond the minimum value
                yrs = np.arange(0,t) if (t-1)<self.n_yr_smooth else np.arange(t-self.n_yr_smooth,t)
                exp_income_hist = np.mean(self.income[yrs], axis=0)
                reductions = np.minimum(nonag_lbr, exp_income_hist/self.labor_wage) # ppl = $ / ($/ppl)
                reductions = np.floor(reductions/self.job_increment) * self.job_increment # be pessimistic
                nonag_lbr -= reductions

            # INCREASES in non-farm labor
            # try to increase if there was negative cash last year BEYOND the consumption reductions
            ppl_req_cash = round_up(np.maximum(-self.savings_post_cons_smooth[t-1] / self.labor_wage, 0), self.job_increment) # ppl = $ / ($/ppl)
            # and extra labor available
            max_ppl_avail = round_up(self.hh_size - self.ag_labor[t] - self.ls_labor[t] - nonag_lbr, self.job_increment)
            consider_amt = np.minimum(ppl_req_cash, max_ppl_avail)

            # allocate the jobs between those that want them
            new_allocations = self.allocate_nonfarm_labor(consider_amt, nonag_lbr)
            self.nonag_labor[t] = nonag_lbr + new_allocations        

    def allocate_nonfarm_labor(self, consider_amt, nonag_lbr):
        '''
        allocate the available non-farm jobs between the agents that want them
        assume that agents are considered successively
        (independent of how much work they need)
        with the order randomized at each time step
        '''
        # subtract off jobs that have already been allocated to continuing workers
        num_jobs_avail = self.job_avail_total - np.sum(nonag_lbr)
        # the order in which agents are considered is randomized each time step
        order_consider = np.random.choice(self.N, size=self.N, replace=False)
        # line up the agents
        cumsums = np.cumsum(consider_amt[order_consider])
        # find the last agent that gets any (first index that it is >= the limit)
        crit_ix = np.argmax(cumsums >= num_jobs_avail)
        # if the last agent has overshot, only allocate up to the limit
        cumsums[crit_ix] = num_jobs_avail
        # take the differences to give agent-level allocations
        allocs = cumsums - np.concatenate([[0], cumsums[:-1]])
        allocs[(crit_ix+1):] = 0
        # convert back to the regular agent order
        agent_allocs = np.full(self.N, np.nan)
        agent_allocs[order_consider] = allocs
        return agent_allocs

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
        # assume the baseline with consumption smoothing here
        self.income[t] = self.crop_sell_price*self.crop_production[t] - (self.living_cost*self.living_cost_min_frac).astype(int) - adap_costs + ls_milk_money

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
        # this proxies using savings to buffer income and expenditure
        self.savings[t+1] = self.savings[t] + self.income[t]
        self.neg_income[t, self.income[t] < 0] = True # record those with -ve income

        ## 2. DESIRED LIVING COSTS
        # increase expenditure up towards the desird living cost level if possible
        extra_spending = np.maximum(np.minimum(self.living_cost * (1-self.living_cost_min_frac), self.savings[t]), 0).astype(int)
        self.savings[t] -= extra_spending
        self.cons_red_rqd[t, extra_spending==0] = True
        self.savings_post_cons_smooth[t] = copy.deepcopy(self.savings[t+1])

        ## 3. STRESS DESTOCKING
        sell_rqmt = np.maximum(np.ceil(-self.savings[t+1]/ls_inp['cost']), 0).astype(int) # calculate amt rqd
        sell_amt = np.minimum(ls_obj, sell_rqmt) # restricted by available livestock
        ls_obj -= sell_amt # reduce the herdsize
        self.savings[t+1] += sell_amt * ls_inp['cost'] # add to income
        self.stress_ls_sell_rqd[t, sell_rqmt>0] = True # record
        self.ls_stress[t] = copy.deepcopy(ls_obj)

        ## 4. reset savings for those that have less than zero still
        # assume that debts can't carry over
        self.cant_cope[t, self.savings[t+1]<0] = True # record
        self.savings[t+1, self.cant_cope[t]] = 0

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
        ## A: how many can be purchased
        max_purchase = np.floor(self.savings[t+1] / ls_inp['cost'])
        ## B: how many is there labor available for (given current employment and farming)
        max_labor_tot = np.floor((self.hh_size-self.ag_labor[t]-self.nonag_labor[t]) / self.ls_labor_rqmt) # head = ppl / ppl/head
        ## C: how many can be grazed on-farm
        # average the production over the past n years
        yrs = np.arange(0,t+1) if t<self.n_yr_smooth else np.arange(t-self.n_yr_smooth+1,t+1)
        max_on_farm = np.mean(self.crop_production[yrs], axis=0) * land.residue_multiplier * land.residue_loss_factor / \
                (ls_inp['consumption']) # TLU = kgCrop * kgDM/kgCrop / kgDM/TLU
        ## D: how many can be grazed off-farm
        if rangeland.rangeland_dynamics:
            if t == 0:
                max_off_farm = self.herds_on_rangeland[t]
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
        ls_change = np.minimum(max_purchase, max_on_farm + max_off_farm - ls_obj, max_labor_tot - ls_obj)
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

def round_up(amts, stepsize):
    '''
    round amts up to the nearest stepsize
    '''
    return np.ceil(amts/stepsize) * stepsize
