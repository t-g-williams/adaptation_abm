import numpy as np
import scipy.stats as stat
import code
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

        # wealth (cash holdings)
        # this represents the START of the year
        self.wealth = np.full([self.T+1, self.N], -9999)
        self.wealth[0] = np.random.normal(self.wealth_init_mean, self.wealth_init_sd, self.N)
        self.wealth[0][self.wealth[0]<0] = 0 # fix any -ve values
        # money
        self.income = np.full([self.T, self.N], -9999)
        self.cash_req = np.random.normal(self.cash_req_mean, self.cash_req_sd, self.N)
        self.leftover_cash = np.full([self.T+1, self.N], 0) # can be used to buy fertilizer
        # coping measures
        self.coping_rqd = np.full([self.T, self.N], False)
        self.cant_cope = np.full([self.T, self.N], False)
        # adaptation option decisions
        self.adapt = np.full([self.T+1, self.N], False)
        # decision-making
        self.risk_tol = np.full(self.N, self.risk_tolerance)
        self.fert_choice = np.full([self.T, self.N], False)
        self.fert_choice_no_risk = np.full([self.T, self.N], False)
        self.fert_costs = np.full([self.T, self.N], 0.)
        self.util = np.full([self.T, self.N], 0.)
        self.util_fert = np.full([self.T, self.N], 0.)
        self.util_fert_no_risk = np.full([self.T, self.N], 0.)
        self.util_no_risk = np.full([self.T, self.N], 0.)
        # pre-generate the agent-level "realizations" from a standard normal distribution
        # that they will each use in their utility calculations
        self.rndm_Zs = np.random.normal(size=(self.T, self.N, self.nsim_utility))

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
        
    def fertilizer_decisions(self, land, adap_properties, climate):
        '''
        make binary decisions about fertilizer
        choose fertilizer if
            (1) enough cash AND
            (2) utility is higher than not choosing fertilizer
        '''
        t = self.t[0]
        ## format the beliefs of the rainfall. shape = [agents,nsim_utility]
        rain_blf = self.rndm_Zs[t] * self.all_inputs['climate']['rain_sd'] + self.all_inputs['climate']['rain_mu']
        rain_blf[rain_blf<0] = 0
        rain_blf[rain_blf>1] = 1
        
        ## simulate the SOM and inorganic N
        [inorg_N, org_N] = land.update_soil(self, adap_properties, climate, decision=True)
        self.fert_choice[t] = True # temporary allocation -- will be overwritten at bottom of this function
        [inorg_N_fert, org_N_fert] = land.update_soil(self, adap_properties, climate, decision=True)
        
        ## calculate crop yields
        Yw = land.max_yield * rain_blf # kg/ha [agents,nsim_utility]
        Yn = inorg_N / (1/land.crop_CN_conversion+land.residue_multiplier/land.residue_CN_conversion) # kgN/ha / (kgN/kgC_yield) = kgC/ha ~= yield(perha
        Yn_fert = inorg_N_fert / (1/land.crop_CN_conversion+land.residue_multiplier/land.residue_CN_conversion) # kgN/ha / (kgN/kgC_yield) = kgC/ha ~= yield(perha
        crop_prod = np.minimum(Yw, Yn[:,None])* self.land_area[:,None] # kg
        crop_prod_fert = np.minimum(Yw, Yn_fert[:,None])* self.land_area[:,None] # kg
        
        ## income calculations
        [payouts, adap_costs] = self.calc_adap_costs(adap_properties, climate)
        income = self.crop_sell_price*crop_prod + (payouts - adap_costs - self.cash_req)[:,None] # - self.cash_req
        # for fertilizer: they can use leftover cash from last year
        fert_cost_full = self.fertilizer_cost*self.fert_kg*self.land_area
        if self.fert_use_savings:
            fert_cost = np.maximum(0, fert_cost_full - self.leftover_cash[t])
        else:
            fert_cost = fert_cost_full
        afford_fert = self.wealth[t] >= fert_cost # start-of-year cash constraints (leftover $ + "wealth")
        income_fert = self.crop_sell_price*crop_prod_fert + (payouts - adap_costs - fert_cost - self.cash_req)[:,None] # - self.cash_req
        
        ## convert to utility
        risk_tolerances = np.repeat(self.risk_tol, self.nsim_utility).reshape(self.N, self.nsim_utility)
        # no fertilizer
        util = np.full(income.shape, np.nan)
        pos = income >= 0
        util[pos] = 1 - np.exp(-income[pos] / risk_tolerances[pos])
        util[~pos] = -(1 - np.exp(income[~pos] / (0.5*risk_tolerances[~pos]))) # assume risk averion = loss aversion (note: this is NOT empirically justified in decision theory)
        # with fertilizer
        util_fert = np.full(income_fert.shape, np.nan)
        pos_fert = income_fert >= 0
        util_fert[pos_fert] = 1 - np.exp(-income_fert[pos_fert] / risk_tolerances[pos_fert])
        util_fert[~pos_fert] = -(1 - np.exp(income_fert[~pos_fert] / (0.5*risk_tolerances[~pos_fert]))) # assume risk averion = loss aversion (note: this is NOT empirically justified in decision theory)
       # calculate expected utility
        exp_util = util.mean(1)
        exp_util_fert = util_fert.mean(1)
        if self.fert_cash_constrained:
            exp_util_fert[~afford_fert] = -99 # cash constraints
        
        ## make decision
        self.fert_choice[t] = exp_util_fert > exp_util
        self.fert_choice_no_risk[t] = income_fert.mean(1) > income.mean(1)
        self.util[t] = exp_util        
        self.util_fert[t] = exp_util_fert  
        self.util_no_risk[t] = income.mean(1)      
        self.util_fert_no_risk[t] = income_fert.mean(1)  

        ##  # with fertilizer - option 2 with diff baseline
        # util_fert = np.full(income_fert.shape, np.nan)
        # dif = income_fert - income
        # pos_fert = dif >= 0
        # util_fert[pos_fert] = 1 - np.exp(-dif[pos_fert] / risk_tolerances[pos_fert])
        # util_fert[~pos_fert] = -(1 - np.exp(dif[~pos_fert] / (0.5*risk_tolerances[~pos_fert]))) # assume risk averion = loss aversion (note: this is NOT empirically justified in decision theory)
        # if self.fert_cash_constrained:
        #     util_fert[~afford_fert] = -99
        # self.fert_choice[t] = util_fert.mean(1) > 0
        # code.interact(local=dict(globals(), **locals()))

    def calculate_income(self, land, climate, adap_properties):
        '''
        calculate end-of-year income
        '''
        t = self.t[0]
        [payouts, adap_costs] = self.calc_adap_costs(adap_properties, climate)
        self.fert_costs[t] = self.fert_choice[t] * self.fertilizer_cost * self.fert_kg * self.land_area

        # income = crop_sales - cash_req - adap_costs - fertilizer costs
        self.income[t] = self.crop_sell_price*self.crop_production[t] - self.cash_req - adap_costs - self.fert_costs[t]
        
        if self.insurance_payout_year:
            # assume that agents first use their payout to neutralize their income
            # and any left over, they use to buy fodder
            # which will increase their maximum wealth capacity
            self.remaining_payout = np.minimum(np.maximum(payouts+self.income[t], 0), payouts) # outer "minimum" is in case their income is +ve --> they can only use the payout for fodder
            self.income[t] += payouts.astype(int)
            # code.interact(local=dict(globals(), **locals()))

    def calc_adap_costs(self, adap_properties, climate):
        # costs and payouts for adaptation option
        t = self.t[0]
        adap_costs = np.full(self.N, 0.)
        payouts = np.full(self.N, 0.)
        self.insurance_payout_year = False
        if adap_properties['type'] in ['insurance','both']:
            # costs
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]
            # payouts
            if climate.rain[t] < adap_properties['magnitude']:
                payouts[self.adapt[t]] = adap_properties['payout'] * self.land_area[self.adapt[t]]
                self.insurance_payout_year = True
        if adap_properties['type'] in ['cover_crop','both']:
            # note: with "both", the cost parameter includes the cost of both options. it's input twice and written over here, so we are not double counting
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]

        return payouts, adap_costs


    def coping_measures(self, land):
        '''
        calculate end-of-year income balance
        and simulate coping measures
        '''
        t = self.t[0]
        # assume those with -ve income are required to engage in coping measure
        self.coping_rqd[t, self.income[t] < 0] = True
        # add (or subtract) agent income to their wealth
        # this proxies the effect of buying (+ve income) or selling (-ve income) livestock
        self.wealth[t+1] = self.wealth[t] + self.income[t]
        # record agents with -ve wealth (not able to cope)
        self.cant_cope[t, self.wealth[t+1] < 0] = True
        # wealth (/livestock) constraints: can't carry more than your crop residues allows
        # if 80% of livestock must be grazed on fodder, then the maximum wealth you can carry
        # is 20% of your current livestock herds + whatever you can sustain from your crop residues
        # i.e. it's assumed that some fraction of your livestock are fully independent of crop residue
        # rather than all livestock requiring this fraction of feed from fodder
        buffer_yrs = 1
        # crop_prod = np.mean(self.crop_production[max(0,t-buffer_yrs):t], axis=0)
        # print(crop_prod)
        max_ls_fodder = self.crop_production[t] * land.residue_multiplier * land.residue_loss_factor / \
                (land.livestock_residue_factor) # TLU = kgCrop * kgDM/kgCrop / kgDM/TLU
        max_wealth = max_ls_fodder*self.livestock_cost + (1-land.livestock_frac_crops) * self.wealth[t]

        if self.insurance_payout_year:
            # assume that any leftover income from the insurance payout is converted to livestock/wealth
            max_wealth += self.remaining_payout
        
        too_much = self.wealth[t+1] > max_wealth
        # too_much[too_much==True] = False # TEMPORARY!!!
        self.leftover_cash[t+1,too_much] = self.wealth[t+1,too_much] - max_wealth[too_much].astype(int)
        self.wealth[t+1, too_much] = max_wealth[too_much]
        self.wealth[t+1, self.wealth[t+1] < self.max_neg_wealth] = self.max_neg_wealth
        # if t == 20:
        # code.interact(local=dict(globals(), **locals()))
        ## TEMPORARY
        # self.wealth[t+1, self.wealth[t+1]<0] = 0

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
                self.adapt[t+1, self.coping_rqd[t]] = True
            elif self.adap_type == 'switching':
                # agents SWITCH adaptation types if they had to cope in this period
                self.adapt[t+1, ~self.coping_rqd[t]] = self.adapt[t, ~self.coping_rqd[t]]
                self.adapt[t+1, self.coping_rqd[t]] = ~self.adapt[t, self.coping_rqd[t]]
            elif self.adap_type == 'affording':
                # agents adapt if they can afford it
                afford = self.wealth[t+1] >= (adap_properties['cost'] * self.land_area)
                self.adapt[t+1, afford] = True
            elif self.adap_type == 'always':
                # all agents adapt
                self.adapt[t+1] = True
            else:
                print('ERROR: unrecognized adaptation type')
                sys.exit()