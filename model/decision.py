'''
decision-making functions for the agents
'''
import numpy as np
import pandas as pd
import code
import copy
import sys
from collections import OrderedDict

class Decision():
    def annual_decisions(agents, land, market):
        '''
        wrapper function for decision-making
        '''
        # different behavior in 1st year
        inp = agents.all_inputs['decisions']
        t = agents.t[0]
        agents.ls_obj = agents.livestock[t] # initialize the temporary object to be worked with
        if t == 0:
            Decision.imposed_decision(agents, land, market, inp) # no decisions
        else:
            if inp['framework'] == 'imposed':
                Decision.imposed_decision(agents, land, market, inp) # no decisions
            elif inp['framework'] == 'util_max':
                Decision.utility_max(agents, land, market, inp)

        if (market.nonfarm_labor and (t > 0)):
            # allocate the non-farm labor
            # first, determine who already has jobs (from last time)
            # take the minimum of their previous labor and their current allocation
            continuing_lbr = np.min(agents.salary_labor[(t-1):(t+1)], axis=0)
            # then, allocate the remaining
            agents.salary_tot_consider_amt[t] = agents.salary_labor[t]
            new_allocations = market.allocate_salary_labor(agents, agents.salary_labor[t], continuing_lbr)
            agents.salary_labor[t] = continuing_lbr + new_allocations

        agents.ls_decision[t] = copy.deepcopy(agents.ls_obj)
        agents.ls_labor[t] = agents.ls_obj * agents.ls_labor_rqmt

    def imposed_decision(agents, land, market, inp):
        '''
        impose a single behavior for all agents
        '''
        adap_info = agents.all_inputs['adaptation']
        t = agents.t[0]
        # code.interact(local=dict(globals(), **locals()))      
        fac = (1-adap_info['conservation']['area_req']) if inp['imposed_action']['conservation'] else 1
        land.ha_farmed['subs'][t] = fac * agents.land_area # represents total area, including fallow

        agents.ag_labor['subs'][t] = agents.ag_labor_rqmt['subs'] * land.ha_farmed['subs'][t]
        agents.tot_ag_labor[t] = agents.ag_labor['subs'][t]
        agents.salary_labor[t] = 0
        if inp['imposed_action']['fertilizer']:
            land.fertilizer['subs'][t] = adap_info['fertilizer']['application_rate']
        # choices
        for k, v in inp['imposed_action'].items():
            agents.choices[k][t] = v

    def utility_max(agents, land, market, inp):
        '''
        make decisions using utility maximization framework
        '''
        t = agents.t[0]
        incr_ha = land.plot_size # increment for LUC
        action_names = list(inp['actions'].keys())
        adap_info = agents.all_inputs['adaptation']
        nZ = inp['nsim_utility']
        risk_tolerances = np.repeat(agents.risk_tolerance, nZ).reshape(agents.N, nZ)
        
        # format beliefs for utility calculations
        mu_np = np.array([agents.blf.mu[i][t] for i in agents.blf.quantities])
        sd_np = np.sqrt(np.array([agents.blf.var[i][t] for i in agents.blf.quantities]))
        # integrate with the random normal simulations
        blf_dist = agents.rndm_Zs[t][None,:,:] * sd_np[:,:,None] + mu_np[:,:,None] # dimension: (agent,nZ) * (blf_type, agent) --> (blf_type,agent,nZ)
        blfs = {}
        for qi, quant in enumerate(agents.blf.quantities):
            blfs[quant] = blf_dist[qi] # shape: (agent,nZ)
            # constrain
            blfs[quant][blfs[quant]<0] = 0
            if quant == 'rain':
                blfs[quant][blfs[quant]>1] = 1

        # init objects for saving the data
        exp_util = np.full((len(agents.decision_options), agents.N), np.nan)

        # loop over the options
        for a, act in enumerate(agents.decision_options):
            crop = 'subs' # for now..
            ha_farmed = agents.land_area*(1-adap_info['conservation']['area_req']) if act['conservation'] else agents.land_area
            
            ## 0. calculate the start-of-year costs of the option (this is the same each year)
            start_costs = (market.farm_cost[crop] * ha_farmed).astype(int)
            if act['fertilizer']:
                start_costs += (market.fertilizer_cost * ha_farmed * adap_info['fertilizer']['application_rate']).astype(int) # birr/kg * ha * kg/ha = birr
            # determine feasibility
            agents.option_feasibility[t,a] = start_costs <= agents.savings[t]

            ## 1. estimate the future levels of SOM and available inorganic nutrients
            # initialize
            som = np.full([inp['horizon']+1, agents.N], 0.) # (kgN/ha) at start of year (before mineralization/addition)
            som[0] = land.organic[t]
            residues = np.full([inp['horizon']+1, agents.N], 0.) # from the PREVIOUS period
            residues[0] = land.residue_production[t-1] / land.land_area / land.residue_CN_conversion # (kgN/ha) assume applied equally over all land
            inorg = np.full([inp['horizon'], agents.N], 0.) # (kgN/ha) after mineralization etc
            org_added = np.full([inp['horizon'], agents.N], 0.)
            crop_yield = np.full([inp['horizon'], agents.N, nZ], 0)

            # iterate over the years: calculate yields and SOM evolution
            for ti in range(inp['horizon']):
                ### A: soil dynamics
                # add the organic and inorganic inputs
                org_added[ti] += residues[ti] # add residues from previous period
                if act['conservation']:
                    org_added[ti] += adap_info['conservation']['organic_N_added']
                if act['fertilizer']:
                    inorg[ti] += adap_info['fertilizer']['application_rate']
                # mineralization
                som_mineralized = land.slow_mineralization_rate * som[ti]
                org_added_mineralized = land.fast_mineralization_rate * org_added[ti]
                inorg[ti] += som_mineralized + org_added_mineralized
                # save SOM for next year
                som[ti+1] = som[ti] - som_mineralized + (org_added[ti] - org_added_mineralized)
                som[ti+1][som[ti+1]<0] = 0
                som[ti+1][som[ti+1]>land.max_organic_N] = land.max_organic_N
                # inorganic losses
                inorg_loss_rate = (land.loss_min + (land.max_organic_N-som[ti+1])/land.max_organic_N * (land.loss_max - land.loss_min))
                inorg[ti] -= inorg[ti] * inorg_loss_rate
                inorg[ti][inorg[ti] < 0] = 0 # constrain
                ### B: yields
                y_w = blfs['rain'] * land.max_yield[crop] # shape: (agent, nZ) - just use expected rainfall amt here
                y_n = inorg[ti] / (1/land.crop_CN_conversion+land.residue_multiplier/land.residue_CN_conversion) #  shape: (agent). kgN/ha / (kgN/kgC_yield) = kgC/ha ~= yield(perha)
                crop_yield[ti] = np.minimum(y_w, y_n[:,None]).astype(int) # shape: (agent,nZ)
                # residue production -- for updating soil just take the average yield of these
                residues[ti+1] = (np.mean(crop_yield[ti],axis=-1) * ha_farmed * land.residue_multiplier * land.residue_loss_factor) / \
                        land.land_area / land.residue_CN_conversion # (kg * ha = kg total) *kgN/kgtot /ha_tot
            
            # calculate net income, incorporating price uncertainty
            # accounting for subsistence requirements
            crop_prod = crop_yield * ha_farmed[None,:,None] # dimension: (horizon, agent, nZ)
            mkt_prod = np.maximum(crop_prod - agents.food_rqmt[None,:,None]*(crop=='subs'), 0) # subsistence crop goes first for subsistence
            cons_deficit = np.maximum(agents.food_rqmt[None,:,None] - crop_prod*(crop=='subs'), 0) # mkt crop can't be used for subsistence
            ag_profits = (mkt_prod * blfs['price_{}'.format(crop)][None,:,:]).astype(int) # dimension:(horizon,agent,nZ)
            food_costs = cons_deficit * blfs['price_subs'][None,:,:]
            # other income sources and costs throughout the year
            other_income = agents.ls_obj * agents.all_inputs['livestock']['income'] # dimension:(agent)
            year_costs = (agents.living_cost * agents.living_cost_min_frac).astype(int)
            net_income = ag_profits + other_income[None,:,None] - food_costs - start_costs[None,:,None] - year_costs[None,:,None] # dimension:(horizon,agent,nZ)
            
            # convert to NPV (take the mean rather than the sum. it's like a weighted average income)
            npv = np.mean(net_income * agents.npv_vals[:,:,None], axis=0) # dimension: (agent,nZ)
            pos = npv>0

            # convert to utility (dimension: (agent,nZ))
            rndm_utils = np.full(npv.shape, np.nan)
            rndm_utils[pos] = 1 - np.exp(-npv[pos] / risk_tolerances[pos])
            rndm_utils[~pos] = -(1 - np.exp(npv[~pos] / risk_tolerances[~pos])) # assume risk averion = loss aversion (note: this is NOT empirically justified in decision theory)
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.scatter(npv.flatten(),rndm_utils.flatten())
            # fig.savefig('utils_{}.png'.format(agents.all_inputs['decisions']['risk_tolerance_params'][0]))
            # sys.exit()
            # 2e: calculate expected utility
            exp_util[a] = np.mean(rndm_utils, axis=1) # dimension : (agent)

        # choose the best option
        Decision.select_max_utility(agents, land, exp_util, t, crop, adap_info)

    def change_ag_type(agents,ha,lbr_constrained,incr_ha,t,act,crops):
        '''
        try to increase the area of "crop"
        taking away from "other1" or "other2"
        '''
        crop = act.split('_')[1]
        other1 = crops[crops!=crop][0]
        other2 = crops[crops!=crop][1]

        # who has area to give?
        a1 = ha[other1] > 0
        a2 = ha[other2] > 0
        # what's the preference?
        # non-labor constrained agents choose based on land productivity per ha
        pref1 = agents.blf.mu['ag_'+other1][t] >= agents.blf.mu['ag_'+other2][t] # $/ppl
        pref1[~lbr_constrained] = (agents.blf.mu['ag_'+other1][t,~lbr_constrained]*agents.ag_labor_rqmt[other1]) >= \
                        (agents.blf.mu['ag_'+other2][t,~lbr_constrained]*agents.ag_labor_rqmt[other2]) # $/ppl * ppl/ha = $/ha

        # give 1 if (you have it AND a preference for it) OR (if you have it AND have no 2)
        give1 = (pref1 * a1) | (a1 * ~a2)
        # give 2 if (you have a preference for it AND you have it) OR (you have it AND don't have 1)
        give2 = (~pref1 * a2) | (a2 * ~a1)
        ixs = give1 | give2 # ixs that can give something
        # make area allocations
        ha[other1][give1] -= incr_ha
        ha[other2][give2] -= incr_ha
        ha[crop][ixs] += incr_ha

        return ixs, ha

    def balance_labor(lbr, lbr_rem, ixs, ha, act, incr_ppl, incr_ha, t, blf, keyz, agents):
        '''
        if remaining labor is negative, 
        '''
        # create deficit object (+ve means a deficit)
        deficit = -lbr_rem
        deficit[deficit<0] = 0

        if np.sum(deficit)==0:
            # don't do anything if no deficit
            return lbr, np.full(deficit.shape[0], True)
        else:
            # eat away at the deficit from the least productive activities
            blf_pref = np.argsort(blf[:,ixs], axis=0)
            for i in range(blf_pref.shape[0]):
                # i represents the preference order (i=0 means least preferred - i.e. lowest return by labor)
                # which option does each agent consider? loop over them until the deficits are filled (if possible)
                for j in range(len(keyz)):
                    if keyz[j] in ['non_farm', 'livestock']:
                        continue # only drop from ag activities
                    else:
                        # ag_drop: this means that an agent's ith preference is option j
                        ag_drop = (blf_pref == i)[j] * (deficit>0)
                        id_drop = agents.id[ixs][ag_drop]
                        if np.sum(ag_drop) > 0:
                            # these agents drop this activity
                            lbr_max = lbr[keyz[j]][id_drop]
                            # increment amount: the amount of labor drop to go down one field size
                            lnd = agents.all_inputs['land']
                            incr_amt = agents.ag_labor_rqmt[keyz[j].split('_')[1]] * lnd['plot_size'] * (1-agents.fallow[t,id_drop]*lnd['fallow_frac'])
                            # round up the deficit - i.e. agents must stop farming an entire field
                            def_round = round_up(deficit[ag_drop], incr_amt)
                            drop_amt = np.maximum(lbr_max-def_round,lbr_max)
                            lbr[keyz[j]][id_drop] -= drop_amt
                            deficit[ag_drop] -= drop_amt
            
            # calculate new labor remaining
            lbr_rem_new = agents.hh_size[ixs] - np.sum(np.array([lbr[l][ixs] for l in list(lbr.keys())]), axis=0)
            ixs_2 = lbr_rem_new >= 0

            return lbr, ixs_2

    def select_max_utility(agents, land, exp_util, t, crop, adap_info):
        '''
        select the option for each agent with the highest utility
        and attribute the choice to the agents object
        '''
        # code.interact(local=dict(globals(), **locals()))
        agents.exp_util[t] = copy.deepcopy(exp_util)
        # effects of feasbility
        exp_util[~agents.option_feasibility[t]] = -99999
        # select the best feasible option
        agents.choice_ixs[t] = np.argmax(exp_util, axis=0) # identify index of maximum utility
        # ^ note: if all are infeasible, the agent will choose the first index (presume this is the baseline option)
        
        for a, act in enumerate(agents.decision_options):
            ixs = agents.choice_ixs[t]==a # agents for which this is the best choice
            ha_farmed = agents.land_area*(1-adap_info['conservation']['area_req']) if act['conservation'] else agents.land_area
            land.ha_farmed[crop][t,ixs] = ha_farmed[ixs]
            agents.ag_labor[crop][t,ixs] = agents.ag_labor_rqmt[crop] * ha_farmed[ixs]
            agents.tot_ag_labor[t,ixs] = agents.ag_labor[crop][t,ixs]
            if act['fertilizer']:
                land.fertilizer[crop][t,ixs] = adap_info['fertilizer']['application_rate'] # kg N/ha farmed
            for k, v in act.items():
                agents.choices[k][t,ixs] = v

        agents.salary_labor[t] = 0

def round_up(amts, stepsize):
    '''
    round amts up to the nearest stepsize
    '''
    return np.ceil(amts/stepsize) * stepsize

def round_down(amts, stepsize):
    '''
    round amts up to the nearest stepsize
    '''
    return np.floor(amts/stepsize) * stepsize
