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
        t = agents.t[0]
        agents.ls_obj = agents.livestock[t] # initialize the temporary object to be worked with
        if t == 0:
            Decision.init_allocations(agents, land, market)
        else:
            inp = agents.all_inputs['decisions']
            if inp['framework'] == 'util_max':
                Decision.utility_max(agents, land, market, inp)

            # allocate the non-farm labor
            # first, determine who already has jobs (from last time)
            # take the minimum of their previous labor and their current allocation
            continuing_lbr = np.min(agents.salary_labor[(t-1):(t+1)], axis=0)
            # then, allocate the remaining
            agents.salary_tot_consider_amt[t] = agents.salary_labor[t]
            new_allocations = market.allocate_salary_labor(agents, agents.salary_labor[t], continuing_lbr)
            agents.salary_labor[t] = continuing_lbr + new_allocations

        agents.ls_decision[t] = copy.deepcopy(agents.ls_obj)

    def init_allocations(agents,land,market):
        '''
        land/labor allocations in first year
        '''
        t = 0
        ## fallow and crop type allocation
        # assume fallow by default (except LSLA outgrowers)
        # (note: amt of land farmed can be a fraction of a field - i.e., part of a field can be fallowed)
        # land.farmed_fraction[t] = 1 - land.fallow_frac * agents.fallow[t]
        # ag type allocation: assume all are traditional
        land.ha_farmed['trad'][t, ~land.outgrower[t]] = agents.land_area[~land.outgrower[t]] # represents total area, including fallow

        ## farm labor
        for crop in land.ag_types:
            lbr_rqmt = agents.ag_labor_rqmt[crop] * land.ha_farmed[crop][t] * (1-land.fallow_frac*agents.fallow[t]) # scale for fallowing
            agents.ag_labor[crop][t] = np.minimum(lbr_rqmt, agents.hh_size) # ppl = ppl/ha*ha
            agents.tot_ag_labor[t] += agents.ag_labor[crop][t] # add to total

        ## livestock labor
        # destock if required (assume they are sold)
        max_ls = np.floor(np.minimum((agents.hh_size - agents.tot_ag_labor[t]) / agents.ls_labor_rqmt, agents.ls_obj)).astype(int) # head = ppl / (ppl/head)
        destock_amt = np.ceil(np.maximum(agents.ls_obj - max_ls, 0)).astype(int)
        agents.ls_obj -= destock_amt
        ls_inc = destock_amt * market.livestock_cost
        agents.savings[t] += ls_inc
        agents.ls_labor[t] = agents.ls_obj * agents.ls_labor_rqmt
        agents.ls_sell_income[t] += ls_inc

        # only allocating labor as coping mechanism so don't do in first period
        agents.salary_labor[t] = 0
        agents.choice_ixs[t] = 0
        # code.interact(local=dict(globals(), **locals()))    
    
    def utility_max(agents, land, market, inp):
        '''
        make decisions using utility maximization framework
        '''
        t = agents.t[0]
        incr_ha = land.plot_size # increment for LUC
        incr_ppl = market.salary_job_increment # increment for labor allocation
        crops = np.array(land.ag_types)

        # build up baseline dataframes of labor and land allocations
        lbrs = OrderedDict()
        ha_farmed = {} # represents total area, including fallow
        lbrs['livestock'] = agents.ls_obj * agents.ls_labor_rqmt # use current livestock amt
        lbrs['non_farm'] = agents.salary_labor[t-1] # previous value
        for crop in crops: # use previous values
            lbrs['ag_'+crop] = copy.deepcopy(agents.ag_labor[crop][t-1])
            ha_farmed[crop] = copy.deepcopy(land.ha_farmed[crop][t-1]) 

        # identify the labor constrained agents
        # these agents will make decisions based on labor productivity
        lbr_rem = agents.hh_size - np.sum(np.array([lbrs[l] for l in list(lbrs.keys())]), axis=0)
        lbr_constrained = lbr_rem <= 0

        # format beliefs for utility calculations
        keyz = list(lbrs.keys())
        mu_np = np.array([agents.blf.mu[i][t] for i in keyz])
        sd_np = np.sqrt(np.array([agents.blf.var[i][t] for i in keyz]))

        # init objects for saving the data
        utils = np.full((len(inp['actions']), agents.N), -99.99)
        action_lbrs = OrderedDict()
        action_ha = OrderedDict()
        ls_sold = OrderedDict()

        # loop over the options
        for a, act in enumerate(inp['actions']):
            # copy
            lbr = copy.deepcopy(lbrs)
            ha = copy.deepcopy(ha_farmed)

            ## 1. make the allocations for this option
            # ixs_1 represent the indexes for which each option is feasible
            # based on land/labor availability
            if act == 'nothing':
                ixs_1 = np.full(agents.N, True)

            elif act in ['incr_trad_ag','incr_int_ag','incr_div_ag']:
                [ixs_1, ha] = Decision.change_ag_type(agents, ha, lbr_constrained, incr_ha, t, act, crops)
                # update labor allocations
                for crop in crops:
                    lbr['ag_'+crop] = ha[crop] * agents.ag_labor_rqmt[crop] * (1-land.fallow_frac*agents.fallow[t]) # scale for fallowing

            elif act == 'incr_nf_labor':
                ixs_1 = np.full(agents.N, True) # issues will be sorted out in labor balancing
                lbr['non_farm'] += incr_ppl

            elif act == 'decr_nf_labor':
                lbr['non_farm'] -= incr_ppl
                ixs_1 = lbr['non_farm'] > 0 # agents with -ve non-farm labor now are not valid for this option

            else:
                print('ERROR: unrecognized decision option "{}"'.format(act))
                sys.exit()
            
            n1 = ixs_1.sum()
            if n1 == 0:
                continue # next action if none are considering

            ## 2. labor balancing if necessary
            # make sure there is labor for livestock
            # reduce livestock if necessary
            lbr_rem = agents.hh_size[ixs_1] - np.sum(np.array([lbr[l][ixs_1] for l in list(lbr.keys())]), axis=0)
            negs = lbr_rem<0
            ls_rmv = np.full(n1, 0.)
            ls_red_rqd = -round_down(lbr_rem[negs], agents.ls_labor_rqmt)
            ls_rmv[negs] = np.minimum(ls_red_rqd, lbr['livestock'][ixs_1][negs])
            lbr['livestock'][ixs_1] -= ls_rmv
            lbr_rem += ls_rmv
            # ^ note: these objects only contain ixs_1, so their size is different
            # ixs *= (lbr_rem>=0) # stop considering for agents that don't have enough labor here
            ## ^ note : there's a potential issue with this - maybe agents should consider dropping the lowest utility activity in this case

            # if lbr_rem is negative, take away from the least productive activity 
            # (excluding the one being considered, if appropriate)

            [lbr, ixs_2] = Decision.balance_labor(lbr, lbr_rem, ixs_1, ha, act, incr_ppl, incr_ha, t, mu_np, keyz, agents)

            # final agents to include
            ids = agents.id[ixs_1][ixs_2]
            ixs = np.in1d(agents.id, ids)
            n2 = ixs.sum()
            if n2 == 0:
                continue

            # check
            if ((n2<agents.N) and (act=='nothing')):
                print('Some agents can not carry out "nothing" option')
                code.interact(local=dict(globals(), **locals()))

            ## 2. calculate the return for this option
            # because the beliefs are stochastic (i.e., have mean and variance)
            # we do not analytically calculate the "expected" return
            # rather, we simulate over the uncertainty of the beliefs
            # (assuming that each belief is perfectly correlated - i.e., a shock in trad_ag is a shock in int_ag)
            # 2a: convert to arrays to make the math easier
            lbr_np = np.array([lbr[i][ixs] for i in keyz]) # dimension: activity, agents
            # 2b: combine the beliefs (mu, sigma) with randomly-generated values to generate a sample of "random productivites"
            rndm_productivities = agents.rndm_Zs[t,ixs] * sd_np[:,ixs,None] + mu_np[:,ixs,None] # dimension: (activity, agent, simulation)
            # 2c: weight each probability by the allocated labor (sum over the activities)
            rndm_incomes = np.sum(lbr_np[:,:,None] * rndm_productivities, axis=0) # dimension: (agent, simulation)
            # 2d: convert to utility -- exponential function
            rndm_utils = 1 - np.exp(-rndm_incomes / agents.risk_aversion[ixs,None])
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.scatter(rndm_incomes.flatten(),rndm_utils.flatten())
            # fig.savefig('utils_{}.png'.format(agents.all_inputs['decisions']['risk_aversion_params'][0]))
            # 2e: calculate expected utility
            utils[a,ixs] = np.mean(rndm_utils, axis=1) # dimension : (agent)

            # save the labor and land
            action_lbrs[act] = lbr
            action_ha[act] = ha
            ls_sold[act] = np.full(agents.N, 0.)
            ls_sold[act][ixs_1] = ls_rmv

        # choose the best option
        Decision.select_max_utility(agents, land, market, utils, action_lbrs, action_ha, ls_sold, inp['actions'], crops, t)

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

    def select_max_utility(agents, land, market, utils, action_lbrs, action_ha, ls_sold, actions, crops, t):
        '''
        select the option for each agent with the highest utility
        and attribute the choice to the agents object
        '''
        agents.choice_ixs[t] = np.argmax(utils, axis=0) # identify index of maximum utility
        ls_orig = copy.deepcopy(agents.ls_obj)

        for a, act in enumerate(actions):
            ixs = agents.choice_ixs[t]==a # agents for which this is the best choice

            if np.sum(ixs)>0:
                # update the labor and ha farmed
                for crop in crops:
                    agents.ag_labor[crop][t, ixs] = action_lbrs[act]['ag_'+crop][ixs]
                    land.ha_farmed[crop][t, ixs] = action_ha[act][crop][ixs]
                    # add to totals
                    agents.tot_ag_labor[t,ixs] += action_lbrs[act]['ag_'+crop][ixs]

                # salary labor -- this is just allocations
                agents.salary_labor[t,ixs] = action_lbrs[act]['non_farm'][ixs]
                
                # livestock -- allocate labor and pay agent if they sold them
                agents.ls_obj[ixs] = np.floor(action_lbrs[act]['livestock'][ixs] / agents.ls_labor_rqmt).astype(int)
                agents.ls_labor[t,ixs] = agents.ls_obj[ixs] * agents.ls_labor_rqmt
                agents.ls_sell_income[t,ixs] += (ls_sold[act][ixs] / agents.ls_labor_rqmt * market.livestock_cost).astype(int)

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
