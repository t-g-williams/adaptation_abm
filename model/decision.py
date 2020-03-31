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
        if agents.t[0] == 0:
            Decision.init_allocations(agents, land, market)

        else:
            inp = agents.all_inputs['decisions']
            if inp['framework'] == 'util_max':
                Decision.utility_max(agents, land, market, inp)

    def init_allocations(agents,land,market):
        '''
        land/labor allocations in first year
        '''
        t = 0
        agents.ls_start[t] = copy.deepcopy(agents.livestock[t])
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
        max_ls = np.floor(np.minimum((agents.hh_size - agents.tot_ag_labor[t]) / agents.ls_labor_rqmt, agents.livestock[t])).astype(int) # head = ppl / (ppl/head)
        destock_amt = np.ceil(np.maximum(agents.livestock[t] - max_ls, 0)).astype(int)
        agents.livestock[t] -= destock_amt
        ls_inc = destock_amt * market.livestock_cost
        agents.savings[t] += ls_inc
        agents.ls_num_lbr[t] = copy.deepcopy(agents.livestock[t])
        agents.ls_labor[t] = agents.livestock[t] * agents.ls_labor_rqmt
        agents.ls_income[t] += ls_inc

        # only allocating labor as coping mechanism so don't do in first period
        agents.salary_labor[t] = 0
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
        lbrs['livestock'] = agents.livestock[t] * agents.ls_labor_rqmt # use current livestock amt
        lbrs['non_farm'] = agents.salary_labor[t-1] # previous value
        for crop in crops: # use previous values
            lbrs['ag_'+crop] = copy.deepcopy(agents.ag_labor[crop][t-1])
            ha_farmed[crop] = copy.deepcopy(land.ha_farmed[crop][t-1]) 

        # identify the labor constrained agents
        # these agents will make decisions based on labor productivity
        lbr_rem = agents.hh_size - np.sum(np.array([lbrs[l] for l in list(lbrs.keys())]), axis=0)
        lbr_constrained = lbr_rem <= 0

        # loop over the options
        for act in inp['actions']:
            # copy
            lbr = copy.deepcopy(lbrs)
            ha = copy.deepcopy(ha_farmed)

            ## 1. make the allocations for this option
            if act == 'nothing':
                ixs = np.full(agents.N, True)
            elif act in ['incr_trad_ag','incr_int_ag','incr_div_ag']:
                [ixs, ha] = Decision.change_ag_type(agents, ha, lbr_constrained, incr_ha, t, act, crops)
            # elif act == 'incr_nf_labor':
            #     lbr['non_farm'] += incr_ppl
            #     xxx
            else:
                print('ERROR: unrecognized decision option "{}"'.format(act))
                sys.exit()

            # update labor allocations
            for crop in crops:
                lbrs['ag_'+crop] = ha[crop] * agents.ag_labor_rqmt[crop] * (1-land.fallow_frac*agents.fallow[t]) # scale for fallowing
            
            # make sure there is labor for livestock
            # reduce livestock if necessary
            lbr_rem = agents.hh_size - np.sum(np.array([lbr[l] for l in list(lbr.keys())]), axis=0)
            negs = lbr_rem<0
            ls_rmv = np.full(agents.N, 0.)
            ls_rmv[negs] = np.minimum(-lbr_rem[negs], lbr['livestock'][negs])
            lbr['livestock'] -= ls_rmv
            lbr_rem += ls_rmv
            ixs *= (lbr_rem>=0) # stop considering for agents that don't have enough labor here

            # check
            if ((ixs.sum()<agents.N) and (act=='nothing')):
                print('Some agents can not carry out "nothing" option')
                code.interact(local=dict(globals(), **locals()))

            ## 2. calculate the return for this option
            # because the beliefs are stochastic (i.e., have mean and variance)
            # we do not analytically calculate the "expected" return
            # rather, we simulate over the uncertainty of the beliefs
            # (assuming that each belief is perfectly correlated - i.e., a shock in trad_ag is a shock in int_ag)
            # 2a: convert to arrays to make the math easier
            keyz = list(lbr.keys())
            lbr_np = np.array([lbr[i] for i in keyz]) # dimension: activity, agents
            mu_np = np.array([agents.blf.mu[i][t] for i in keyz])
            sd_np = np.sqrt(np.array([agents.blf.var[i][t] for i in keyz]))
            # 2b: combine the beliefs (mu, sigma) with randomly-generated values to generate a sample of "random productivites"
            rndm_productivities = agents.rndm_Zs[t] * sd_np[:,:,None] + mu_np[:,:,None] # dimension: (activity, agent, simulation)
            # 2c: weight each probability by the allocated labor (sum over the activities)
            rndm_incomes = np.sum(lbr_np[:,:,None] * rndm_productivities, axis=0) # dimension: (agent, simulation)
            # 2d: convert to utility -- exponential function
            rndm_utils = 1 - np.exp(-rndm_incomes / agents.risk_aversion[:,None])
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.scatter(rndm_incomes.flatten(),rndm_utils.flatten())
            # fig.savefig('utils_{}.png'.format(agents.all_inputs['decisions']['risk_aversion_params'][0]))
            # 2e: calculate expected utility
            exp_util = np.mean(rndm_utils, axis=1)

            code.interact(local=dict(globals(), **locals()))

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