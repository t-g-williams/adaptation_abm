'''
add a land acquisition into the model
'''
import numpy as np
import code
import copy
import sys
from collections import Counter

class LSLA:
    def __init__(self, inputs, agents, land, rangeland):
        '''
        implement the LSLA
        '''
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['LSLA']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        
        ## calculate area and displacement stats
        self.area = self.size * agents.N # ha = ha/agent * agents
        self.area_encroach = round_down(self.area * self.frac_retain, land.plot_size) if self.LUC == 'farm' else 0
        self.area_checks(land, rangeland, agents)

        ## change employment
        self.tot_salary_jobs = round_down(self.employment * self.size, agents.salary_job_increment)
        agents.salary_job_avail_total += self.tot_salary_jobs

        ## direct LUC
        if self.LUC == 'commons':
            rangeland.size_ha -= self.area  
        elif self.LUC == 'farm':
            new_farmland = self.area_encroach if self.encroachment == 'farm' else 0
            land.tot_area += new_farmland - self.area
            # ^^ note: land.tot_area has no function in the main model

        ## displacement
        if self.LUC == 'farm':
            self.agent_displacement(agents, land, rangeland)

        code.interact(local=dict(globals(), **locals()))

    def area_checks(self, land, rangeland, agents):
        '''
        check that there is enough area for the LUC and displacement
        '''
        if self.LUC == 'farm':
            if (self.encroachment == 'commons' and self.area_encroach > rangeland.size_ha):
                print('ERROR: not enough area to redistribute all agents to rangeland')
                sys.exit()
            elif(self.encroachment == 'farm' and self.area_encroach > (land.tot_area-self.area)):
                print('ERROR: not enough area to redistribute all agents to remaining farmland')
                sys.exit()
        elif self.LUC == 'commons':
            if self.area > rangeland.size_ha:
                print('ERROR: not enough area in commons for LSLA direct LUC')
                sys.exit()

    def agent_displacement(self, agents, land, rangeland):
        '''
        simulate the displacement of agents
        - determine how is affected
        - determine how much area each agent gets
        - determine where this area comes from
        - account for these changes
        '''
        ## 1. find the affected agents
        ## and calculate how much land they lose
        area_cumsum = np.cumsum(agents.land_area)
        crit_ix = np.where(area_cumsum >= self.area)[0][0]
        # remove the LSLA area from the cumsum
        new_cumsum = np.maximum(0, area_cumsum-self.area)
        # calculate new farm area
        self.area_after_disp = new_cumsum - np.concatenate([[0], new_cumsum[:-1]])
        self.area_lost = agents.land_area - self.area_after_disp

        ## 2. determine how much new area each agent gets
        self.assign_ha = self.redistribute_land(agents, land.plot_size)

        ## 3. determine where this new land comes from
        if self.encroachment == 'farm':
            self.encroach_ha_lost = self.farmland_encroachment(agents, land.plot_size)
        elif self.encroachment == 'commons':
            self.encroach_ha_lost = np.full(agents.N, 0.) # for the agents
            rangeland.size_ha -= self.assign_ha.sum() # displaced to rangeland
        else:
            print('ERROR: unknown encroachment type specified: {}'.format(self.encroachment))

        # save the changes
        self.net_change = -self.area_lost + self.assign_ha - self.encroach_ha_lost
        agents.land_area += self.net_change

    def redistribute_land(self, agents, plot_size):
        '''
        determine how much land each agent receives
        '''
        ## 2A probability of each agent getting each new plot
        if self.land_distribution_type == 'amt_lost':
            probs = self.area_lost / self.area_lost.sum()
        elif self.land_distribution_type == 'equal_hh':
            probs = (self.area_lost > 0).astype(float) / sum(self.area_lost>0)
        elif self.land_distribution_type == 'equal_pp':
            probs = ((self.area_lost > 0)*agents.hh_size) / sum(agents.hh_size[self.area_lost>0])
        else:
            print('ERROR: unknown land distribution type specified: {}'.format(self.land_distribution_type))
            sys.exit()
        
        ## 2B random assignment of the new plots -- lottery
        # note: an agent can end up with more land than they started with...
        # we are not sampling "without replacement" 
        # -- the agent's probability stays the same regardless of their previous assignments
        num_new_plots = int(self.area_encroach / plot_size)
        assign_ids = np.random.choice(np.arange(agents.N), p=probs, size=num_new_plots, replace=True)
        assign_counts = np.array(list(Counter(assign_ids).items()))
        assign_ha = np.full(agents.N, 0.)
        assign_ha[assign_counts[:,0]] = assign_counts[:,1] * plot_size

        return assign_ha

    def farmland_encroachment(self, agents, plot_size):
        '''
        simulate the encroachment of displaced agents into existing farmland
        return the amount of land lost due to encroachment
        '''
        if self.land_taking_type == 'random':
            owner_ids = np.repeat(agents.id, (self.area_after_disp/plot_size).astype(int))
            ids_rmv = np.random.choice(np.arange(owner_ids.shape[0]), size=num_new_plots, replace=False)
            owner_ids_rem = owner_ids[ids_rmv]
            agent_plots_lost = np.array(list(Counter(owner_ids_rem).items()))
            encroach_ha_lost = np.full(agents.N, 0.)
            encroach_ha_lost[agent_plots_lost[:,0]] = agent_plots_lost[:,1] * plot_size
        elif self.land_taking_type == 'equalizing':
            # do this iteratively:
            # iteratively take land from those with the most
            size_i = self.area_after_disp.max() # maximum land area
            ha_claimed = 0
            ha_remaining = copy.deepcopy(self.area_encroach)
            new_areas_tmp = copy.deepcopy(self.area_after_disp)
            while ha_remaining > 0:
                # print('{} remaining, size={}'.format(ha_remaining, size_i))
                # identify who has this much land
                has_size_i = new_areas_tmp==size_i
                # determine whether all need it taken or just some
                ha_claimed_i = plot_size * has_size_i.sum()
                if ha_remaining - ha_claimed_i < 0:
                    # only some of these agents need it taken
                    num_rqd = int(ha_remaining / plot_size)
                    take_land_ids = np.random.choice(np.arange(agents.N), p=(has_size_i.astype(int)/sum(has_size_i)), size=num_rqd, replace=False)
                else:
                    # take from all
                    take_land_ids = agents.id[has_size_i]
                # take it off them
                ha_remaining -= plot_size * len(take_land_ids)
                new_areas_tmp[take_land_ids] -= plot_size
                # move to next step
                size_i -= plot_size
            # save
            encroach_ha_lost = self.area_after_disp - new_areas_tmp
        else:
            print('ERROR: unknown land taking type : {}'.format(self.land_taking_type))
            sys.exit()

        return encroach_ha_lost

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