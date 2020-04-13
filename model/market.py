import numpy as np
import code
from .agents import round_down

class Market():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['market']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # set crop prices
        self.crop_sell = {}
        for crop_type, params in self.crop_sell_params.items():
            rvs = np.random.normal(params[0], params[1], size=self.T)

            # price is a weighted sum of this year and last year's price
            # higher values of param[2] mean higher correlation between years
            for t in range(1,self.T):
                rvs[t] = params[2] * rvs[t-1] + (1-params[2]) * rvs[t]
            self.crop_sell[crop_type] = rvs

        # job availability
        self.p_wage_labor = np.full(self.T, np.nan)

        ##### MARKET #####
        N = self.all_inputs['model']['n_agents']
        self.salary_job_avail_total = round_down(self.salary_jobs_availability * N, self.salary_job_increment)
        self.wage_job_avail_total = round_down(self.wage_jobs_availability * N, self.wage_job_increment)

    def allocate_salary_labor(self, agents, consider_amt, nonag_lbr):
        '''
        allocate the available salary jobs between the agents that want them
        assume that agents are considered successively
        (independent of how much work they need)
        with the order randomized at each time step
        '''
        rndm_num = np.random.randint(1e6)
        # subtract off jobs that have already been allocated to continuing workers
        num_jobs_avail = self.salary_job_avail_total - np.sum(nonag_lbr)

        if np.sum(consider_amt) <= num_jobs_avail:
            rtn = consider_amt
        else:
            # the order in which agents are considered is randomized each time step
            order_consider = np.random.choice(agents.N, size=agents.N, replace=False)
            # line up the agents
            cumsums = np.cumsum(consider_amt[order_consider])
            # find the last agent that gets any (first index that it is >= the limit)
            crit_ix = np.argmax(cumsums >= num_jobs_avail)
            # if the last agent has overshot, only allocate up to the limit
            cumsums[crit_ix] = min(num_jobs_avail, cumsums[crit_ix])
            # take the differences to give agent-level allocations
            allocs = cumsums - np.concatenate([[0], cumsums[:-1]])
            allocs[(crit_ix+1):] = 0
            # convert back to the regular agent order
            agent_allocs = np.full(agents.N, np.nan)
            agent_allocs[order_consider] = allocs
            rtn = agent_allocs

        np.random.seed(rndm_num) # set seed to control stochasticity
        return rtn

    def allocate_wage_labor(self, agents, consider_amt):
        '''
        allocate the available wage jobs between the agents that want them
        assume that each job is randomly allocated each day
        NOTE: this involves a variable number of calls to the random generator
        so control stochasticity
        '''
        rndm_num = np.random.randint(1e6)
        tot_consider = consider_amt.sum()
        
        if tot_consider == 0:
            rtn = consider_amt
        else:
            self.p_wage_labor[agents.t[0]] = min(1, self.wage_job_avail_total/tot_consider)
            if self.p_wage_labor[agents.t[0]] == 1:
                rtn = consider_amt
            else:
                ntries = (consider_amt/self.wage_job_increment).astype(int)
                # just honor the regional probability
                rtn = np.random.binomial(n=ntries,p=self.p_wage_labor[agents.t[0]]) * self.wage_job_increment

        np.random.seed(rndm_num) # set seed to control stochasticity
        return rtn