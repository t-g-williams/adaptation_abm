'''
All model input parameters
'''
from collections import OrderedDict

def compile():
    d = {}
    d['model'] = model()
    d['agents'] = agents()
    d['land'] = land()
    d['climate'] = climate()
    d['adaptation'] = adaptation()
    d['rangeland'] = rangeland()
    d['livestock'] = livestock()
    d['LSLA'] = LSLA()
    d['market'] = market()
    d['beliefs'] = beliefs(d['climate'], d['market'])
    d['decisions'] = decisions()
    return d

def model():
    d = {
        'n_agents' : 200,
        'T' : 100, # number of years to simulate
        'exp_name' : 'test',
        'seed' : 0,
        'sim_id' : 0,
        'rep_id' : 0,
        'adaptation_option' : 'none', # set to "none" for baseline
        'shock' : False,
        'lsla_simulation' : False,
    }
    return d

def adaptation():
    d = {
        # 'fertilizer_fixed' : {
        #     'application_rate' : 147.2, # kg/ha. median from LSMS (ALL fertilizer. i assume this equals Nitrogen)
        #     },
        'burnin_period' : 15, # years before adaptation options come into effect
        'adap_type' : 'always', # coping, switching, affording, or always
        'insurance' : {
            'climate_percentile' : 0.1,
            'payout_magnitude' : 1, # relative to the expected yield (assuming perfect soil quality). if =1.5, then payout = 1.5*expected_yield
            'cost_factor' : 1, # multiplier on insurance cost
            },
        'diversify' : {
            'N_fixation_min' : 80, # (with full organic matter) 80kg/ha representative of values reported in the literature. wittwer2017, buechi, couedel2018
            'N_fixation_max' : 80, # (with no organic matter) could do 50 and 200 if making them different
            'cost_factor_ag' : 0.1, # relative to normal agricultural cost per ha
            'labor_req' : 0.1, # ppl per ha
        },
        'fertilizer' : {
            'application_rate' : 100, # kgN/ha
        },
        'conservation' : {
            'organic_N_added' : 80, # kgN/ha on overall farm. independent of area_req
            'area_req' : 0.2, # fraction of crop area used for conservation
        },
        'outgrower' : {
            'active' : True,
            'land_rqmt_type' : 'ha', # {fraction or ha}
            'land_rqmt_amt' : 1,
            'fixed_price' : True,
        },
    }
    return d

def agents():
    d = {
        # binary switches
        'savings_acct' : True, # if false, agents can't carry over extra money between years

        #### DEMOGRAPHICS ####
        # data import
        'read_from_file' : False,
        'props_from_file' : ['hh_size','land_area_init'], # the variable names in the file should be equivalent to the names in this file
        'file_name' : '../inputs/lsla_for_abm.csv',
        'data_filter' : 'site=="OR1"',
        # if read_from_file == False:
        'land_area_init' : [0.25,0.5,1, 1.5, 2, 2.5], # ha. uniformly sample from each
        'land_area_multiplier' : 1, # for sensitivity analysis
        'hh_size_init' : 6,
        'types' : {'land-poor' : {'land_area' : 1, 'hh_size' : 3},
                    'middle' : {'land_area' : 1.5, 'hh_size' : 3},
                    'land-rich' : {'land_area' : 2, 'hh_size' : 3}},

        # living costs
        'living_cost_pp' : 1000, # birr/yr. "desired living costs" not including food consumption. 17261 median (hh) value from 2015 LSMS
        'living_cost_min_frac' : 0.5, # fraction of the living costs that must be spent (ie they will sell livestock to do this)
        'food_rqmt_pp' : 18*12, # kg/yr. from resilience paper

        # decision-making
        'n_yr_smooth' : 3, # number of smoothing years for livestock management decisions (fodder availability assumption)

        ##### cash + wealth #####
        # initial cash savings (normal distribution)
        'savings_init_mean' : 0,# 15000, # birr
        'savings_init_sd' : 0,
        'livestock_init' : 2,# 0, (homogeneous for all agents)
        
        ##### socio-environmental condns #####
        'ag_labor_rqmt' : {'subs' : 1.5, 'mkt' : 1.5}, # ppl/ha
        'ls_labor_rqmt' : 0.2, # ppl/head
    }
    return d

def beliefs(climate, market):
    d = {
        'quantities' : ['rain','price_subs','price_mkt'],
        'n0' : [1,1,1], # prior strength on the mean
        'alpha0' : [1,1,1], # prior strength on the variance
        'beta0' : [0.25,0.5,0.5], # E[variance0]
        'mu0' : [climate['rain_mu'],market['crop_sell_params']['subs'][0],market['crop_sell_params']['mkt'][0]], # E[mu0]. the expected value is accurate. this makes the beliefs not affect the results too much
    }
    return d

def decisions():
    d = {
        'framework' : 'util_max', # util_max or imposed
        'actions' : OrderedDict({'conservation' : [False,True], 'fertilizer' : [False,True], 'outgrower' : [False,True]}), # it should be such that the first option to be created is the "baseline"
        'imposed_action' : {'conservation' : True, 'fertilizer' : True, 'outgrower' : False}, # only when t==0 or framework==imposed
        'risk_aversion' : True,
        'risk_tolerance_mu' : 3000, # mu for a normal distribution (high = more tolerant to risk)
        'risk_tolerance_cov' : 0.5, # coefficient of variation - i.e., sigma/mu. 0.5 means 50%
        'horizon' : 5, # decision horizon
        'discount_rate_mu' : 0.586, # fraction. drawn from holden et al 1998 (58.6 = 100/(1+r) --> r = 0.706)
        'discount_rate_sigma' : 0.1,
        'nsim_utility' : 10, # number of sims from beliefs when calculating utility
    }
    return d

def market():
    d = {
    # binary switches
    'nonfarm_labor' : False, # if false, exclude wage and salary labor processes

    'crop_sell_params' : {'subs' : [2.17,0.25,0.5], 'mkt' : [2.17,0.5,0.5]}, # [x,x,x]=[mean,sd,rho] 2.17 birr/kg. mean 2015 maize price (FAO)
    'farm_cost_baseline' : 100, # birr/ha. arbitrary
    'farm_cost_factors' : {'subs' : 1, 'mkt' : 1.5}, # multiplier on baseline. arbitrary
    'fertilizer_cost' : 13.2, # 13.2 birr/kg. median from 2015 LSMS
    'labor_salary' : 70*365*5/7, # birr/person/year: 70 birr/day * 5 days per week all year
    'labor_wage' : 70*365*5/7, # birr/person/year: 70 birr/day * 5 days per week all year
    'salary_jobs_availability' : 0.1, # full-time salary jobs per agent (used as livelihood strategy)
    'wage_jobs_availability' : 0.1, # single-day wage jobs per agent (used as coping)
    'salary_job_increment' : 0.5, # minimum fraction of a person's time that can be devoted to non-farm SALARY labor (e.g., if 1 then ONLY full-time jobs are available)
    'wage_job_increment' : 0.005, # minimum fraction of a person's time that can be devoted to non-farm WAGE labor (e.g., if 0.005 then 1/200th of year)
    'livestock_cost' : 3000, # birr/head. Ethiopia CSA data 2015
    }
    return d

def land():
    d = {
        'plot_size' : 0.25, # ha. resolution of the landscape. NOTE: not fully integrated into the model
        ##### SOM #####
        # initial vals
        'organic_N_min_init' : 4000, # kgN/ha. similar to initial value in Li2004
        'organic_N_max_init' : 4000, # NOTE: CURRENTLY THE MODEL SETS THIS TO BE THE SAME AS MIN
        # soil model
        'max_organic_N' : 6000, # kgN/ha. arbitrary (set in relation to the initial value)
        'fast_mineralization_rate' : 0.5, # 0.6, # what fraction of applied organic matter mineralizes straight away. 0.1 in POM for CC-ins
        'slow_mineralization_rate' : 0.02, # 0.02 rate of mineralization from organic->inorganic (assume linear decay). taken from schmidt2011 -- 50year turnover time of bulk SOM
        'loss_max' : 0.5, # 0.5 inorganic loss fraction with no SOM. Di2002 data had ~50% maximum leaching rates of N. giller1997 says up to 50% in high-rainfall environments
        'loss_min' : 0.05, # 0.05 inorganic loss fraction with maximum SOM. Di2002 had ~5% minimum leaching.
        
        ##### ag practices #####
        'fallow_frac' : 0.3, # fraction fallow under traditional settings. calibrate to get stable SOM
        'fallow_N_add' : 60/0.3, # FOR NOW TRY TO GET NEUTRAL SOIL N BALANCE. this is unrealistic # 40, # kg N/ha. lower limit from N-fixing legumes https://www.tandfonline.com/doi/pdf/10.1080/01904160009382074

        ##### yield #####
        'ag_types' : ['subs','mkt'],
        'max_yield' : {'subs' : 6590, 'mkt' : 6590}, # 6590 kg/ha. maximum, unconstrained yield. 95%ile for Ethiopia-wide LSMS (all 3 years) maize yields
        'rain_crit' : 0.8, # value at which rainfall starts to be limiting. 0.8 in CENTURY
        'rain_cropfail_high_SOM' : 0, # rainfall value at which crop yields are 0 with highest SOM. arbitrary
        'rain_cropfail_low_SOM' : 0.1, # rainfall value at which crop yields are 0 with lowest SOM. arbitrary
        'random_effect_sd' : 0.3, # std dev of yield multiplier effect (normal distribution, mu=1)
        'crop_CN_conversion' : 50, # 50 from Century model curves (middle of the y axis) -- pretty arbitrary. represents C:N ratio kind of
        'residue_CN_conversion' : 200, # 1/4 of the crop. elias1998

        ##### residues #####
        'residue_loss_factor' : 0.9, #  90% conversion efficiency  
        'residue_multiplier' : 2, # 2x crop yield->maize residue conversion factor (FAO1987), 
        # 'wealth_N_conversion' : 0.026, # 0.026 kgN/yr per birr. a proxy for livestock manure. derived as 3000birr/head and using values from Newcombe1987. nitrogen %age in manure also similar in Lupwayi2000
    }
    return d

def climate():
    d = {
        # annual climate measure -- assume normal distribution (truncated to [0,1])
        'rain_mu' : 0.6, # 0.5 approximately fits country-wide CYF distribution for maize (BUT this variable is rain not CYF)
        'rain_sd' : 0.2,

        'shock_years' : [30], # starting at 0 (pythonic)
        'shock_rain' : 0.1, # the rain value in the simulated shock
    }
    return d

def rangeland():
    d = {
        # binary switches
        'rangeland_dynamics' : False, # if false, just use the livestock "frac_crops" parameter
        'integer_consumption' : False, # if true, only an integer number of livestock can be grazed on residues/rangeland

        # the following are only used if rangeland_dynamics == True
        # rangeland size relative to farmland
        'range_farm_ratio' : 0.5, # eg 0.5 means rangeland is 0.5x the size of the total farmland
        # initial conditions
        'R0_frac' : 0.5,
        # growth parameters
        'R_biomass_growth' : 0.8, # w for gunnar (reserve biomass growth rate)
        'R_mortality' : 0.1, # m_r for gunnar (reserve biomass mortality rate)
        'G_mortality' : 0, # m_g for gunnar (green biomass mortality rate)
        'gr1' : 0.5, # grazing harshness (what fraction of the grazed biomass contributes to R growth) (CHECK??)
        'gr2' : 0.1, # fraction of reserve biomass that can be consumed
        # constants
        'rain_use_eff' : 1, # rue for gunnar (CALIBRATION RQD)
        'G_R_ratio' : 0.5, # lambda for gunnar (limit of ratio of green to reserve biomass (i.e. G:R can't be larger than this))
        'R_max' : 5000, # kg/ha (gunnar had 1500). this constrains NPP. max(NPP) = R_max. amsalu2014: could be from 1-7t DM/ha. use 5?
        'SOM' : 3000, # if agents are displaced. this is constant and exogenous
    }
    return d

def livestock():
    d = {
        # if rangeland_dynamics == False:
        'frac_crops' : 0.5, # IF rangeland_dynamics==True, this parameter is unnecessary. fraction of livestock feed that comes from crops (in an ~average year). this influences the nitrogen input to farmland and the maximum herdsize attainable
        
        # always used 
        'frac_N_import' : 0, # amount of N from rangeland that can be imported to the farm. controls livestock/SOM feedback
        'N_production' : 78.3, # kg N/year/cattle. see CC_Ins paper for derivation
        'income' : 125, # birr/year/head represents the value of milk production: taken directly from Redda2002 -- 240-480birr/year with local cow. assume 350 and 50% are female --> 125 birr/year/animal
        'consumption' : 2280, # kg/annum (640 in gunnar. i derived 2280 for cc_ins paper: kg dry matter / TLU / year.(Amsalu2014)). also compare to 2700kgDM in (NOTE: ITS AUSTRALIA HERE NOT BORANA) desta1999 for 400kg animal
        'birth_rate' : 0, # use no birth rate -- livestock aren't for breeding. livestock birth rate (probability) (gunnar has 0.8). see also angassa2007 (~0.5)
    }
    return d

def LSLA():
    d = {
        ## for all simulations ##
        'tstart' : 5, # e.g. 5 means start of 6th year of simulation
        'size' : 0.5, # ha per ha of farmland. note: the maximum feasible value for this is the same as the range:farm ratio
        'outgrower' : False, # has implications for other params
        
        ## outgrower params ##
        'fert_amt' : 50, # kg/ha -- constant over time. set to 0 for no fertilizer
        'irrig' : True,
        'no_fallow' : True, # if true, agents w outgrower don't fallow their land

        ## non-outgrower params ##
        'employment' : 0.153, # jobs/ha taken. scheidel2013 has values. 153 jobs / 1000 ha
        'LUC' : 'farm', # 'farm' or 'commons'' or ?'none'?
        'encroachment' : 'farm', # where do displaced HHs encroach on? 'farm' or 'commons'
        'frac_retain' : 0.5, # fraction of land that was originally taken that HHs retain (on average)
        'land_distribution_type' : 'amt_lost', # amt_lost: proportional to the amt of land lost, 'equal_hh' : equal per hh, "equal_pp" : equal per person
        'land_taking_type' : 'equalizing', # random or equalizing
    }
    return d