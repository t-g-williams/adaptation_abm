'''
All model input parameters
'''
def compile():
    d = {}
    d['model'] = model()
    d['agents'] = agents()
    d['land'] = land()
    d['climate'] = climate()
    d['adaptation'] = adaptation()
    d['rangeland'] = rangeland()
    d['livestock'] = livestock()
    return d

def model():
    d = {
        'n_agents' : 200,
        'T' : 100, # number of years to simulate
        'exp_name' : 'test',
        'seed' : 0,
        'sim_id' : 0,
        'rep_id' : 0,
        'adaptation_option' : 'insurance', # set to "none" for baseline
        'shock' : False,
    }
    return d

def adaptation():
    d = {
        # 'fertilizer_fixed' : {
        #     'application_rate' : 147.2, # kg/ha. median from LSMS (ALL fertilizer. i assume this equals Nitrogen)
        #     },
        'burnin_period' : 10, # years before adaptation options come into effect
        'insurance' : {
            'climate_percentile' : 0.1,
            'payout_magnitude' : 1, # relative to the expected yield (assuming perfect soil quality). if =1.5, then payout = 1.5*expected_yield
            'cost_factor' : 1, # multiplier on insurance cost
            },
        'cover_crop' : {
            'N_fixation_min' : 80, # (with full organic matter) 80kg/ha representative of values reported in the literature. wittwer2017, buechi, couedel2018
            'N_fixation_max' : 80, # (with no organic matter) could do 50 and 200 if making them different
            'cost_factor' : 1, # assume the cost is the same as the annual cost of insurance multiplied by this factor
        }
    }
    return d

def agents():
    d = {
        # binary switches
        'savings_acct' : True, # if false, agents can't carry over extra money between years

        # adaptation / decision-making
        'adap_type' : 'always', # coping, switching, affording, or always

        # plot ownership
        'land_area_init' : [1, 1.5, 2], # ha. uniformly sample from each
        'land_area_multiplier' : 1, # for sensitivity analysis

        ##### cash + wealth #####
        # initial cash savings (normal distribution)
        'savings_init_mean' : 15000, # birr
        'savings_init_sd' : 0,
        # 'max_neg_wealth' : 0, # birr. just for plotting. these ppl cant recover anyway
        # initial livestock (constant amount)
        'livestock_init_mean' : 0,
        # requirements
        'cash_req_mean' : 17261, # 17261 birr/yr. median value from 2015 LSMS
        'cash_req_sd' : 0,
        # market prices
        'crop_sell_price' : 2.17, # 2.17 birr/kg. mean 2015 maize price (FAO)
        'fertilizer_cost' : 13.2, # 13.2 birr/kg. median from 2015 LSMS
    }
    return d

def land():
    d = {
        ##### SOM #####
        # initial vals
        'organic_N_min_init' : 4000, # kgN/ha. similar to initial value in Li2004
        'organic_N_max_init' : 4000, # NOTE: CURRENTLY THE MODEL SETS THIS TO BE THE SAME AS MIN
        # soil model
        'max_organic_N' : 8000, # kgN/ha. arbitrary (set in relation to the initial value)
        'fast_mineralization_rate' : 0.6, # what fraction of applied organic matter mineralizes straight away
        'slow_mineralization_rate' : 0.02, # 0.02 rate of mineralization from organic->inorganic (assume linear decay). taken from schmidt2011 -- 50year turnover time of bulk SOM
        'loss_max' : 0.5, # 0.5 inorganic loss fraction with no SOM. Di2002 data had ~50% maximum leaching rates of N. giller1997 says up to 50% in high-rainfall environments
        'loss_min' : 0.05, # 0.05 inorganic loss fraction with maximum SOM. Di2002 had ~5% minimum leaching.
        
        ##### yield #####
        'max_yield' : 6590, # 6590 kg/ha. maximum, unconstrained yield. 95%ile for Ethiopia-wide LSMS (all 3 years) maize yields
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
        'rain_mu' : 0.5, # 0.5 approximately fits country-wide CYF distribution for maize (BUT this variable is rain not CYF)
        'rain_sd' : 0.2,

        'shock_years' : [30], # starting at 0 (pythonic)
        'shock_rain' : 0.1, # the rain value in the simulated shock
    }
    return d

def rangeland():
    d = {
        # binary switches
        'rangeland_dynamics' : True, # if false, just use the livestock "frac_crops" parameter

        # rangeland size relative to farmland
        'range_farm_ratio' : 0.5, # eg 0.5 means rangeland is 0.5x the size of the total farmland
        # initial conditions
        'R0_frac' : 0.3,
        # growth parameters
        'R_biomass_growth' : 0.8, # w for gunnar
        'R_mortality' : 0.1, # m_r for gunnar (reserve biomass mortality rate)
        'G_mortality' : 0, # m_g for gunnar (green biomass mortality rate)
        'gr1' : 0.5, # grazing harshness (what fraction of the grazed biomass contributes to R growth) (CHECK??)
        'gr2' : 0.1, # fraction of reserve biomass that can be consumed
        # constants
        'rain_use_eff' : 1, # rue for gunnar (CALIBRATION RQD)
        'G_R_ratio' : 0.5, # lambda for gunnar (limit of ratio of green to reserve biomass (i.e. G:R can't be larger than this))
        'R_max' : 1500, # kg/ha
    }
    return d

def livestock():
    d = {
        'N_production' : 78.3, # kg N/year/cattle. see CC_Ins paper for derivation
        'frac_crops' : 0.5, # IF rangeland_dynamics==True, this parameter is unnecessary. fraction of livestock feed that comes from crops (in an ~average year). this influences the nitrogen input to farmland and the maximum herdsize attainable
        'income' : 0, # 125, # birr/year/head represents the value of milk production: taken directly from Redda2002 -- 240-480birr/year with local cow. assume 350 and 50% are female --> 125 birr/year/animal
        'consumption' : 2280, # kg/annum (640 in gunnar. i derived 2280 for cc_ins paper: kg dry matter / TLU / year.(Amsalu2014))
        'birth_rate' : 0.8, # livestock birth rate
        'cost' : 3000, # birr/head. Ethiopia CSA data 2015
    }
    return d