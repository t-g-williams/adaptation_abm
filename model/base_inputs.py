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
    return d

def model():
    d = {
        'n_agents' : 100,
        'T' : 10, # number of years to simulate
        'exp_name' : 'test',
        'seed' : 0,
        'sim_id' : 0,
        'rep_id' : 0,
        'adaptation_option' : 'none' # set to a random string for baseline
    }
    return d

def adaptation():
    d = {
        # 'fertilizer_fixed' : {
        #     'application_rate' : 147.2, # kg/ha. median from LSMS (ALL fertilizer. i assume this equals Nitrogen)
        #     },
        'insurance' : {
            'climate_percentile' : 0.1,
            'payout_magnitude' : 1, # relative to the expected yield (assuming perfect soil quality). if =1.5, then payout = 1.5*expected_yield
            },
        'cover_crop' : {
            'N_fixation' : 80, # 80kg/ha representative of values reported in the literature. wittwer2017, buechi, couedel2018
            'cost_factor' : 1, # assume the cost is the same as the annual cost of insurance multiplied by this factor
        }
    }
    return d

def agents():
    d = {
        # adaptation / decision-making
        'adap_type' : 'affording', # coping, switching, or affording

        # plot ownership
        'n_plots_init' : [8, 11, 14], # uniformly sample from each

        ##### cash + wealth #####
        # initial (normal distribution)
        'wealth_init_mean' : 15000,
        'wealth_init_sd' : 0,
        'max_neg_wealth' : -100000, # birr. just for plotting. these ppl cant recover anyway
        # requirements
        'cash_req_mean' : 17261, # 17261 birr/yr. median value from 2015 LSMS
        'cash_req_sd' : 0,
        # market prices
        'crop_sell_price' : 2.17, # 2.17 birr/kg. mean 2015 maize price (FAO)
        'fertilizer_cost' : 13.2, # 13.2 birr/kg. median from 2015 LSMS
        'livestock_cost' : 3000, # birr/head. Ethiopia CSA data 2015
    }
    return d

def land():
    d = {
        ##### SOM #####
        # initial vals
        'organic_N_min_init' : 4000, # kgN/ha. similar to initial value in Li2004
        'organic_N_max_init' : 4000,
        # soil model
        'max_organic_N' : 8000, # kgN/ha. arbitrary (set in relation to the initial value)
        'fast_mineralization_rate' : 0.6, # what fraction of applied organic matter mineralizes straight away
        'slow_mineralization_rate' : 0.02, # 0.02 rate of mineralization from organic->inorganic (assume linear decay). taken from schmidt2011 -- 50year turnover time of bulk SOM
        'loss_max' : 0.5, # 0.5 inorganic loss fraction with no SOM. Di2002 data had ~50% maximum leaching rates of N. giller1997 says up to 50% in high-rainfall environments
        'loss_min' : 0.05, # 0.05 inorganic loss fraction with maximum SOM. Di2002 had ~5% minimum leaching.
        
        ##### yield #####
        'area' : 0.13, # 0.13 ha. mean in LSMS 2015
        'max_yield' : 6590, # 6590 kg/ha. maximum, unconstrained yield. 95%ile for Ethiopia-wide LSMS (all 3 years) maize yields
        'rain_crit' : 0.8, # value at which rainfall starts to be limiting. 0.8 in CENTURY
        'rain_cropfail_high_SOM' : 0, # rainfall value at which crop yields are 0 with highest SOM. arbitrary
        'rain_cropfail_low_SOM' : 0.1, # rainfall value at which crop yields are 0 with lowest SOM. arbitrary
        'random_effect_sd' : 0.3, # std dev of yield multiplier effect (normal distribution, mu=1)
        'crop_CN_conversion' : 50, # 50 from Century model curves (middle of the y axis) -- pretty arbitrary. represents C:N ratio kind of
        'residue_CN_conversion' : 200, # 1/4 of the crop. elias1998

        ##### livestock #####
        'residue_loss_factor' : 0.9, #  90% conversion efficiency  
        'residue_multiplier' : 2, # 2x crop yield->maize residue conversion factor (FAO1987), 
        'wealth_N_conversion' : 0.026, # 0.026 kgN/yr per birr. a proxy for livestock manure. derived as 3000birr/head and using values from Newcombe1987. nitrogen %age in manure also similar in Lupwayi2000
        'livestock_frac_crops' : 0.5, # fraction of livestock feed that comes from crops (in an ~average year). this influences the nitrogen input to farmland and the maximum herdsize attainable
        'livestock_residue_factor' : 2280, # kg dry matter / TLU / year.(Amsalu2014)
    }
    return d

def climate():
    d = {
        # annual climate measure -- assume normal distribution (truncated to [0,1])
        'rain_mu' : 0.6, # 0.5 approximately fits country-wide CYF distribution for maize (BUT this variable is rain not CYF)
        'rain_sd' : 0.2,
    }
    return d