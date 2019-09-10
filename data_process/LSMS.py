'''
processing of LSMS data for model inputs
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import code
import scipy.stats as stat
import copy

def main():
    # expenditures()
    # fertilizer()
    area()

def area():
    '''
    fit distribution to area cultivated in 2015
    '''
    d = pd.read_csv('../data/LSMS/ethiopia_2015_formatted.csv', index_col=0)
    a = np.array(d['area'])
    fig, ax = plt.subplots()
    perc = 95
    qt = np.percentile(a, q=perc)
    a2 = a[a<qt]

    # plot simulated values
    xs = np.linspace(0, a2.max(), 1000)
    s, loc, scale = stat.lognorm.fit(a2)
    ys = stat.lognorm.pdf(xs, s, loc, scale) 

    ax.hist(a2, bins=20, label='LSMS', density=True)
    ax.plot(xs, ys, label='fit')
    ax.legend()
    ax.set_xlabel('Area (ha)')
    ax.text(ax.get_xlim()[1]*0.5, ax.get_ylim()[1]*0.5, 
        's={}\nloc={}\nscale={}\n{}%={}ha'.format(np.round(s,3), np.round(loc,3), np.round(scale,3), perc, np.round(qt, 2)))
    fig.savefig('../data/LSMS/areas.png')
    # code.interact(local=dict(globals(), **locals()))

def fertilizer():
    '''
    look at inorganic fertilizer: how much do people apply, what is the cost
    '''
    d = pd.read_csv('../data/LSMS/Wave_3/ETH_2015_ESS_v01_M_CSV/Post-Planting/sect3_pp_w3.csv')
    fert = d[['household_id','household_id2','field_id','parcel_id']].copy()
    fert['urea'] = d['pp_s3q16c']
    fert['urea_cost'] = d['pp_s3q16d']
    fert['DAP'] = d['pp_s3q19c']
    fert['DAP_cost'] = d['pp_s3q19d']
    fert['NPS'] = d['pp_s3q20a_4']
    fert['NPS_cost'] = d['pp_s3q20a_5']
    fert['other'] = d['pp_s3q20b_1']
    fert['other_cost'] = d['pp_s3q20c']
    # totals
    ferts = ['urea','DAP','NPS','other']
    fert['total'] = fert[ferts].sum(axis=1, skipna=True)
    fert.loc[fert['total']==0, 'total'] = np.nan
    fert['total_cost'] = fert[['urea_cost','DAP_cost','NPS_cost','other_cost']].sum(axis=1, skipna=True)
    fert.loc[fert['total_cost']==0, 'total_cost'] = np.nan

    ## get land area
    fert['area_ha'] = d['pp_s3q05_a'] / 10000

    # create plots
    fig = plt.figure(figsize=(15,4))
    # 1. boxplots of fertilizer types
    ax1 = fig.add_subplot(131)
    plt_data = fert[ferts + ['total']]
    plt_data.boxplot(ax=ax1)
    ax1.set_ylabel('kg')
    ax1.set_title('Type of fertilizer')

    # total use per ha
    tmp = np.array(fert['total'])
    tmp = tmp / np.array(fert['area_ha'])
    tmp = tmp[~np.isnan(tmp)]
    pp = np.nanpercentile(tmp, 95)
    tmp2 = copy.copy(tmp)
    tmp2[tmp2>pp] = pp
    ax2 = fig.add_subplot(132)
    ax2.hist(tmp2, bins=20)
    ax2.set_xlabel('Amount applied (kg/ha)')
    ax2.set_title('Total application rate')
    ax2.text(0.5*pp, ax2.get_ylim()[1]*0.5, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(np.nanmean(tmp), 1), 
        np.round(np.nanpercentile(tmp, 50), 1), 
        np.round(np.nanpercentile(tmp, 90), 1), 
        np.round(np.nanpercentile(tmp, 95), 1), 
        np.round(np.nanpercentile(tmp, 99), 1)))
    ax2.grid(False)

    # costs
    tmp = np.array(fert['total_cost'])
    tmp = tmp / np.array(fert['total'])
    tmp = tmp[~np.isnan(tmp)]
    pp = np.nanpercentile(tmp, 95)
    tmp2 = copy.copy(tmp)
    tmp2[tmp2>pp] = pp
    ax3 = fig.add_subplot(133)
    ax3.hist(tmp2, bins=20)
    ax3.set_xlabel('Total cost (birr/kg)')
    ax3.set_title('Costs')
    ax3.text(0.2*pp, ax3.get_ylim()[1]*0.5, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(np.nanmean(tmp), 1), 
        np.round(np.nanpercentile(tmp, 50), 1), 
        np.round(np.nanpercentile(tmp, 90), 1), 
        np.round(np.nanpercentile(tmp, 95), 1), 
        np.round(np.nanpercentile(tmp, 99), 1)))
    ax3.grid(False)

    fig.savefig('../data/LSMS/fertilizer.png')
    # code.interact(local=dict(globals(), **locals()))

def expenditures():
    ##### Expenditures #####
    d = pd.read_csv('../data/LSMS/Wave_3/ETH_2015_ESS_v01_M_CSV/Consumption Aggregate/cons_agg_w3.csv')
    cons = d['total_cons_ann']
    cons = cons.loc[~cons.isna() & (d['rural'] == 1)]
    upr = np.percentile(cons, 99)
    cons[cons>upr] = upr

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.hist(cons, alpha=0.6, bins=20)
    ax.set_xlabel('Consumption (birr)')
    ax.set_title('Consumption')
    # calculate values
    ax.text(0.5*upr, ax.get_ylim()[1]*0.5, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(cons.mean(), 1), 
        np.round(np.percentile(cons, 50), 1), 
        np.round(np.percentile(cons, 90), 1), 
        np.round(np.percentile(cons, 95), 1), 
        np.round(np.percentile(cons, 99), 1)))
    ax.grid(False)
    fig.savefig('../data/LSMS/consumption_rural.png')

if __name__ == '__main__':
    main()