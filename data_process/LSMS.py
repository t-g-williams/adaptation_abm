'''
processing of LSMS data for model inputs
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
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