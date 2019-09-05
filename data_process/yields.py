'''
Explore maize yield data
using CSA's AgSS data and LSMS 2015 data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import code

from plot import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

def main():
    fdir = '../data/yields/'
    fig = plt.figure(figsize=(12,5))
    ##### CSA #####
    d = pd.read_csv(fdir + 'data_C_maize.csv', index_col=0)
    d.index = np.arange(d.shape[0])
    ax = fig.add_subplot(121)
    ax.hist(d.yldp, alpha=0.6, bins=20)
    ax.set_xlabel('Yield (quint/ha)')
    ax.set_title('AgSS maize yields')
    # calculate values
    ax.text(50, ax.get_ylim()[1]*0.5, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(d.yldp.mean(), 1), 
        np.round(np.percentile(d.yldp, 50), 1), 
        np.round(np.percentile(d.yldp, 90), 1), 
        np.round(np.percentile(d.yldp, 95), 1), 
        np.round(np.percentile(d.yldp, 99), 1)))

    ##### LSMS #####
    d = pd.read_csv(fdir + 'lsms_data_with_raw_climate.csv')
    d['yldp'] = d.harv_kg / d.area_m2_plant / (100 / 10000) # quintals/ha
    d.loc[d.yldp > 100, 'yldp'] = 100 # keep consistent with CSA data
    d = d.loc[d.crop_name == 'MAIZE']
    d = d.loc[~d.yldp.isna()]
    ax2 = fig.add_subplot(122)
    ax2.hist(d.yldp, alpha=0.6, bins=20)
    ax2.set_xlabel('Yield (quint/ha)')
    ax2.set_title('LSMS maize yields')
    # calculate values
    ax2.text(50, ax2.get_ylim()[1]*0.5, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(d.yldp.mean(), 1), 
        np.round(np.percentile(d.yldp, 50), 1), 
        np.round(np.percentile(d.yldp, 90), 1), 
        np.round(np.percentile(d.yldp, 95), 1), 
        np.round(np.percentile(d.yldp, 99), 1)))

    fig.tight_layout()
    fig.savefig(fdir + 'yield_distribution.png')
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()