'''
processing of LSMS data for model inputs
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import code

def main():
    ##### Expenditures #####
    d = pd.read_csv('../data/yields/wereda_with_cyf.csv')

    yrs = np.arange(17)
    all_d = []
    for yr in yrs:
        all_d.append(np.array(d['maize{}'.format(yr)]))
    all_d = np.array(all_d).flatten()

    # code.interact(local=dict(globals(), **locals()))
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.hist(all_d, alpha=0.6, bins=20)
    ax.set_xlabel('CYF')
    ax.set_title('Climate yield factor (maize)')
    # calculate values
    ax.text(0.8, ax.get_ylim()[1]*0.8, 'mean = {}\nmedian = {}\n90% = {}\n95% = {}\n99% = {}'.format(
        np.round(all_d.mean(), 2), 
        np.round(np.percentile(all_d, 50), 2), 
        np.round(np.percentile(all_d, 90), 2), 
        np.round(np.percentile(all_d, 95), 2), 
        np.round(np.percentile(all_d, 99), 2)))
    ax.grid(False)
    fig.savefig('../data/yields/CYFs_maize.png')

if __name__ == '__main__':
    main()