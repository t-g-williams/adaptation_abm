import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
import copy
import sys
from . import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

def main(results, shock_mags, shock_times, T_res, exp_name):
    savedir = '../outputs/{}/shocks/'.format(exp_name)

    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns
    
    # create a separate plot for each shock magnitude
    for m, mag in enumerate(shock_mags):

        fig = plt.figure(figsize=(6*len(T_res),4*len(land_area)))
        axs = []
        lims = {}
        for m, mi in enumerate(T_res):
            lims[m] = [99999,-99999]
        ts = []

        for n, nplot in enumerate(land_area):
            for ti, T in enumerate(T_res):
                # create the axis
                ax = fig.add_subplot(len(T_res), len(land_area), len(land_area)*ti+(n+1))
                axs.append(ax)

                for sc in adap_scenarios:
                    # extract the plot data
                    d = results[sc][nplot][mag][T]
                    ax.plot(d.index, d, label=sc, marker='o')

                # formatting
                if ti==0: # top row
                    ax.set_title('{} ha'.format(nplot))
                    ax.legend(loc='upper left')
                if n==0: # left column
                    ax.set_ylabel('Resilience T = {}'.format(T))
                else:
                    ax.set_yticklabels([])
                if ti==(len(T_res)-1): # bottom row
                    ax.set_xlabel('Time of shock (years)')
                else:
                    ax.set_xticklabels([])

                # get limits
                lims[ti][0] = min(lims[ti][0], ax.get_ylim()[0])
                lims[ti][1] = max(lims[ti][1], ax.get_ylim()[1])
                ts.append(ti)
        
        for a, ax in enumerate(axs):
            ax.grid(False)
            ax.axhline(y=0, color='k', ls=':')
            ax.set_ylim(lims[ts[a]])

        axs[0].text(axs[0].get_xlim()[1], axs[0].get_ylim()[1], 'Larger damage', ha='right', va='top')
        axs[0].text(axs[0].get_xlim()[1], axs[0].get_ylim()[0], 'Smaller damage', ha='right', va='bottom')


        fig.tight_layout()
        fig.savefig(savedir + 'shock_effects_{}.png'.format(str(mag).replace('.','_')))
