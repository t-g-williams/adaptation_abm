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

def main(results, shock_mags, shock_times, exp_name):
    savedir = '../outputs/{}/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)


    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns
    fig = plt.figure(figsize=(6*len(shock_mags),4*len(land_area)))
    axs = []
    lims = {}
    for m, mi in enumerate(shock_mags):
        lims[m] = [99999,-99999]
    ms = []

    for n, nplot in enumerate(land_area):
        for m, mag in enumerate(shock_mags):
            # create the axis
            ax = fig.add_subplot(len(shock_mags), len(land_area), len(shock_mags)*m+(n+1))
            axs.append(ax)

            for sc in adap_scenarios:
                # extract the plot data
                d = results[sc][nplot][mag]
                ax.plot(d.index, d, label=sc, marker='o')

            # formatting
            if m==0: # top row
                ax.set_title('{} ha'.format(nplot))
                ax.legend(loc='upper left')
            if n==0: # left column
                ax.set_ylabel('Magnitude = {}'.format(mag))
            else:
                ax.set_yticklabels([])
            if m==(len(shock_mags)-1): # bottom row
                ax.set_xlabel('Time of shock (years)')
            else:
                ax.set_xticklabels([])

            # get limits
            lims[m][0] = min(lims[m][0], ax.get_ylim()[0])
            lims[m][1] = max(lims[m][1], ax.get_ylim()[1])
            ms.append(m)
    
    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.axhline(y=0, color='k', ls=':')
        ax.set_ylim(lims[ms[a]])

    axs[0].text(axs[0].get_xlim()[1], axs[0].get_ylim()[1], 'Larger damage', ha='right', va='top')
    axs[0].text(axs[0].get_xlim()[1], axs[0].get_ylim()[0], 'Smaller damage', ha='right', va='bottom')


    fig.tight_layout()
    fig.savefig(savedir + 'shock_effects.png')
    # code.interact(local=dict(globals(), **locals()))