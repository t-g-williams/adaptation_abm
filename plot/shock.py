import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
import copy
import sys
import xarray
from . import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

def main(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome):
    savedir = '../outputs/{}/shocks/'.format(exp_name)

    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns
    
    grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome)
    line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome)

def grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome):
    '''
    for each agent type, plot a grid showing P(CC>ins) as a function of T_res and T_shock
    '''
    # calculate the probability that CC > insurance
    # these are measures of DAMAGE
    # so for cover_crop to be better, damage should be lower
    bools = results['cover_crop'] < results['insurance']
    probs = bools.groupby(level=[0,1,2]).mean() # take the mean over the replications

    # create a separate figure for each shock magnitude
    for m, mag in enumerate(shock_mags):
        fig, axs = plt.subplots(1, len(land_area), figsize=(6*len(land_area), 5))

        for li, land in enumerate(land_area):
            ax = axs[li]
            vals = probs.loc[[mag], land].to_xarray()

            # create imshow plot (using xarray imshow wrapper)
            if li==(len(land_area)-1): # include color bar
                vals[0].plot(ax=ax, cmap='bwr',vmin=0,vmax=1, 
                    cbar_kwargs={'label' : 'P(CC>ins)'})
            else:
                vals[0].plot(ax=ax, cmap='bwr',vmin=0,vmax=1, add_colorbar=False)
            
            # formatting
            if li > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            ax.set_title('{} ha'.format(land))

        ext = '_baseline' if baseline_resilience else ''
        fig.savefig(savedir + '{}_shock_grid_{}{}.png'.format(outcome, str(mag).replace('.','_'), ext)) 

def line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcome):
    '''
    compare the relative benefit of each policy over time
    '''

    # create a separate plot for each shock magnitude
    for m, mag in enumerate(shock_mags):
        mag_str = str(mag).replace('.','_')

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
                    d = results[sc][nplot][mag][T].groupby('time').mean()
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
        ext = '_baseline' if baseline_resilience else ''
        fig.savefig(savedir + '{}_shock_effects_{}{}.png'.format(outcome, mag_str, ext))