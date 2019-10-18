import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import pandas as pd
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

def resilience(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    savedir = '../outputs/{}/shocks/'.format(exp_name)

    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns   
    grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)
    line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)

def shock_mag_grid_plot(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    grid plot with x = shock mag, y=T_res and z=T_shock at which P(CC>ins)=0.5
    '''
    savedir = '../outputs/{}/shocks/'.format(exp_name)
    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns

    code.interact(local=dict(globals(), **locals()))
    # calculate the probability that CC > insurance
    # these are measures of DAMAGE
    # so for cover_crop to be better, damage should be lower
    bools = results['cover_crop'] < results['insurance']
    probs = bools.groupby(level=[0,1,2]).mean() # take the mean over the replications
    crit_val = 0.5

    # find the minimum T_shock at which P(cc>ins)>crit_val
    mags_str = np.array([str(si).replace('.','_') for si in shock_mags])
    pts = np.full((len(land_area), len(shock_mags), len(T_res)), np.nan)
    for m, mag in enumerate(mags_str):
        for t, t_res in enumerate(T_res):

            tmp = probs.loc[(mag, t_res)]
            for l, li in enumerate(land_area):
                if max(tmp[li]) > crit_val:
                    pts[l,m,t] = np.argmax(np.array(tmp[li])>crit_val)
                else:
                    code.interact(local=dict(globals(), **locals()))
    
    pts = np.ma.array(pts, mask=np.isnan(pts))
    
    fig, axs = plt.subplots(1, len(land_area), figsize=(6*len(land_area), 5))

    for li, land in enumerate(land_area):
        ax = axs[li]

        cmap = mpl.cm.YlOrRd
        cmap.set_bad('gray')
        hm = ax.imshow(pts[li], cmap=cmap)#, extent=[min(shock_mags), max(shock_mags), min(T_res), max(T_res)])

        if li==(len(land_area)-1): # include color bar
            cbar = fig.colorbar(hm)
            cbar.set_label('T_shock at which P(CC>ins)=0.5')
        
        # formatting
        if li > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Assessment pd (T_res)')
        ax.set_title('{} ha'.format(land))
        ax.set_xlabel('Shock magnitude')

    ext = '_baseline' if baseline_resilience else ''
    fig.savefig(savedir + '{}_shock_magnitude_grid{}.png'.format(outcome, ext)) 
    plt.close('all')

def grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    for each agent type, plot a grid showing P(CC>ins) as a function of T_res and T_shock
    '''
    mags_str = np.array([str(si).replace('.','_') for si in shock_mags])
    for outcome in outcomes:
        # calculate the probability that CC > insurance
        # these are measures of DAMAGE
        # so for cover_crop to be better, damage should be lower
        bools = results['cover_crop'].loc[(outcome)] < results['insurance'].loc[(outcome)]
        probs = bools.groupby(level=[0,1,2]).mean() # take the mean over the replications

        # create a separate figure for each shock magnitude
        for m, mag in enumerate(mags_str):
            fig, axs = plt.subplots(1, len(land_area), figsize=(6*len(land_area), 5))

            for li, land in enumerate(land_area):
                ax = axs[li]
                vals = probs.loc[[mag], land].to_xarray()

                # create imshow plot (using xarray imshow wrapper)
                if li==(len(land_area)-1): # include color bar
                    vals[m].plot(ax=ax, cmap='bwr',vmin=0,vmax=1, 
                        cbar_kwargs={'label' : 'P(CC>ins)'})
                else:
                    vals[m].plot(ax=ax, cmap='bwr',vmin=0,vmax=1, add_colorbar=False)
                
                # formatting
                if li > 0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                ax.set_title('{} ha'.format(land))

            ext = '_baseline' if baseline_resilience else ''
            fig.savefig(savedir + '{}_shock_grid_{}{}.png'.format(outcome, mag, ext))
            plt.close('all')

def line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    compare the relative benefit of each policy over time
    '''
    for outcome in outcomes:
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
                        d = results[sc].loc[(outcome,mag_str,T),nplot].groupby('time').mean()
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
            plt.close('all')