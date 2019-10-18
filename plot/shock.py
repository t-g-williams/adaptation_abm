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
    savedir = '../outputs/{}/plots/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns   
    grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)
    line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)

def policy_design(d_cc, d_ins, shock_mags, shock_times, T_res, exp_name):
    '''
    plot as a function of policy parameters
    create one plot with both policies for a selected T_shock and T_res
    and a grid-plot for each policy with all T_shock and T_res
    '''
    mag_str = str(shock_mags[0]).replace('.','_')
    land_area = res_cc.columns

    #### 1. simple figure ####
    t_res = 5
    t_shock = 10
    fig, axs = plt.subplots(2,3,figsize=(16,10))
    # cover crop
    cc = d_cc.query('mag=={} AND assess_pd=={} AND time=={}'.format(mag_str, t_res, t_shock))
    print('UP TO HERE!!!')
    code.interact(local=dict(globals(), **locals()))
    for li, land in enumerate(land_area):
        ax = axs[li]
        plt_data = np.array(cc[land].unstack().unstack())
        hm = ax.imshow(plt_data, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(shock_mags), max(shock_mags), min(T_res), max(T_res)],
                    aspect='auto')

def shock_mag_grid_plot_old(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    grid plot with x = shock mag, y=T_res and z=T_shock at which P(CC>ins)=0.5
    '''
    savedir = '../outputs/{}/plots/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns
    mags_str = np.array([str(si).replace('.','_') for si in shock_mags])

    for outcome in outcomes:
        # calculate the probability that CC > insurance
        # these are measures of DAMAGE
        # so for cover_crop to be better, damage should be lower
        bools = results['cover_crop'].loc[(outcome)] < results['insurance'].loc[(outcome)]
        probs = bools.groupby(level=[0,1,2]).mean() # take the mean over the replications
        
        ## different plot for each shock time
        for t, shock_time in enumerate(shock_times):
            probs_t = probs.query('time=={}'.format(shock_time))

            fig, axs = plt.subplots(1, len(land_area), figsize=(6*len(land_area), 5))

            for li, land in enumerate(land_area):
                ax = axs[li]
                plt_data = np.array(probs_t[land].unstack().unstack()).transpose()
                hm = ax.imshow(plt_data, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(shock_mags), max(shock_mags), min(T_res), max(T_res)],
                    aspect='auto')

                # if li==(len(land_area)-1): # include color bar
                if li==1: # include color bar
                    cbar = fig.colorbar(hm, orientation='horizontal', pad=0.2)
                    cbar.set_label('P(CC>ins)')
                
                # formatting
                if li > 0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('Assessment period (T_res)')
                ax.set_title('{} ha'.format(land))
                ax.set_xlabel('Shock magnitude')
                ax.grid(False)

            ext = '_baseline' if baseline_resilience else ''
            fig.savefig(savedir + '{}_shock_magnitude_grid{}_shockyr_{}.png'.format(outcome, ext, shock_time)) 
            code.interact(local=dict(globals(), **locals()))
            plt.close('all')

def shock_mag_grid_plot(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    grid plot with x = shock mag, y=T_res and z=T_shock at which P(CC>ins)=0.5
    '''
    savedir = '../outputs/{}/plots/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    adap_scenarios = list(results.keys())
    land_area = results[adap_scenarios[0]].columns
    mags_str = np.array([str(si).replace('.','_') for si in shock_mags])

    for outcome in outcomes:
        # calculate the probability that CC > insurance
        # these are measures of DAMAGE
        # so for cover_crop to be better, damage should be lower
        bools = results['cover_crop'].loc[(outcome)] < results['insurance'].loc[(outcome)]
        probs = bools.groupby(level=[0,1,2]).mean() # take the mean over the replications
        
        ## different plot for each shock time
        for t, shock_time in enumerate(shock_times):
            probs_t = probs.query('time=={}'.format(shock_time))

            fig = plt.figure(figsize=(6*len(land_area), 5))
            from mpl_toolkits.axes_grid1 import ImageGrid
            axs = ImageGrid(fig, 111, nrows_ncols=(1,len(land_area)), axes_pad=0.5, add_all=True, label_mode='L',
                cbar_mode='single',cbar_location='right', aspect=False)

            for li, land in enumerate(land_area):
                ax = axs[li]
                plt_data = np.array(probs_t[land].unstack().unstack()).transpose()
                hm = ax.imshow(plt_data, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(shock_mags), max(shock_mags), min(T_res), max(T_res)],
                    aspect='auto')
                
                # formatting
                if li == 0:
                    ax.set_ylabel('Assessment period (T_res)')
                ax.set_title('{} ha'.format(land))
                ax.set_xlabel('Shock magnitude')
                ax.grid(False)

            # colorbar
            cax = axs.cbar_axes[0]
            cbar = cax.colorbar(hm)
            axis = cax.axis[cax.orientation]
            axis.label.set_text("P(CC>ins)")

            ext = '_baseline' if baseline_resilience else ''
            fig.savefig(savedir + '{}_shock_magnitude_grid{}_shockyr_{}.png'.format(outcome, ext, shock_time), bbox_inches='tight') 
            # sys.exit()
            # code.interact(local=dict(globals(), **locals()))
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