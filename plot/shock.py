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
from mpl_toolkits.axes_grid1 import ImageGrid
import logging
logger = logging.getLogger('sLogger')
from . import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

import warnings
warnings.simplefilter("ignore", UserWarning) # ImageGrid spits some annoying harmless warnings w.r.t. tight_layout

def resilience(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    savedir = '../outputs/{}/plots/resilience/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    adap_scenarios = list(results.keys())
    
    ## ALL AGENTS
    land_area = results[adap_scenarios[0]].columns   
    grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)
    # line_plots(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes)
    
    ## JUST MIDDLE AGENTS
    grid_plot(savedir, adap_scenarios, [land_area[1]], results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes, ext2='_middle')
    # line_plots(savedir, adap_scenarios, land_area[1], results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes, ext2='_middle')
    # code.interact(local=dict(globals(), **locals()))
    

def policy_design_both_res_types(dev_cc, dev_ins, res_cc, res_ins, shock_mags, shock_times, T_res, T_dev, exp_name):
    '''
    plot a selected plot of both of the resilience types together
    '''
    T_res = 3
    T_shock = shock_times[0]
    mag_str = str(shock_mags[0]).replace('.','_')
    savedir = '../outputs/{}/plots/policy_design/{}/'.format(exp_name, shock_mags[0])
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    land_area = 1.5
    outcome = 'income'

    fig, axs = plt.subplots(2,2, figsize=(12,8))
    ax_flat = axs.flatten()

    #### DEVELOPMENT RESILIENCE ####
    ## cover crop
    ax = axs[0,0]
    plt_data = np.array(dev_cc[land_area].unstack())
    xs = dev_cc.index.levels[1] # cost
    ys = dev_cc.index.levels[0] # n fixed
    hm = ax.imshow(plt_data.astype(float), cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                aspect='auto')
    ax.scatter([1], [80], color='k') # default value
    ## insurance
    ax = axs[1,0]
    plt_data = np.array(dev_ins[land_area].unstack())
    x2s = dev_ins.index.levels[1] # cost
    y2s = dev_ins.index.levels[0] * 100 # convert to %age
    hm2 = ax.imshow(plt_data.astype(float), cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(x2s), max(x2s), min(y2s), max(y2s)],
                aspect='auto')
    ax.scatter([1], [10], color='k') # default value

    #### CLIMATE RESILIENCE ####
    ## cover crop
    query_str = 'mag=="{}" & assess_pd=={} & time=={}'.format(mag_str, T_res, T_shock)
    d_subs = res_cc[outcome].query(query_str)
    d_plot = np.array(d_subs[str(land_area)].unstack()) # just single agent type
    hm = axs[0,1].imshow(d_plot, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                aspect='auto')
    # add points for default
    axs[0,1].scatter([1], [80], color='k')
    ## insurance
    d_subs = res_ins[outcome].query(query_str)
    d_plot = []
    for land in d_subs.columns:
        d_plot.append(np.array(d_subs[land].unstack()))
    d_plot = np.mean(np.array(d_plot), axis=0)
    hm = axs[1,1].imshow(d_plot, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(x2s), max(x2s), min(y2s), max(y2s)],
                aspect='auto')
    # add points for default
    axs[1,1].scatter([1], [10], color='k')

    ## labels
    fs = 20
    # axs[0,0].set_title('Development resilience: cover crop', fontsize=fs)
    # axs[1,0].set_title('Development resilience: insurance', fontsize=fs)
    # axs[0,1].set_title('Climate resilience: cover crop', fontsize=fs)
    # axs[1,1].set_title('Climate resilience: insurance', fontsize=fs)
    axs[0,0].set_ylabel('Legume cover crop\nN fixation (kg N/ha)')
    axs[1,0].set_ylabel('Insured climate percentile')
    axs[1,0].set_xlabel('Cost factor')
    axs[1,1].set_xlabel('Cost factor')

    for ax in axs[0]:
        ax.set_xticklabels([])
    for ax in axs[:,-1]:
        ax.set_yticklabels([])


    axs[0,0].text(0.5, 1.05, 'Development resilience\n'+r'$T_{dev}=$'+str(T_dev), fontsize=fs, ha='center',va='bottom', transform=axs[0,0].transAxes)
    axs[0,1].text(0.5, 1.05, 'Shock resilience\n'+r'$T_{shock}=$'+str(T_shock)+r', $T_{assess}=$'+str(T_res), fontsize=fs, ha='center',va='bottom', transform=axs[0,1].transAxes)
    # axs[0,0].text(-0.2, 0.5, 'Cover crop', fontsize=fs, ha='right', va='center', transform=axs[0,0].transAxes)
    # axs[1,0].text(-0.2, 0.5, 'Insurance', fontsize=fs, ha='right', va='center', transform=axs[1,0].transAxes)

    # cb_ax = fig.add_axes([0.34, -0.03, 0.37, 0.03])
    # cb_ax = fig.add_axes([0.5, -0.03, 0.5, 0.02])
    cb_ax = fig.add_axes([1, 0.22, 0.02, 0.5])
    cbar = fig.colorbar(hm, orientation='vertical', cax=cb_ax)
    cbar.set_label('P(CC>ins)')

    labels = ['(A)','(B)','(C)','(D)']
    for a, ax in enumerate(ax_flat):
        ax.grid(False)
        ax.text(0.02,0.98,labels[a], fontsize=20, transform=ax.transAxes, ha='left', va='top')
    fig.savefig(savedir + 'policy_both_resilience_{}ha.png'.format(str(land_area).replace('.','_')), bbox_inches='tight', dpi=200)
    # sys.exit()
    # code.interact(local=dict(globals(), **locals()))

def policy_design_dev_res(d_cc, d_ins, shock_mags, exp_name):
    '''
    plot as a function of policy parameters
    create one plot with both policies for a selected T_shock and T_res
    and a grid-plot for each policy with all T_shock and T_res
    '''
    savedir = '../outputs/{}/plots/policy_design/{}/'.format(exp_name, shock_mags[0])
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    mag_str = str(shock_mags[0]).replace('.','_')
    land_area = d_cc.columns

    fig, axs = plt.subplots(2,3,figsize=(16,10))
    for li, land in enumerate(land_area):
        ## cover crop
        ax = axs[1,li]
        plt_data = np.array(d_cc[land].unstack())
        xs = d_cc.index.levels[1] # cost
        ys = d_cc.index.levels[0] # n fixed
        
        hm = ax.imshow(plt_data.astype(float), cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                    aspect='auto')
        ax.scatter([1], [80], color='k') # default value

        ## insurance
        ax2 = axs[0,li]
        plt_data2 = np.array(d_ins[land].unstack())
        x2s = d_ins.index.levels[1] # cost
        y2s = d_ins.index.levels[0] * 100 # convert to %age
        hm2 = ax2.imshow(plt_data2.astype(float), cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(x2s), max(x2s), min(y2s), max(y2s)],
                    aspect='auto')
        ax2.scatter([1], [10], color='k') # default value
        
        # formatting
        if li > 0:
            for axx in [ax, ax2]:
                axx.set_ylabel('')
                axx.set_yticklabels([])
        else:
            ax.set_ylabel('Nitrogen fixation (kg N/ha)')
            ax2.set_ylabel('Insured climate percentile')
        for axx in [ax, ax2]:
            axx.set_xlabel('Cost factor')
        for axx in [ax, ax2]:
            axx.set_title('{} ha'.format(land))
            axx.grid(False)

    # color bar
    cb_ax = fig.add_axes([0.34, -0.03, 0.37, 0.03])
    cbar = fig.colorbar(hm2, orientation='horizontal', cax=cb_ax)
    cbar.set_label('P(CC>ins)')

    # labels
    axs[0,0].text(-0.2, 1.1, 'A: Insurance', fontsize=28, transform=axs[0,0].transAxes)
    axs[1,0].text(-0.2, 1.1, 'B: Legume cover', fontsize=28, transform=axs[1,0].transAxes)

    fig.savefig(savedir + 'policy_development_mag_{}.png'.format(mag_str),
        bbox_inches='tight', dpi=200) 
    plt.close('all')

def policy_design_single(d_cc, d_ins, shock_mags, shock_times, T_res, exp_name):
    '''
    plot as a function of policy parameters
    create one plot with both policies for a selected T_shock and T_res
    and a grid-plot for each policy with all T_shock and T_res
    '''
    savedir = '../outputs/{}/plots/policy_design/{}/'.format(exp_name, shock_mags[0])
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    mag_str = str(shock_mags[0]).replace('.','_')
    
    for outcome in d_cc.keys():
        cc = d_cc[outcome]
        ins = d_ins[outcome]
        land_area = cc.columns

        #### 1. simple figure ####
        t_res = 5
        t_shock = 10
        fig, axs = plt.subplots(2,3,figsize=(16,10))
        query_str = 'mag=="{}" & assess_pd=={} & time=={}'.format(mag_str, t_res, t_shock)
        cc = cc.query(query_str)
        ins = ins.query(query_str)
        for li, land in enumerate(land_area):
            ## cover crop
            try:
                ax = axs[0,li]
            except:
                code.interact(local=dict(globals(), **locals()))
            plt_data = np.array(cc[land].unstack())
            xs = cc.index.levels[4] # cost
            ys = cc.index.levels[3] # n fixed
            hm = ax.imshow(plt_data, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                        aspect='auto')
            ax.scatter([1], [80], color='k') # default value

            ## insurance
            ax2 = axs[1,li]
            plt_data2 = np.array(ins[land].unstack())
            x2s = ins.index.levels[4] # cost
            y2s = ins.index.levels[3] * 100 # convert to %age
            hm2 = ax2.imshow(plt_data2, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(x2s), max(x2s), min(y2s), max(y2s)],
                        aspect='auto')
            ax2.scatter([1], [10], color='k') # default value
            
            # formatting
            if li > 0:
                for axx in [ax, ax2]:
                    axx.set_ylabel('')
                    axx.set_yticklabels([])
            else:
                ax.set_ylabel('Nitrogen fixation (kg N/ha)')
                ax2.set_ylabel('Insured climate percentile')
            for axx in [ax, ax2]:
                axx.set_xlabel('Cost factor')
            for axx in [ax, ax2]:
                axx.set_title('{} ha'.format(land))
                axx.grid(False)

        # color bar
        cb_ax = fig.add_axes([0.34, -0.03, 0.37, 0.03])
        cbar = fig.colorbar(hm2, orientation='horizontal', cax=cb_ax)
        cbar.set_label('P(CC>ins)')

        # labels
        axs[0,0].text(-0.2, 1.1, 'A: Legume cover', fontsize=28, transform=axs[0,0].transAxes)
        axs[1,0].text(-0.2, 1.1, 'B: Insurance', fontsize=28, transform=axs[1,0].transAxes)

        fig.savefig(savedir + 'policy_{}_shockyr_{}_assess_{}_mag_{}.png'.format(outcome, t_shock, t_res, mag_str),
            bbox_inches='tight', dpi=200) 
        # code.interact(local=dict(globals(), **locals()))
        plt.close('all')

def policy_design_all(d_cc, d_ins, shock_mags, shock_times, T_res, exp_name):
    '''
    create a separate figure for each land size and each outcome and each policy
    in each figure, include all the T_res and T_shock values
    '''
    savedir = '../outputs/{}/plots/policy_design/{}/'.format(exp_name, shock_mags[0])
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    mag_str = str(shock_mags[0]).replace('.','_')
    
    policies = ['insurance','cover_crop']
    labels = ['fraction insured','N fixation']
    defaults = [10, 80]
    T_res_plot = [1,3,5,7,9]
    T_shock_plot = [2,6,10,16]

    for p, d in enumerate([d_ins,d_cc]):
        for outcome in d_cc.keys():
            land_area = d_cc[outcome].columns
            for li, land in enumerate(land_area):
                #### create figure ####
                fig = plt.figure(figsize=(4*len(T_res_plot),4*len(T_shock_plot)))
                axs = ImageGrid(fig, 111, nrows_ncols=(len(T_shock_plot), len(T_res_plot)), 
                    axes_pad=0.15, add_all=True, label_mode='L',
                    cbar_mode='single',cbar_location='bottom', aspect=False,
                    cbar_pad='5%', direction='row')

                i=0
                # loop over the plots
                for ts, t_shock in enumerate(T_shock_plot):
                    for tr, t_res in enumerate(T_res_plot):
                        ax = axs[i]
                        
                        query_str = 'mag=="{}" & assess_pd=={} & time=={}'.format(mag_str, t_res, t_shock)
                        d_subs = d[outcome].query(query_str)
                        d_plot = np.array(d_subs[land].unstack())
                        xs = d[outcome].index.levels[4]
                        ys = d[outcome].index.levels[3]
                        ys *= 100 if policies[p] == 'insurance' else 1
                        hm = ax.imshow(d_plot, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                                    aspect='auto')
                        # add point for default
                        ax.scatter([1], [defaults[p]], color='k')
                        i += 1
                        
                        # labels
                        if ts == 0:
                            ax.set_title('T_res={}'.format(t_res), fontsize=16)
                        if tr == 0:
                            ax.set_ylabel('T_shock={}\n\n{}'.format(t_shock, labels[p]), fontsize=16)
                        if t_shock == max(T_shock_plot):
                            ax.set_xlabel('Cost factor')
                            
                        # formatting
                        ax.grid(False)

                # colorbar
                cax = axs.cbar_axes[0]
                cbar = cax.colorbar(hm)
                axis = cax.axis[cax.orientation]
                axis.label.set_text("P(CC>ins)")
                fig.savefig(savedir + 'all_policy_{}_{}_{}ha_mag_{}.png'.format(outcome, policies[p], land, mag_str),
                    bbox_inches='tight', dpi=200) 
                plt.close('all')

def policy_design_all_combined(d_cc, d_ins, shock_mags, shock_times, T_res, exp_name):
    '''
    average over all land sizes
    single shock year
    columns represent different t_res
    one row for CC and one for insurance
    create a separate figure for each land size and each outcome and each policy
    in each figure, include all the T_res and T_shock values
    '''
    savedir = '../outputs/{}/plots/policy_design/{}/'.format(exp_name, shock_mags[0])
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    mag_str = str(shock_mags[0]).replace('.','_')
    
    policies = ['insurance','cover_crop']
    names = ['A: Insurance', 'B: Legume cover']
    ds = [d_ins, d_cc]
    ylabels = ['Insured climate %ile','N fixation (kg N/ha)']
    defaults = [10, 80]
    T_res_plot = [1,3,5,7,9]
    t_shock = 10

    for outcome in d_cc.keys():
        for i in range(len(policies)):
            fig = plt.figure(figsize=(0.7*4*len(T_res_plot),0.7*5))#, sharey='row', sharex='col')
            axs = ImageGrid(fig, 111, nrows_ncols=(1, len(T_res_plot)), 
                axes_pad=0.15, add_all=True, label_mode='L',
                cbar_mode='single',cbar_location='right', aspect=False,
                cbar_pad='5%', direction='column')

            for t, t_res in enumerate(T_res_plot):
                ax = axs[t]
            
                query_str = 'mag=="{}" & assess_pd=={} & time=={}'.format(mag_str, t_res, t_shock)
                # code.interact(local=dict(globals(), **locals()))
                d_subs = ds[i][outcome].query(query_str)
                d_plot = []
                for land in d_subs.columns:
                    d_plot.append(np.array(d_subs[land].unstack()))
                d_plot = np.mean(np.array(d_plot), axis=0)
                xs = ds[i][outcome].index.levels[4]
                ys = ds[i][outcome].index.levels[3]
                ys *= 100 if policies[i] == 'insurance' else 1
                hm = ax.imshow(d_plot, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(xs), max(xs), min(ys), max(ys)],
                            aspect='auto')
                # add point for default
                ax.scatter([1], [defaults[i]], color='k')
                
                # labels
                ax.set_title(r'$T_{assess}=$'+str(t_res), fontsize=16)
                ax.set_xlabel('Cost factor')
                if t == 0:
                    ax.set_ylabel(ylabels[i])
                    ax.text(-0.2, 1.2, names[i], fontsize=28, transform=ax.transAxes)
                    
                # formatting
                ax.grid(False)


            # colorbar
            cax = axs.cbar_axes[0]
            cbar = cax.colorbar(hm)
            axis = cax.axis[cax.orientation]
            axis.label.set_text("P(CC>ins)")
            fig.savefig(savedir + 'combined_policy_{}_{}_mag_{}_shockyr{}.png'.format(outcome, policies[i], mag_str, t_shock),
                bbox_inches='tight', dpi=200) 
            plt.close('all')

def shock_mag_grid_plot(results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes):
    '''
    grid plot with x = shock mag, y=T_res and z=T_shock at which P(CC>ins)=0.5
    '''
    savedir = '../outputs/{}/plots/shock_mag/'.format(exp_name)
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
            axs = ImageGrid(fig, 111, nrows_ncols=(1,len(land_area)), axes_pad=0.5, add_all=True, label_mode='L',
                cbar_mode='single',cbar_location='right', aspect=False)

            for li, land in enumerate(land_area):
                ax = axs[li]
                plt_data = np.array(probs_t[land].unstack().unstack()).transpose()
                hm = ax.imshow(plt_data, cmap='bwr', vmin=0, vmax=1, origin='lower', extent=[min(shock_mags), max(shock_mags), min(T_res), max(T_res)],
                    aspect='auto')
                
                # formatting
                if li == 0:
                    ax.set_ylabel(r'Assessment period ($T_{assess}$)')
                ax.set_title('{} ha'.format(land))
                ax.set_xlabel('Shock magnitude')
                ax.grid(False)

            # colorbar
            cax = axs.cbar_axes[0]
            cbar = cax.colorbar(hm)
            axis = cax.axis[cax.orientation]
            axis.label.set_text("P(CC>ins)")

            ext = '_baseline' if baseline_resilience else ''
            fig.savefig(savedir + '{}_shock_magnitude_grid{}_shockyr_{}.png'.format(outcome, ext, shock_time), bbox_inches='tight', dpi=200) 
            # sys.exit()
            # code.interact(local=dict(globals(), **locals()))
            plt.close('all')

def grid_plot(savedir, adap_scenarios, land_area, results, shock_mags, shock_times, T_res, exp_name, baseline_resilience, outcomes, ext2=''):
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
            fig = plt.figure(figsize=(6*len(land_area), 5))
            axs = ImageGrid(fig, 111, nrows_ncols=(1,len(land_area)), axes_pad=0.5, add_all=True, label_mode='L',
                cbar_mode='single',cbar_location='right', aspect=False)

            for li, land in enumerate(land_area):
                ax = axs[li]
                vals = probs.loc[[mag], land].to_xarray()

                # create imshow plot (using xarray imshow wrapper)
                hm = vals[m].plot(ax=ax, cmap='bwr',vmin=0,vmax=1, add_colorbar=False)
                
                # formatting
                if li > 0:
                    ax.set_ylabel('')
                    # ax.set_yticklabels([])
                else:
                    ax.set_ylabel(r'Assessment period ($T_{assess}$)')
                if len(land_area)>1:
                    ax.set_title('{} ha'.format(land))
                else:
                    ax.set_title('')
                ax.set_xlabel(r'Time of shock ($T_{shock}$)')

            # color bar
            cax = axs.cbar_axes[0]
            cbar = cax.colorbar(hm)
            axis = cax.axis[cax.orientation]
            axis.label.set_text("P(CC>ins)")

            ext = '_baseline' if baseline_resilience else ''
            fig.savefig(savedir + '{}_shock_grid_{}{}{}.png'.format(outcome, mag, ext, ext2), bbox_inches='tight', dpi=200)
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
            fig.savefig(savedir + '{}_shock_effects_{}{}.png'.format(outcome, mag_str, ext), dpi=200)
            plt.close('all')