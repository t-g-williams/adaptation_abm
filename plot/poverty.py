import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
import string
import copy
import sys
import matplotlib.transforms as transforms
from . import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

import warnings
warnings.simplefilter("ignore", UserWarning) # ImageGrid spits some annoying harmless warnings w.r.t. tight_layout

def main(mods, nreps, inp_base, scenarios, exp_name, T, shock_years=[], dir_ext=''):
    '''
    plot each agent type (number of plots) separately
    this assumes there's 3 agent types
    '''
    savedir = '../outputs/{}/plots/{}'.format(exp_name, dir_ext)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if len(shock_years) == 0:
        ## FOR PAPER
        utility_plot_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        time_plot_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        time_plot('util', 'E[utility]', mods, nreps, inp_base, scenarios, exp_name, T, savedir, risk_tol=500)
        time_plot('util', 'E[utility]', mods, nreps, inp_base, scenarios, exp_name, T, savedir, risk_tol=1000)
        time_plot('util', 'E[utility]', mods, nreps, inp_base, scenarios, exp_name, T, savedir, risk_tol=3000)
        time_plot('util', 'E[utility]', mods, nreps, inp_base, scenarios, exp_name, T, savedir, risk_tol=5000)
        time_plot('var_income', 'std.dev(income)', mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        time_plot('exp_income', 'E[income]', mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        time_plot('poverty', 'P(wealth > 0)', mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        time_plot('fert_choice', 'P(choose fertilizer)', mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        # code.interact(local=dict(globals(), **locals()))

        # OLD PLOTS
        # only run these for the adaptation scenarios -- this assumes the length of shock years here is zero
        # poverty_trap_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        # poverty_trap(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        # combined_wealth_income(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        # agent_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir, 'wealth')
        # agent_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir, 'income')
    
    # first_round_plots(mods, nreps, inp_base, scenarios, exp_name, T, savedir, shock_years)

def poverty_trap_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot wealth(t) against wealth(t+1)
    '''
    burnin_yrs = 10
    fig = plt.figure(figsize=(5,5))
    cols = ['black','red','blue']
    sc = -1 # scenario counter
    ax = fig.add_subplot(111)

    for scenario, mods_sc in mods.items():
        sc +=1
        wlth = mods_sc['wealth']

        ## calculate pairs of wealth(t) and wealth(t+1)
        wlth_t = []
        wlth_t1 = []
        for r in range(nreps):        
            for t in range(burnin_yrs, inp_base['model']['T']):
                wlth_t.append(wlth[r,t,:])
                wlth_t1.append(wlth[r,t+1,:])
        
        wlth_t = np.array([item for sublist in wlth_t for item in sublist])
        wlth_t1 = np.array([item for sublist in wlth_t1 for item in sublist])

        # format for plotting
        xs = np.linspace(wlth_t.min(), np.percentile(wlth_t, q=90), 100)
        ys = np.full((99,3), np.nan)
        for i in range(1, len(xs)):
            try:
                ys[i-1] = np.percentile(wlth_t1[(wlth_t >= xs[i-1]) & (wlth_t < xs[i])], q=[10,50,90])
            except:
                ys[i-1] = np.array([np.nan,np.nan,np.nan])
        
        # plot
        ax.plot(np.sqrt(xs[:-1]), np.sqrt(ys[:,1]), color=cols[sc], lw=1.5, label=scenario) # median
        # ax.plot(xs[:-1], ys[:,1], color=cols[sc], lw=1.5, label=scenario) # median

    # formatting
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim([0, mx])
    ax.set_ylim([0, mx])
    ax.grid(False)
    ax.set_xlabel('Wealth(t)')
    ax.set_ylabel('Wealth(t+1)')
    ax.set_title('Poverty trap dynamics')
    ax.legend()
    ax.plot([0,mx], [0,mx], lw=0.75, color='k')
    fig.savefig(savedir + 'poverty_trap_combined.png')
    plt.close('all')

def poverty_trap(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot wealth(t) against wealth(t+1)
    '''
    burnin_yrs = 10
    plots = inp_base['agents']['land_area_init']
    N = len(plots)
    fig = plt.figure(figsize=(5*N,5))
    axs = []
    cols = ['black','red','blue']
    mx = 0

    for n, land_area in enumerate(plots):
        sc = -1 # scenario counter
        ax = fig.add_subplot(1,N,n+1)
        axs.append(ax)

        for scenario, mods_sc in mods.items():
            sc +=1
            wlth = mods_sc['wealth']

            ## calculate pairs of wealth(t) and wealth(t+1)
            wlth_t = []
            wlth_t1 = []
            for r in range(nreps):
                agents = mods_sc['land_area'][r] == land_area
            
                for t in range(burnin_yrs, inp_base['model']['T']):
                    wlth_t.append(wlth[r,t,agents])
                    wlth_t1.append(wlth[r,t+1,agents])
            
            wlth_t = np.array([item for sublist in wlth_t for item in sublist])
            wlth_t1 = np.array([item for sublist in wlth_t1 for item in sublist])

            # format for plotting
            xs = np.linspace(wlth_t.min(), np.percentile(wlth_t, q=90), 100)
            ys = np.full((99,3), np.nan)
            for i in range(1, len(xs)):
                try:
                    ys[i-1] = np.percentile(wlth_t1[(wlth_t >= xs[i-1]) & (wlth_t < xs[i])], q=[10,50,90])
                except:
                    ys[i-1] = np.array([np.nan,np.nan,np.nan])
            
            # plot
            ax.plot(xs[:-1], ys[:,1], color=cols[sc], lw=1.5, label=scenario) # median
            mx = max(mx, max(xs.max(), ys.max()))

    # formatting
    for a, ax in enumerate(axs):
        ax.set_xlim([0, mx])
        ax.set_ylim([0, mx])
        ax.grid(False)
        ax.set_xlabel('Wealth(t)')
        ax.set_title('{} ha'.format(plots[a]))
        ax.legend()
        ax.plot([0,mx], [0,mx], lw=0.75, color='k')

        if a>0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Wealth(t+1)')

    fig.tight_layout()
    fig.savefig(savedir + 'poverty_trap.png')
    plt.close('all')
    # code.interact(local=dict(globals(), **locals()))

def time_plot_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the specified outcome over time for each agent type
    '''
    outcomes = ['poverty','exp_income','var_income']
    ylabs = ['P(wealth > 0)','E[income]','std.dev(income)']

    lands = inp_base['agents']['land_area_init']
    titles = ['Land poor','Middle','Land rich']
    burnin = inp_base['adaptation']['burnin_period']
    fig, ax_all = plt.subplots(len(outcomes)+1,len(lands), figsize=(5*len(lands), 3*len(outcomes)+0.5), sharey='row', gridspec_kw={'height_ratios':[1]*len(outcomes)+[0.05],
        'hspace':0.15,'wspace':0.075})
    axs = ax_all[:-1]
    [axi.remove() for axi in ax_all[-1,:]]
    ax_flat = axs.flatten()
    cols = {'baseline':'k','cover_crop':'r','insurance':'b','both':'g'}
    lss = {'baseline':'-','cover_crop':'--','insurance':'-.','both':':'}

    for n, land_area in enumerate(lands):
        for scenario, mods_sc in mods.items():
            m = scenario
            ## get the relevant outcome data
            for o,outcome in enumerate(outcomes):
                if outcome == 'poverty': # plot probability of non-poverty (wealth>0) over time
                    d = mods_sc['wealth']
                    xs = np.arange(T+burnin+1)
                elif outcome in ['exp_income','var_income','util']:
                    d = mods_sc['income']
                    xs = np.arange(T+burnin)

                # flatten it
                all_d = []
                for r in range(nreps):
                    agents = mods_sc['land_area'][r] == land_area
                    all_d.append(list(d[r,:,agents]))
                all_d = np.array([item for sublist in all_d for item in sublist])
                # ^ this is (agents, time) shape

                # extract the relevant info for plotting
                if outcome == 'poverty':
                    plt_data = np.mean(all_d>0, axis=0)
                elif outcome == 'exp_income':
                    plt_data = all_d.mean(0)
                elif outcome == 'var_income':
                    plt_data = np.std(all_d, axis=0)

                axs[o,n].plot(xs, plt_data, label=scenario, lw=2, ls=lss[m], color=cols[m])#, marker='o')

                if outcome in ['exp_income','util']:
                    axs[o,n].axhline(0, color='k', lw=1)

    for a, ax in enumerate(ax_flat):
        ax.grid(False)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between([0,burnin], [0,0],[1,1], color='0.5', alpha=0.3, label='burn-in', transform=trans)
        ax.set_xticks(np.arange(0,T+burnin+1,10))
        ax.set_xlim([0,T+burnin+1])
    for a, ax in enumerate(axs[0]):
        ax.set_title(titles[a])
    for a, ax in enumerate(axs[-1]):
        ax.set_xlabel('Year')
    for a, ax in enumerate(axs[:-1].flatten()):
        ax.set_xticks(np.arange(0,T+burnin+1,10))
        ax.set_xticklabels([])

    for a, ax in enumerate(axs[:,0]):
        ax.set_ylabel(ylabs[a])
    for a, ax in enumerate(axs[0]): # poverty
        ax.set_ylim([0,1])

    for o,outc in enumerate(outcomes):
        axs[o,0].text(-0.2,1,string.ascii_uppercase[o],ha='right',va='top',transform=axs[o,0].transAxes, fontsize=24)

    lg = fig.legend(list(mods.keys()) + ['burn-in'], loc=10, bbox_to_anchor=(0.5, 0.01), ncol=len(mods)+1, frameon=False)
    fig.tight_layout()
    ext = '_{}'.format(risk_tol) if outcome == 'util' else ''
    fig.savefig(savedir + 'time_plot_combined.png', bbox_extra_artists=(lg,), dpi=200)
    plt.close('all')
    # sys.exit()


def utility_plot_combined(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the utility under different levels of risk aversion for a land-rich agent
    '''
    tols = [50,500,5000]
    burnin = inp_base['adaptation']['burnin_period']
    fig, ax_all = plt.subplots(2,len(tols), figsize=(5*len(tols), 4), sharey=True, gridspec_kw={'height_ratios':[1,0.05]})
    axs = ax_all[0]
    [axi.remove() for axi in ax_all[1,:]]
    cols = {'baseline':'k','cover_crop':'r','insurance':'b','both':'g'}
    lss = {'baseline':'-','cover_crop':'--','insurance':'-.','both':':'}

    for to, tol in enumerate(tols):
        for scenario, mods_sc in mods.items():
            m = scenario
            ## get the relevant outcome data
            d = mods_sc['income']
            xs = np.arange(T+burnin)
            # flatten it
            all_d = []
            for r in range(nreps):
                agents = mods_sc['land_area'][r] == 2 # land-rich agents
                all_d.append(list(d[r,:,agents]))
            all_d = np.array([item for sublist in all_d for item in sublist])
            # ^ this is (agents, time) shape

            # extract the relevant info for plotting
            utils = np.full(all_d.shape, np.nan)
            utils[all_d>0] = (1 - np.exp(-all_d / tol))[all_d>0]
            utils[all_d<=0] = -(1 - np.exp(all_d / tol))[all_d<=0]
            plt_data = utils.mean(0)

            axs[to].plot(xs, plt_data, label=scenario, lw=2, ls=lss[m], color=cols[m])#, marker='o')

    limz = axs[to].get_ylim()
    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.set_xlabel('Year')
        ax.set_title('Risk tolerance = {}'.format(tols[a]))
        ax.fill_between([0,burnin], [-9e10,-9e10], [9e10,9e10], color='0.5', alpha=0.3, label='burn-in')
    [ax.set_ylim(limz) for ax in axs] # re-set the y lims
    [ax.set_ylim([0,1]) for ax in axs]
    axs[0].set_ylabel('Utility')
    
    lg = fig.legend(list(mods.keys()) + ['burn-in'], loc=10, bbox_to_anchor=(0.5, 0.1), ncol=len(mods)+1, frameon=False)
    fig.tight_layout()
    fig.savefig(savedir + 'utility_plot_combined.png', bbox_extra_artists=(lg,), dpi=200)
    plt.close('all')
    # sys.exit()

def time_plot(outcome, ylab, mods, nreps, inp_base, scenarios, exp_name, T, savedir, risk_tol=False):
    '''
    plot the specified outcome over time for each agent type
    '''
    lands = inp_base['agents']['land_area_init']
    titles = ['Land poor','Middle','Land rich']
    burnin = inp_base['adaptation']['burnin_period']
    fig, ax_all = plt.subplots(2,len(lands), figsize=(5*len(lands), 4), sharey=True, gridspec_kw={'height_ratios':[1,0.05]})
    axs = ax_all[0]
    [axi.remove() for axi in ax_all[1,:]]
    cols = {'baseline':'k','cover_crop':'r','insurance':'b','both':'g'}
    lss = {'baseline':'-','cover_crop':'--','insurance':'-.','both':':'}

    for n, land_area in enumerate(lands):
        for scenario, mods_sc in mods.items():
            m = scenario
            ## get the relevant outcome data
            if outcome == 'poverty': # plot probability of non-poverty (wealth>0) over time
                d = mods_sc['wealth']
                xs = np.arange(T+burnin+1)
            elif outcome in ['exp_income','var_income','util']:
                d = mods_sc['income']
                xs = np.arange(T+burnin)
            elif outcome == 'fert_choice':
                d = mods_sc['fert_choice']
                xs = np.arange(T+burnin)

            # flatten it
            all_d = []
            for r in range(nreps):
                agents = mods_sc['land_area'][r] == land_area
                all_d.append(list(d[r,:,agents]))
            all_d = np.array([item for sublist in all_d for item in sublist])
            # ^ this is (agents, time) shape

            # extract the relevant info for plotting
            if outcome == 'poverty':
                plt_data = np.mean(all_d>0, axis=0)
            elif outcome in ['exp_income','fert_choice']:
                plt_data = all_d.mean(0)
            elif outcome == 'var_income':
                plt_data = np.std(all_d, axis=0)
            elif outcome == 'util':
                utils = np.full(all_d.shape, np.nan)
                utils[all_d>0] = (1 - np.exp(-all_d / risk_tol))[all_d>0]
                utils[all_d<=0] = -(1 - np.exp(all_d / risk_tol))[all_d<=0]
                plt_data = utils.mean(0)

            axs[n].plot(xs, plt_data, label=scenario, lw=2, ls=lss[m], color=cols[m])#, marker='o')

    limz = axs[n].get_ylim()
    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.set_xlabel('Year')
        ax.set_title(titles[a])
        ax.fill_between([0,burnin], [-9e10,-9e10], [9e10,9e10], color='0.5', alpha=0.3, label='burn-in')
    [ax.set_ylim(limz) for ax in axs] # re-set the y lims
    axs[0].set_ylabel(ylab)
    
    # plot-specific formatting
    if outcome == 'poverty':
        [ax.set_ylim([0,1]) for ax in axs] # re-set the y lims
    elif outcome in ['exp_income','util']:
        [ax.axhline(0, color='k', lw=1) for ax in axs]


    lg = fig.legend(list(mods.keys()) + ['burn-in'], loc=10, bbox_to_anchor=(0.5, 0.05), ncol=len(mods)+1, frameon=False, fontsize=14)
    fig.tight_layout()
    ext = '_{}'.format(risk_tol) if outcome == 'util' else ''
    fig.savefig(savedir + 'time_plot_{}{}.png'.format(outcome,ext), bbox_extra_artists=(lg,), dpi=200)
    plt.close('all')

def combined_wealth_income(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the trajectories of wealth mean and variance for the different agent groups
    create two plots: one of mean-vs-variance and one with separate mean-variance plots
    '''
    plots = inp_base['agents']['land_area_init']
    N = len(plots)
    fig = plt.figure(figsize=(5*N,10))
    axs = {1 : [], 2 : [], 3 : []}
    lims = {1 : [9999999,-999999], 2 : [9999999,-999999], 3 : [9999999,-999999]}

    for n, land_area in enumerate(plots):
        ax1 = fig.add_subplot(3,N,n+1)
        ax2 = fig.add_subplot(3,N,N+n+1)
        ax3 = fig.add_subplot(3,N,2*N+n+1)

        for scenario, mods_sc in mods.items():
            ## calculate the wealth mean and std dev over time
            # extract and format the wealth info
            wlth = mods_sc['wealth']
            inc = mods_sc['income']
            all_wealth = []
            all_income = []
            for r in range(nreps):
                agents = mods_sc['land_area'][r] == land_area
                all_wealth.append(list(wlth[r,:,agents]))
                all_income.append(list(inc[r,:,agents]))
            all_wealth = np.array([item for sublist in all_wealth for item in sublist])
            all_income = np.array([item for sublist in all_income for item in sublist])
            # ^ this is (agents, time) shape
            # extract mean and variance
            mean_wlth = np.nanmean(all_wealth, axis=0)
            mean_inc = np.nanmean(all_income, axis=0)
            var_inc = np.var(all_income, axis=0)

            ## create the plot
            if scenario == 'baseline':
                mean_base_wlth = mean_wlth
                mean_base_inc = mean_inc
                var_base_inc = var_inc
            elif scenario in ['insurance','cover_crop']:
                plt_mean_wlth = mean_wlth - mean_base_wlth
                plt_mean_inc = mean_inc - mean_base_inc
                plt_var_inc = var_base_inc - var_inc

                ax1.plot(plt_mean_wlth, label=scenario)
                ax2.plot(plt_mean_inc, label=scenario)
                ax3.plot(plt_var_inc, label=scenario)

        axs[1].append(ax1)
        axs[2].append(ax2)
        axs[3].append(ax3)
        # plot-specific formatting
        if n == 0:
            ax1.set_ylabel('Change in wealth mean')
            ax2.set_ylabel('Change in income mean')
            ax3.set_ylabel('Change in income variance')
            top_axs = [ax1,ax2,ax3]
        ax1.set_title('{} ha'.format(land_area))
        # limits
        ax_tmp = [ax1,ax2,ax3]
        for k, v in lims.items():
            v[0] = min(v[0], ax_tmp[k-1].get_ylim()[0])
            v[1] = max(v[1], ax_tmp[k-1].get_ylim()[1])

    ## formatting of plots
    for k,v in axs.items():
        for i, vi in enumerate(v):
            vi.grid(False)
            vi.axhline(y=0, color='k')
            vi.set_ylim(lims[k])

            if k != 3:
                vi.set_xticklabels([])
            if i != 0:
                vi.set_yticklabels([])
            if k == 1:
                vi.legend()
            if k == 3:
                vi.set_xlabel('Time (years)')

    fig.tight_layout()
    fig.savefig(savedir + 'combined_wealth_income.png')
    plt.close('all')

def agent_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir, plt_type):
    '''
    plot the trajectories of wealth mean and variance for the different agent groups
    create two plots: one of mean-vs-variance and one with separate mean-variance plots
    '''
    for n, land_area in enumerate(inp_base['agents']['land_area_init']):
        fig, ax = plt.subplots(figsize=(8,8))
        fig2 = plt.figure(figsize=(7,10))
        ax1 = fig2.add_subplot(211)
        ax2 = fig2.add_subplot(212)
        for scenario, mods_sc in mods.items():
            ## calculate the wealth mean and std dev over time
            # extract and format the wealth info
            w = mods_sc[plt_type]
            all_wealth = []
            for r in range(nreps):
                agents = mods_sc['land_area'][r] == land_area
                all_wealth.append(list(w[r,:,agents]))
            all_wealth = np.array([item for sublist in all_wealth for item in sublist])
            # ^ this is (agents, time) shape
            # extract mean and variance
            mean_t = np.nanmean(all_wealth, axis=0)
            var_t = np.var(all_wealth, axis=0)

            ## create the plot
            if scenario == 'baseline':
                mean_base = mean_t
                var_base = var_t
            elif scenario in ['insurance','cover_crop']:
                plt_mean = mean_t - mean_base
                plt_var = var_base - var_t
                ax.plot(plt_mean, plt_var, label=scenario)#, marker='o')
                ax1.plot(plt_mean, label=scenario)
                ax2.plot(-plt_var, label=scenario) # NOTE: VARIANCE INCREASE IS UP (OPPOSITE TO AX)
                for t in range(T-1):
                    # add time labels
                    if ((t % 5 == 0) and (t<30)) or (t % 100 == 0):
                        ax.text(plt_mean[t], plt_var[t], str(t))
                        # # add an arrow
                        # try:
                        #     ax.arrow(plt_mean[t], plt_var[t], plt_mean[t+1]-plt_mean[t], plt_var[t+1]-plt_var[t],
                        #         lw=0, length_includes_head=True, 
                        #         head_width=max(np.abs(plt_mean))/25, head_length=max(np.abs(plt_var))/10) 
                        # except:
                        #     pass
                # ax.quiver(plt_mean[:-1], plt_var[:-1],plt_mean[1:]-plt_mean[:-1], plt_var[1:]-plt_var[:-1], angles='xy', units='width', pivot='mid', lw=0)#, scale=1000000000)

        ## formatting of mean-vs-variance plot
        ax.legend(loc='center left')
        ax.grid(False)
        ax.set_xlabel('Increase in mean {}'.format(plt_type))
        ax.set_ylabel('Decrease in {} variance'.format(plt_type))
        ax.set_title('{}, {} plots'.format(plt_type, nplot))
        ax.axhline(y=0, color='k', ls=':')
        ax.axvline(x=0, color='k', ls=':')
        xval = max(np.abs(ax.get_xlim()))
        yval = max(np.abs(ax.get_ylim()))
        ax.set_xlim([-xval, xval])
        ax.set_ylim([-yval, yval])
        ax.text(xval, yval, 'SYNERGY', fontsize=20, ha='right', va='top')
        ax.text(-xval, -yval, 'MALADAPTATION', fontsize=20, ha='left', va='bottom')
        ax.text(xval, -yval, 'DESTABILIZING,\nHIGHER MEAN', fontsize=20, ha='right', va='bottom')
        ax.text(-xval, yval, 'STABILIZING,\nLOWER MEAN', fontsize=20, ha='left', va='top')
        fig.tight_layout()
        fig.savefig(savedir + '{}_trajectories_{}_ha.png'.format(plt_type, str(land_area).replace('.','_')))

        ## formatting of separate plots
        for axx in [ax1, ax2]:
            axx.legend()
            axx.set_xlabel('Time (years)')
            axx.grid(False)
            axx.axhline(y=0, color='k')
        ax1.set_ylabel('Change in {} mean'.format(plt_type))
        ax2.set_ylabel('Change in {} variance'.format(plt_type))
        ax1.set_title('{}, {} plots'.format(plt_type, nplot))
        fig2.tight_layout()
        fig2.savefig(savedir + 'timeseries_{}_trajectories_{}_ha.png'.format(plt_type, str(land_area).replace('.','_')))
        plt.close('all')

def first_round_plots(mods, nreps, inp_base, scenarios, exp_name, T, savedir, shock_years=[]):
    colors = ['b','r','k','g','y']
    fig = plt.figure(figsize=(18,6))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    fig2 = plt.figure(figsize=(18,6))
    ax2s = [fig2.add_subplot(131),fig2.add_subplot(132),fig2.add_subplot(133)]
    fig3 = plt.figure(figsize=(18,6))
    ax3s = [fig3.add_subplot(131),fig3.add_subplot(132),fig3.add_subplot(133)]
    fig4 = plt.figure(figsize=(18,6))
    ax4s = [fig4.add_subplot(131),fig4.add_subplot(132),fig4.add_subplot(133)]

    ii = 0
    for m, mod in mods.items():

        ## generate the plot data
        ## we want to take time-averages for each agent type
        out_dict = {'wealth' : np.full(T+1, np.nan), 'SOM' : np.full(T+1, np.nan), 'coping' : np.full(T, np.nan),
            'income' : np.full(T, np.nan)}
        ag_data = [copy.copy(out_dict), copy.copy(out_dict), copy.copy(out_dict)]
        for r in range(nreps):
            # find agent types
            ags = [mod['land_area'][r] == inp_base['agents']['land_area_init'][0],
                mod['land_area'][r] == inp_base['agents']['land_area_init'][1],
                mod['land_area'][r] == inp_base['agents']['land_area_init'][2]]

            # add their data to the dictionaries
            for a, ag in enumerate(ags):
                ag_data[a]['wealth'] = np.nanmean(np.array([ag_data[a]['wealth'], np.mean(mod['wealth'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['income'] = np.nanmean(np.array([ag_data[a]['income'], np.mean(mod['income'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['coping'] = np.nanmean(np.array([ag_data[a]['coping'], np.mean(mod['coping'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['SOM'] =    np.nanmean(np.array([ag_data[a]['SOM'], np.mean(mod['organic'][r][:,ag], axis=0)]), axis=0)

        ## PLOT ##
        for a, ag in enumerate(ags):
            axs[a].plot(ag_data[a]['wealth'], label=m, color=colors[ii])
            ax2s[a].plot(ag_data[a]['coping'], label=m, color=colors[ii])
            ax3s[a].plot(ag_data[a]['SOM'], label=m, color=colors[ii])
            ax4s[a].plot(ag_data[a]['income'], label=m, color=colors[ii])

        ii += 1
    
    # some formatting
    for a in range(3):
        axs[a].set_title('Wealth : agent type {}'.format(a+1))
        ax2s[a].set_title('Coping : agent type {}'.format(a+1))
        ax3s[a].set_title('SOM : agent type {}'.format(a+1))
        ax4s[a].set_title('Income : agent type {}'.format(a+1))
        axs[a].set_xlabel('Time (yrs)')
        ax2s[a].set_xlabel('Time (yrs)')
        ax3s[a].set_xlabel('Time (yrs)')
        axs[a].set_ylabel('Birr')
        ax2s[a].set_ylabel('P(coping rqd)')
        ax3s[a].set_ylabel('kg/ha')
        ax3s[a].set_ylabel('Birr')
        axs[a].axhline(y=0, color='k', ls=':')
        ax3s[a].axhline(y=0, color='k', ls=':')
        for axx in [axs[a], ax2s[a], ax3s[a], ax4s[a]]:
            axx.legend()
            axx.grid(False)
            axx.set_xlabel('Time (yrs)')
            # show the shock on the plot, if necessary
            if len(shock_years) > 0:
                for yr in shock_years:
                    axx.axvline(x=yr, color='k', ls=':')
                    axx.text(yr, axx.get_ylim()[0]+(axx.get_ylim()[1]-axx.get_ylim()[0])*0.9, 'SHOCK', ha='center', rotation=90)

    fig.savefig(savedir + 'type_wealth.png')
    fig2.savefig(savedir + 'type_coping.png')
    fig3.savefig(savedir + 'type_SOM.png')
    fig4.savefig(savedir + 'type_income.png')
    plt.close('all')
    # code.interact(local=dict(globals(), **locals()))