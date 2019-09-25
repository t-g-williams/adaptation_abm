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

def main(mods, nreps, inp_base, scenarios, exp_name, T, shock_years=[]):
    '''
    plot each agent type (number of plots) separately
    this assumes there's 3 agent types
    '''
    savedir = '../outputs/{}/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if len(shock_years) == 0:
        # only run these for the adaptation scenarios -- this assumes the length of shock years here is zero
        combined_wealth_income(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        neg_wealth_probabilities(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
        agent_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir, 'wealth')
        agent_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir, 'income')
    
    first_round_plots(mods, nreps, inp_base, scenarios, exp_name, T, savedir, shock_years)

def neg_wealth_probabilities(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the probability that each agents' wealth is below zero over time
    '''
    for n, nplot in enumerate(inp_base['agents']['n_plots_init']):
        fig, ax = plt.subplots(figsize=(8,8))
        for scenario, mods_sc in mods.items():
            ## calculate the wealth mean and std dev over time
            # extract and format the wealth info
            w = mods_sc['wealth']
            all_wealth = []
            for r in range(nreps):
                agents = mods_sc['n_plots'][r] == nplot
                all_wealth.append(list(w[r,:,agents]))
            all_wealth = np.array([item for sublist in all_wealth for item in sublist])
            # ^ this is (agents, time) shape
            # extract mean and variance
            probs_t = np.mean(all_wealth<=0, axis=0)

            ax.plot(np.arange(T+1), probs_t, label=scenario)#, marker='o')

        ax.legend(loc='center left')
        ax.grid(False)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('P(wealth < 0)')
        ax.set_title('Agent wealth')
        fig.tight_layout()
        fig.savefig(savedir + 'neg_wealth_prob_{}_plots.png'.format(nplot))
        plt.close('all')

def combined_wealth_income(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the trajectories of wealth mean and variance for the different agent groups
    create two plots: one of mean-vs-variance and one with separate mean-variance plots
    '''
    plots = inp_base['agents']['n_plots_init']
    N = len(plots)
    fig = plt.figure(figsize=(5*N,10))
    axs = {1 : [], 2 : [], 3 : []}
    lims = {1 : [9999999,-999999], 2 : [9999999,-999999], 3 : [9999999,-999999]}

    for n, nplot in enumerate(plots):
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
                agents = mods_sc['n_plots'][r] == nplot
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
        ax1.set_title('{} plots'.format(nplot))
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
    for n, nplot in enumerate(inp_base['agents']['n_plots_init']):
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
                agents = mods_sc['n_plots'][r] == nplot
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
        fig.savefig(savedir + '{}_trajectories_{}_plots.png'.format(plt_type, nplot))

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
        fig2.savefig(savedir + 'timeseries_{}_trajectories_{}_plots.png'.format(plt_type, nplot))
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
            ags = [mod['n_plots'][r] == inp_base['agents']['n_plots_init'][0],
                mod['n_plots'][r] == inp_base['agents']['n_plots_init'][1],
                mod['n_plots'][r] == inp_base['agents']['n_plots_init'][2]]

            # add their data to the dictionaries
            for a, ag in enumerate(ags):
                ag_data[a]['wealth'] = np.nanmean(np.array([ag_data[a]['wealth'], np.mean(mod['wealth'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['income'] = np.nanmean(np.array([ag_data[a]['income'], np.mean(mod['income'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['coping'] = np.nanmean(np.array([ag_data[a]['coping'], np.mean(mod['coping'][r,:,ag], axis=0)]), axis=0)
                # find the land-level agent types
                lan = np.in1d(mod['owners'][r], np.arange(ag.shape[0])[ag])
                ag_data[a]['SOM'] = np.nanmean(np.array([ag_data[a]['SOM'], np.mean(mod['organic'][r][:,lan], axis=1)]), axis=0)

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