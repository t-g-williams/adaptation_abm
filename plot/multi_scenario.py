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

def main(mods, nreps, inp_base, scenarios, exp_name, T):
    '''
    plot each agent type (number of plots) separately
    this assumes there's 3 agent types
    '''
    savedir = '../outputs/{}/'.format(exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # first_round_plots(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
    wealth_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir)
    coping_probabilities(mods, nreps, inp_base, scenarios, exp_name, T, savedir)

def wealth_trajectories(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    '''
    plot the trajectories of wealth mean and variance for the different agent groups
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
            mean_t = np.mean(all_wealth, axis=0)
            var_t = np.var(all_wealth, axis=0)

            ## create the plot
            if scenario == 'baseline':
                mean_base = mean_t
                var_base = var_t
            elif scenario in ['insurance','cover_crop']:
                plt_mean = mean_t - mean_base
                plt_var = var_base - var_t
                ax.plot(plt_mean, plt_var, label=scenario)#, marker='o')
                for t in range(T-1):
                    # add time labels
                    if (t % 5 == 0) and (t<30):
                        ax.text(plt_mean[t], plt_var[t], str(t))
                        # # add an arrow
                        # try:
                        #     ax.arrow(plt_mean[t], plt_var[t], plt_mean[t+1]-plt_mean[t], plt_var[t+1]-plt_var[t],
                        #         lw=0, length_includes_head=True, 
                        #         head_width=max(np.abs(plt_mean))/25, head_length=max(np.abs(plt_var))/10) 
                        # except:
                        #     pass
                # ax.quiver(plt_mean[:-1], plt_var[:-1],plt_mean[1:]-plt_mean[:-1], plt_var[1:]-plt_var[:-1], angles='xy', units='width', pivot='mid', lw=0)#, scale=1000000000)

        ax.legend(loc='center left')
        ax.grid(False)
        ax.set_xlabel('Increase in mean wealth')
        ax.set_ylabel('Decrease in wealth variance')
        ax.set_title('Wealth trajectories')
        ax.axhline(y=0, color='k', ls=':')
        ax.axvline(x=0, color='k', ls=':')
        xval = max(np.abs(ax.get_xlim()))
        yval = max(np.abs(ax.get_ylim()))
        ax.set_xlim([-xval, xval])
        ax.set_ylim([-yval, yval])
        ax.text(xval, yval, 'SYNERGY', fontsize=22, ha='right', va='top')
        ax.text(-xval, -yval, 'MALADAPTATION', fontsize=22, ha='left', va='bottom')
        ax.text(xval, -yval, 'POVERTY REDUCTION', fontsize=22, ha='right', va='bottom')
        ax.text(-xval, yval, 'STABILIZING', fontsize=22, ha='left', va='top')
        fig.tight_layout()
        fig.savefig(savedir + 'wealth_trajectories_{}_plots.png'.format(nplot))


def first_round_plots(mods, nreps, inp_base, scenarios, exp_name, T, savedir):
    colors = ['b','r','k','g','y']
    fig = plt.figure(figsize=(12,4))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    fig2 = plt.figure(figsize=(12,4))
    ax2s = [fig2.add_subplot(131),fig2.add_subplot(132),fig2.add_subplot(133)]
    fig3 = plt.figure(figsize=(12,4))
    ax3s = [fig3.add_subplot(131),fig3.add_subplot(132),fig3.add_subplot(133)]

    ii = 0
    for m, mod in mods.items():

        ## generate the plot data
        ## we want to take time-averages for each agent type
        out_dict = {'wealth' : np.full(T+1, np.nan), 'SOM' : np.full(T+1, np.nan), 'coping' : np.full(T, np.nan)}
        ag_data = [copy.copy(out_dict), copy.copy(out_dict), copy.copy(out_dict)]
        for r in range(nreps):
            # find agent types
            ags = [mod['n_plots'][r] == inp_base['agents']['n_plots_init'][0],
                mod['n_plots'][r] == inp_base['agents']['n_plots_init'][1],
                mod['n_plots'][r] == inp_base['agents']['n_plots_init'][2]]

            # add their data to the dictionaries
            for a, ag in enumerate(ags):
                ag_data[a]['wealth'] = np.nanmean(np.array([ag_data[a]['wealth'], np.mean(mod['wealth'][r,:,ag], axis=0)]), axis=0)
                ag_data[a]['coping'] = np.nanmean(np.array([ag_data[a]['coping'], np.mean(mod['coping'][r,:,ag], axis=0)]), axis=0)
                # find the land-level agent types
                lan = np.in1d(mod['owners'][r], np.arange(ag.shape[0])[ag])
                ag_data[a]['SOM'] = np.nanmean(np.array([ag_data[a]['SOM'], np.mean(mod['organic'][r][:,lan], axis=1)]), axis=0)


        ## PLOT ##
        for a, ag in enumerate(ags):
            axs[a].plot(ag_data[a]['wealth'], label=m, color=colors[ii])
            ax2s[a].plot(ag_data[a]['coping'], label=m, color=colors[ii])
            ax3s[a].plot(ag_data[a]['SOM'], label=m, color=colors[ii])

        ii += 1
    # some formatting
    for a in range(3):
        axs[a].set_title('Wealth : agent type {}'.format(a+1))
        ax2s[a].set_title('Coping : agent type {}'.format(a+1))
        ax3s[a].set_title('SOM : agent type {}'.format(a+1))
        axs[a].set_xlabel('Time (yrs)')
        ax2s[a].set_xlabel('Time (yrs)')
        ax3s[a].set_xlabel('Time (yrs)')
        axs[a].set_ylabel('Birr')
        ax2s[a].set_ylabel('P(coping rqd)')
        ax3s[a].set_ylabel('kg/ha')
        axs[a].legend()
        ax2s[a].legend()
        ax3s[a].legend()
        axs[a].axhline(y=0, color='k', ls=':')
        axs[a].grid(False)
        ax2s[a].grid(False)
        ax3s[a].grid(False)

    fig.savefig(savedir + 'type_wealth.png')
    fig2.savefig(savedir + 'type_coping.png')
    fig3.savefig(savedir + 'type_SOM.png')
    # code.interact(local=dict(globals(), **locals()))