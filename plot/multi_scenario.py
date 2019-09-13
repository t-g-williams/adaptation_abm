import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
import copy
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

    colors = ['b','r','k','g','y']
    fig = plt.figure(figsize=(12,4))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    fig2 = plt.figure(figsize=(12,4))
    ax2s = [fig2.add_subplot(131),fig2.add_subplot(132),fig2.add_subplot(133)]
    fig3 = plt.figure(figsize=(12,4))
    ax3s = [fig3.add_subplot(131),fig3.add_subplot(132),fig3.add_subplot(133)]

    ii = 0
    for m, mods in mods.items():
        ## generate the plot data
        ## we want to take time-averages for each agent type
        out_dict = {'wealth' : np.full(T+1, np.nan), 'SOM' : np.full(T+1, np.nan), 'coping' : np.full(T, np.nan)}
        ag_data = [copy.copy(out_dict), copy.copy(out_dict), copy.copy(out_dict)]
        for r, mod in enumerate(mods):
            # find agent types
            ags = [mod.agents.n_plots == mod.agents.n_plots_init[0],
                mod.agents.n_plots == mod.agents.n_plots_init[1],
                mod.agents.n_plots == mod.agents.n_plots_init[2]]

            # add their data to the dictionaries
            for a, ag in enumerate(ags):
                ag_data[a]['wealth'] = np.nanmean(np.array([ag_data[a]['wealth'], np.mean(mod.agents.wealth[:,ag], axis=1)]), axis=0)
                ag_data[a]['coping'] = np.nanmean(np.array([ag_data[a]['coping'], np.mean(mod.agents.coping_rqd[:,ag], axis=1)]), axis=0)
                # find the land-level agent types
                lan = np.in1d(mod.land.owner, mod.agents.id[ag])
                ag_data[a]['SOM'] = np.nanmean(np.array([ag_data[a]['SOM'], np.mean(mod.land.organic[:,lan], axis=1)]), axis=0)


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