import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
from . import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

def main(mods, save=True):
    if save == True:
        savedir = '../outputs/{}/'.format(mod.exp_name)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        savedir = False

    agent_type_plots(mods, savedir)

    # soil_wealth(mods, savedir)
    # coping(mods, savedir)

def agent_type_plots(mods, savedir):
    '''
    plot each agent type (land area) separately
    this assumes there's 3 agent types
    '''
    colors = ['b','r','k','g','y']
    fig = plt.figure(figsize=(18,6))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    fig2 = plt.figure(figsize=(12,4))
    ax2s = [fig2.add_subplot(131),fig2.add_subplot(132),fig2.add_subplot(133)]
    fig3 = plt.figure(figsize=(12,4))
    ax3s = [fig3.add_subplot(131),fig3.add_subplot(132),fig3.add_subplot(133)]
    fig4 = plt.figure(figsize=(12,20))
    ax4s = [fig4.add_subplot(311),fig4.add_subplot(312),fig4.add_subplot(313)]
    fig5 = plt.figure(figsize=(12,20))
    ax5s = [fig5.add_subplot(311),fig5.add_subplot(312),fig5.add_subplot(313)]
    ii = 0
    shock = False
    for m, mod in mods.items():
        if mod.shock:
            shock = True
            shock_years = mod.climate.shock_years

        # find agent types
        ags = [mod.agents.land_area == mod.agents.land_area_init[0],
            mod.agents.land_area == mod.agents.land_area_init[1],
            mod.agents.land_area == mod.agents.land_area_init[2]]

        for a, ag in enumerate(ags):
            axs[a].plot(np.median(mod.agents.wealth[:,ag], axis=1), label=m, color=colors[ii])
            ax2s[a].plot(np.mean(mod.agents.coping_rqd[:,ag], axis=1), label=m, color=colors[ii])
            ax3s[a].plot(np.median(mod.land.organic[:,ag], axis=1), label=m, color=colors[ii])
            ax4s[a].plot(mod.agents.wealth[:,ag], color=colors[ii])#, lw=0.5)
            ax5s[a].plot(mod.land.organic[:,ag], color=colors[ii])#, lw=0.5)

        ii += 1

    # some formatting
    for a in range(3):
        axs[a].set_title('Wealth : agent type {}'.format(a+1))
        ax4s[a].set_title('Wealth : agent type {}'.format(a+1))
        ax2s[a].set_title('Coping : agent type {}'.format(a+1))
        ax3s[a].set_title('SOM : agent type {}'.format(a+1))
        ax5s[a].set_title('SOM : agent type {}'.format(a+1))
        axs[a].set_ylabel('Birr')
        ax4s[a].set_ylabel('Birr')
        ax2s[a].set_ylabel('P(coping rqd)')
        ax3s[a].set_ylabel('kg/ha')
        ax5s[a].set_ylabel('kg/ha')
        axs[a].legend()
        ax2s[a].legend()
        ax3s[a].legend()
        axs[a].axhline(y=0, color='k', ls=':')
        ax4s[a].axhline(y=0, color='k', ls=':')
        ax3s[a].axhline(y=0, color='k', ls=':')
        ax5s[a].axhline(y=0, color='k', ls=':')
        for axx in [axs[a], ax2s[a], ax3s[a], ax4s[a], ax5s[a]]:
            axx.grid(False)
            axx.set_xlabel('Time (yrs)')
            # show the shock on the plot, if necessary
            if shock:
                for yr in shock_years:
                    axx.axvline(x=yr, color='k', ls=':')
                    axx.text(yr, axx.get_ylim()[0]+(axx.get_ylim()[1]-axx.get_ylim()[0])*0.1, 'SHOCK', ha='center', va='bottom', rotation=90)

    if isinstance(savedir, bool):
        # return fig, fig2, fig3, fig4, fig5
        return fig4, fig5
    else:
        fig.savefig(savedir + 'type_wealth.png')
        fig4.savefig(savedir + 'type_wealth_all.png')
        fig2.savefig(savedir + 'type_coping.png')
        fig3.savefig(savedir + 'type_SOM.png')
        fig5.savefig(savedir + 'type_SOM_all.png')


def soil_wealth(mods, savedir):
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axs = [ax1, ax2, ax3]

    for m, mod in mods.items():
        ax1.plot(np.mean(mod.land.organic, axis=1), label=m)
        ax2.plot(np.mean(mod.land.inorganic, axis=1), label=m)
        ax3.plot(np.mean(mod.agents.wealth, axis=1), label=m)

    ax1.set_title('Organic N')
    ax2.set_title('Inorganic N')
    ax3.set_title('Wealth')
    for ax in axs:
        ax.set_xlabel('Time (yrs)')
        ax.legend()
        ax.grid(False)

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'soil_wealth.png')

def coping(mods, savedir):
    '''
    plot coping and adaptation
    '''
    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axs = [ax1, ax2, ax3]

    for m, mod in mods.items():
        frac = np.mean(mod.agents.coping_rqd, axis=1)
        frac_cant = np.mean(mod.agents.cant_cope, axis=1)
        xs = np.arange(mod.T)
        ax1.plot(xs, frac, label=m)
        ax2.plot(xs, frac_cant, label=m)

        # adaptation
        adap = mod.agents.adapt
        x1s = np.arange(adap.shape[0])
        fracs = np.mean(adap, axis=1)
        ax3.plot(x1s, fracs, label=m)

    ax1.set_title('Frac coping')
    ax2.set_title("Frac that can't cope")
    ax3.set_title('Fraction of population adapting')

    for ax in axs:
        ax.set_ylim([0,1])
        ax.set_xlabel('Time (yrs)')
        ax.legend()
        ax.grid(False)

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'adap_coping.png')