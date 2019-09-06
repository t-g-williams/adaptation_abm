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

    soil_wealth(mods, savedir)
    n_plots(mods, savedir)
    coping(mods, savedir)

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

def n_plots(mods, savedir):
    '''
    effect of number of plots on outcomes
    '''
    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    axs = [ax1,ax2,ax3,ax4]

    for m, mod in mods.items():
        xs = np.arange(0, mod.agents.n_plots.max()+1)
        ys = np.full(xs.shape, np.nan)
        y2s = np.full(xs.shape, np.nan)
        y3s = np.full(xs.shape, np.nan)
        y4s = np.full(xs.shape, np.nan)
        ag_soil_quality = mod.land.land_to_agent(mod.land.organic[-1], mod.agents.n_plots, mode='average')
        for i, x in enumerate(xs):
            ags = mod.agents.n_plots == x
            if np.sum(ags) > 0:
                ys[i] = np.mean(mod.agents.coping_rqd[:, ags])
                y2s[i] = np.mean(mod.agents.crop_production[:, ags])
                y3s[i] = np.mean(ag_soil_quality[ags])
                y4s[i] = np.mean(mod.agents.wealth[-1, ags])

        ax1.plot(xs, ys, marker='o', label=m)
        ax2.plot(xs, y2s, marker='o', label=m)
        ax3.plot(xs, y3s, marker='o', label=m)
        ax4.plot(xs, y4s, marker='o', label=m)
    
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Number of plots')
    ax1.set_title('Coping frequency')
    ax2.set_xlabel('Number of plots')
    ax2.set_title('Crop production (kg)')
    ax3.set_xlabel('Number of plots')
    ax3.set_title('Final SOM (kg/ha)')
    ax4.set_xlabel('Number of plots')
    ax4.set_title('Wealth (birr)')

    for ax in axs:
        ax.grid(False)
        ax.legend()
    
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'n_plots.png')


    
    

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