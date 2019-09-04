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

def main(mod, save=True):
    if save == True:
        savedir = '../outputs/{}/'.format(mod.exp_name)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        savedir = False

    qs = [1,5,50,95,99]
    inputs(mod, savedir)
    soil(mod, qs, savedir)
    yields(mod, qs, savedir)
    coping(mod, qs, savedir)
    n_plots(mod, savedir)

def inputs(mod, savedir):
    '''
    show distributions derived from model inputs
    '''
    fig = plt.figure(figsize=(12,4))
    alpha = 0.6

    # 1. Rainfall
    mu = mod.climate.rain_mu
    sd = mod.climate.rain_sd
    rndms = np.random.normal(mu, sd, 10000)
    rndms[rndms<0] = 0
    rndms[rndms>1] = 1
    ax1 = fig.add_subplot(131)
    ax1.hist(rndms, alpha=alpha)
    ax1.set_xlabel('Rainfall measure')
    ax1.set_ylabel('Frequency')

    # 2. land plots
    mu = mod.agents.land_mean
    rndms = np.random.poisson(mu, 1000)
    breaks = np.arange(0, rndms.max()+1)
    ax2 = fig.add_subplot(132)
    ax2.hist(rndms, breaks, alpha=alpha)
    ax2.set_xlabel('Number of plots')
    ax2.set_ylabel('Frequency')

    # 3. wealth
    mu = mod.agents.wealth_init_mean
    sd = mod.agents.wealth_init_sd
    rndms = np.random.normal(mu, sd, 10000)
    ax3 = fig.add_subplot(133)
    ax3.hist(rndms, alpha=alpha)
    ax3.set_xlabel('Initial wealth')
    ax3.set_ylabel('Frequency')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'inputs.png')

def soil(mod, qs, savedir):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    band_plot(mod.land.SOM, qs, ax1, 'Soil organic matter', [0,1])
    band_plot(mod.land.inorganic, qs, ax2, 'Inorganic', [0,1])

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'soil.png')

def yields(mod, qs, savedir):
    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    band_plot(mod.land.rf_factors, qs, ax1, 'Rainfall effect', [0,1])
    band_plot(mod.land.nutrient_factors, qs, ax2, 'Nutrient effect', [0,1])
    band_plot(mod.land.yields, qs, ax3, 'Crop yield')
    band_plot(mod.agents.crop_production, qs, ax4, 'Crop production')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'yields.png')

def coping(mod, qs, savedir):
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    frac = np.mean(mod.agents.coping_rqd, axis=1)
    frac_cant = np.mean(mod.agents.cant_cope, axis=1)
    xs = np.arange(mod.T)
    ax1.plot(xs, frac, marker='o')
    ax1.set_xlabel('t')
    ax1.set_title('Frac coping')
    ax1.set_ylim([0,1])
    ax2.plot(xs, frac_cant, marker='o')
    ax2.set_xlabel('t')
    ax2.set_title("Frac that can't cope")
    ax2.set_ylim([0,1])
    band_plot(mod.agents.wealth, qs, ax3, 'Wealth')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'coping.png')

def n_plots(mod, savedir):
    '''
    outcomes as a function of number of plots
    '''
    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    # 1. coping amount
    xs = np.arange(0, mod.agents.n_plots.max()+1)
    ys = np.full(xs.shape, np.nan)
    y2s = np.full(xs.shape, np.nan)
    y3s = np.full(xs.shape, np.nan)
    y4s = np.full(xs.shape, np.nan)
    ag_soil_quality = mod.land.land_to_agent(mod.land.SOM[-1], mod.agents.n_plots, mode='average')
    for i, x in enumerate(xs):
        ags = mod.agents.n_plots == x
        if np.sum(ags) > 0:
            ys[i] = np.mean(mod.agents.coping_rqd[:, ags])
            y2s[i] = np.mean(mod.agents.crop_production[:, ags])
            y3s[i] = np.mean(ag_soil_quality[ags])
            y4s[i] = np.mean(mod.agents.wealth[-1, ags])

    ax1.plot(xs, ys, marker='o')
    ax1.set_xlabel('Number of plots')
    ax1.set_title('Coping frequency')

    ax2.plot(xs, y2s, marker='o')
    ax2.set_xlabel('Number of plots')
    ax2.set_title('Crop production')

    ax3.plot(xs, y3s, marker='o')
    ax3.set_xlabel('Number of plots')
    ax3.set_title('Final SOM')

    ax4.plot(xs, y4s, marker='o')
    ax4.set_xlabel('Number of plots')
    ax4.set_title('Wealth')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'n_plots.png')

def band_plot(d, qs, ax, title, ylim=False):
    '''
    create a plot with median and std deviations
    '''
    color = 'b'
    vals = np.percentile(d, q=qs, axis=1)
    xs = np.arange(vals.shape[1])
    ax.plot(xs, vals[2], color=color, marker='o')
    ax.fill_between(xs, vals[0], vals[-1], alpha=0.2, color=color)
    ax.fill_between(xs, vals[1], vals[-2], alpha=0.4, color=color)
    ax.set_title(title)
    ax.set_xlabel('t')
    if not isinstance(ylim, bool):
        ax.set_ylim(ylim)
    # code.interact(local=dict(globals(), **locals()))
