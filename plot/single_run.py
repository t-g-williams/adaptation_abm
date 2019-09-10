import matplotlib as mpl
# mpl.use('Agg')
import scipy.stats as stat
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

    # plotting by agent type
    type_wealth(mod, savedir)
    type_coping(mod, savedir)
    type_nutrients(mod, savedir)
    type_yields(mod, savedir)

    # band and original plots
    # soil(mod, qs, savedir)
    # yields(mod, qs, savedir)
    # coping(mod, qs, savedir)
    # n_plots(mod, savedir)
    adaptation(mod, savedir)
    

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
    # rndm_area = stat.lognorm.rvs(mod.agents.land_s, loc=mod.agents.land_loc, scale=mod.agents.land_scale, size=1000)
    # rndm_area[rndm_area>mod.agents.land_max] = mod.agents.land_max
    # rndm_plots = np.round(rndm_area / mod.land.area).astype(int)
    # rndm_plots[rndm_plots <= 0] = 1
    rndms = np.random.choice(mod.agents.n_plots_init, size=1000)
    # breaks = np.arange(0, rndms.max()+1)
    ax2 = fig.add_subplot(132)
    ax2.hist(rndms, alpha=alpha)
    ax2.set_xlabel('Number of plots')
    ax2.set_ylabel('Frequency')

    # 3. wealth
    mu = mod.agents.wealth_init_mean
    sd = mod.agents.wealth_init_sd
    rndms = np.random.normal(mu, sd, 10000)
    ax3 = fig.add_subplot(133)
    ax3.hist(rndms, alpha=alpha)
    ax3.set_xlabel('Initial wealth (birr)')
    ax3.set_ylabel('Frequency')
    fig.tight_layout()

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'inputs.png')

def soil(mod, qs, savedir):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    band_plot(mod.land.organic, qs, ax1, 'Organic N (kg/ha)')
    band_plot(mod.land.inorganic, qs, ax2, 'Inorganic N (kg/ha)')
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax2.set_ylim([0, ax2.get_ylim()[1]])

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
    band_plot(mod.land.yields, qs, ax3, 'Crop yield (kg/ha)')
    band_plot(mod.agents.crop_production, qs, ax4, 'Crop production (kg)')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'yields.png')

def coping(mod, qs, savedir):
    fig = plt.figure(figsize=(18,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

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
    band_plot(mod.agents.wealth, qs, ax3, 'Wealth (birr)')

    # trajectories
    agent_trajectories(mod.agents.coping_rqd, ax4)
    ax4.set_xlabel("No coping rqd (cumsum)")
    ax4.set_ylabel('Coping rqd (cumsum)')
    ax4.set_title('Agent coping trajectories')

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
    ag_soil_quality = mod.land.land_to_agent(mod.land.organic[-1], mod.agents.n_plots, mode='average')
    for i, x in enumerate(xs):
        ags = mod.agents.n_plots == x
        if np.sum(ags) > 0:
            ys[i] = np.mean(mod.agents.coping_rqd[:, ags])
            y2s[i] = np.mean(mod.agents.crop_production[:, ags])
            y3s[i] = np.mean(ag_soil_quality[ags])
            y4s[i] = np.mean(mod.agents.wealth[-1, ags])

    ax1.plot(xs, ys, marker='o')
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Number of plots')
    ax1.set_title('Coping frequency')

    ax2.plot(xs, y2s, marker='o')
    ax2.set_xlabel('Number of plots')
    ax2.set_title('Crop production (kg)')

    ax3.plot(xs, y3s, marker='o')
    ax3.set_xlabel('Number of plots')
    ax3.set_title('Final SOM (kg/ha)')

    ax4.plot(xs, y4s, marker='o')
    ax4.set_xlabel('Number of plots')
    ax4.set_title('Wealth (birr)')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'n_plots.png')

def adaptation(mod, savedir):
    '''
    plot regional and agent-level adaptation trajectories
    '''
    adap = mod.agents.adapt
    fig = plt.figure(figsize=(18,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # 1. regional-level adoption
    x1s = np.arange(adap.shape[0])
    fracs = np.mean(adap, axis=1)
    ax1.plot(x1s, fracs, marker='o')
    ax1.set_ylim([0,1])
    ax1.set_title('Fraction of population adapting')
    ax1.set_xlabel('time (years)')
    ax1.grid(False)

    # 2. agent-level trajectories
    agent_trajectories(adap, ax2)
    ax2.set_xlabel('NOT adapt (cumsum)')
    ax2.set_ylabel('Adapt (cumsum)')
    ax2.set_title('Agent-level adaptation trajectories')

    # 3. population-level transitions
    scale = 80
    scale2 = 5
    # lines
    for t in x1s[:-1]:
        ax3.plot([t,t+1], [0,0], lw=scale2*np.mean((adap[t]==False) & (adap[t+1]==False)), color='b')
        ax3.plot([t,t+1], [1,1], lw=scale2*np.mean((adap[t]==True) & (adap[t+1]==True)), color='b')
        ax3.plot([t,t+1], [0,1], lw=scale2*np.mean((adap[t]==False) & (adap[t+1]==True)), color='b')
        ax3.plot([t,t+1], [1,0], lw=scale2*np.mean((adap[t]==True) & (adap[t+1]==False)), color='b')
    # points
    ax3.scatter(x1s, np.full(x1s.shape, 0), s=(1-fracs)*scale, color='b')
    ax3.scatter(x1s, np.full(x1s.shape, 1), s=(fracs)*scale, color='b')
    ax3.set_xlabel('time (years)')
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['NOT adapt','Adapt'])
    ax3.grid(False)
    ax3.set_title('Population-level switches')

    fig.tight_layout()
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'adaptation.png')

def type_wealth(mod, savedir):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    type_timeseries(mod.agents.wealth, mod.agents.n_plots, ax, 'birr', 'Wealth')
    fig.tight_layout()
    ax.axhline(y=0, color='k')
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'single_wealth.png')

def type_coping(mod, savedir):
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    type_timeseries(mod.agents.coping_rqd, mod.agents.n_plots, ax, 'P(coping rqd)', 'Coping', mean=True)
    type_timeseries(mod.agents.cant_cope, mod.agents.n_plots, ax2, 'P(cant cope)', 'Not able to cope', mean=True)
    agent_trajectories(mod.agents.coping_rqd, ax3)
    ax3.set_xlabel("No coping rqd (cumsum)")
    ax3.set_ylabel('Coping rqd (cumsum)')
    ax3.set_title('Agent coping trajectories')

    fig.tight_layout()
    ax.axhline(y=0, color='k')
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'single_coping.png')

def type_nutrients(mod, savedir):
    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    type_timeseries(mod.land.organic, mod.agents.n_plots[mod.land.owner], ax, 'kg/ha', 'Organic N', mean=True)
    type_timeseries(mod.land.inorganic, mod.agents.n_plots[mod.land.owner], ax2, 'kg/ha', 'Inorganic N', mean=True)
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax2.set_ylim([0, ax2.get_ylim()[1]])
    fig.tight_layout()
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'single_soil.png')

def type_yields(mod, savedir):
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    type_timeseries(mod.land.rf_factors, mod.agents.n_plots[mod.land.owner], ax, '', 'Rainfall effect', mean=True)
    type_timeseries(mod.land.nutrient_factors, mod.agents.n_plots[mod.land.owner], ax2, '', 'Nutrient effect', mean=True)
    type_timeseries(mod.land.yields, mod.agents.n_plots[mod.land.owner], ax3, 'kg/ha', 'Crop yield', mean=True)
    fig.tight_layout()
    ax.set_ylim([0,1])
    ax2.set_ylim([0,1])
    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'single_yields.png')

def type_timeseries(d, n_plots, ax, ylab, title, mean=False):
    '''
    create a plot separating agents by their "type"
    which is given by their number of plots
    it assumes each agent has a value for "d" at each time step
    '''
    colors = ['k','b','r']
    uniqs = np.unique(n_plots)
    if len(uniqs) > 3:
        print('too many unique types!')
        return
    
    for ui, u in enumerate(uniqs):
        ixs = n_plots==u
        ix_nums = np.arange(n_plots.shape[0])[ixs]
        if mean:
            plt_d = np.nanmean(d[:,ixs], axis=1)
            ax.plot(plt_d, label='{} plots'.format(u), color=colors[ui])
        else:
            plt_d = d[:,ixs]
            for i, ix in enumerate(ix_nums):
                lgd = '{} plots'.format(u) if i == 0 else '_nolegend_'
                ax.plot(d[:,ix], label=lgd, color=colors[ui])

    ax.grid(False)
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (yrs)')
    ax.set_ylabel(ylab)
    ax.set_title(title)

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

def agent_trajectories(d, ax):
    '''
    plot of agent trajectories w.r.t. a binary object (d)
    '''
    xs = np.cumsum(~d, axis=0)
    ys = np.cumsum(d, axis=0)
    # add sums of agents in each final state
    finals = np.array([xs[-1], ys[-1]])
    ends = []
    counts = []
    # loop over agents
    for a in range(finals.shape[1]):
        val = list(finals[:,a])
        if val not in ends:
            ends.append(val)
            counts.append(1)
        else:
            counts[ends.index(val)] += 1
    # add to plot
    for i, en in enumerate(ends):
        ax.text(en[0], en[1]+0.3, str(counts[i]), horizontalalignment='center')
    # code.interact(local=dict(globals(), **locals()))
    ax.plot(xs, ys, color='b')
    mx = max(xs.max(), ys.max()) + 1.5
    ax.set_ylim([0, mx])
    ax.set_xlim([0, mx])
