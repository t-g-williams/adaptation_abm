'''
plot livelihood distributions for a single model run
'''

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

    ext = '_' + mod.exp_name.split('/')[-1] if '/' in mod.exp_name else ''
    tplot = 15

    rangeland(mod, savedir, ext)
    time_trajectories(mod, savedir, ext)
    income_dists(mod, savedir, tplot, ext)
    labor_dists(mod, savedir, tplot, ext)
    plt.close('all')

    
def rangeland(m, savedir, ext):
    ## plot the rangeland states
    fig, axs = plt.subplots(3,1,figsize=(12,10),sharex=True, gridspec_kw={'height_ratios':[1,0.15,0.15]})
    ax = axs[0]
    # reserve biomass
    ax.fill_between(np.arange(m.T+1), 0, m.rangeland.R, color='red', label='Reserve')
    # format the green biomass
    G = np.full([2*m.T], np.nan)
    G[np.arange(m.T)*2] = m.rangeland.G_no_cons
    G[np.arange(m.T)*2+1] = m.rangeland.G[:-1]
    mids = (m.rangeland.R[:-1] + m.rangeland.R[1:]) / 2
    R_all = np.full(2*m.T, np.nan)
    R_all[np.arange(m.T)*2] = m.rangeland.R[:-1]
    R_all[np.arange(m.T)*2+1] = mids

    R_base = np.repeat(m.rangeland.R[:-1],2)
    ax.fill_between(np.arange(2*m.T)/2, R_all, R_all+G, color='green', label='Green')

    ax.grid(False)
    ax.set_ylabel('Biomass (kg/ha)')
    ax.legend()
    ax.set_title('Rangeland dynamics')

    # add rainfall and livestock total
    axs[1].plot(m.climate.rain, color='k')
    axs[1].grid(False)
    axs[1].set_ylabel('Climate\ncondition')
    axs[1].axhline(0, color='k', lw=0.5)
    axs[1].axhline(1, color='k', lw=0.5)

    axs[2].plot(m.rangeland.livestock_supported, color='k')
    axs[2].grid(False)
    axs[2].set_ylabel('Rangeland\nlivestock')
    axs[2].set_xlabel('Year')
    axs[2].axhline(0, color='k',lw=0.5)

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'rangeland{}.png'.format(ext))

def time_trajectories(m, savedir, ext):
    fig, axs = plt.subplots(2,3,figsize=(16,8))
    axs = axs.flatten()

    axs[0].plot(m.agents.wealth, color='k', lw=0.5)
    axs[0].plot(np.median(m.agents.wealth, axis=1), color='b', lw=4)
    axs[0].set_ylabel('Total wealth')
    axs[0].set_title('Wealth')

    axs[1].plot(m.agents.livestock, color='k', lw=0.5)
    axs[1].plot(np.median(m.agents.livestock, axis=1), color='b', lw=4)
    axs[1].set_ylabel('E[Herd size]')
    axs[1].set_title('Livestock')
    axs[1].boxplot(m.agents.livestock[-1],positions=[m.T+m.T/10], widths=[int(m.T/10)])

    axs[2].plot(m.land.organic, color='k', lw=0.5)
    axs[2].plot(np.median(m.land.organic, axis=1), color='b', lw=4)
    axs[2].set_ylabel('kg N/ha')
    axs[2].set_title('Soil organic matter')

    axs[3].plot(m.agents.wage_labor, color='k', lw=0.5)
    axs[3].plot(np.median(m.agents.wage_labor, axis=1), color='b', lw=4)
    axs[3].set_ylabel('ppl')
    axs[3].set_title('Wage labor')

    axs[4].plot(m.agents.salary_labor, color='k', lw=0.5)
    axs[4].plot(np.median(m.agents.salary_labor, axis=1), color='b', lw=4)
    axs[4].set_ylabel('ppl')
    axs[4].set_title('Salary labor')

    for ax in axs:
        ax.grid(False)
        ax.set_xlabel('Year')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'trajectories{}.png'.format(ext))

def income_dists(m, savedir, tplot, ext):
    ## agent income distribution
    bins = np.arange(0,1.1,0.1)
    fig, axs = plt.subplots(2,4,figsize=(15,8))
    objs = [m.agents.farm_income, m.agents.ls_income, m.agents.salary_income, m.agents.wage_income]
    tot_income = np.sum(np.array(objs), axis=0)
    titles = ['Farm income','Livestock income','Salary income', 'Wage income']
    for o, obj in enumerate(objs):
        axs[0,o].plot(obj/tot_income, color='k',lw=0.5)
        axs[0,o].plot(np.nanmedian(obj/tot_income, axis=1), color='b',lw=4)
        axs[0,o].set_title(titles[o] + ' fraction')
        axs[0,o].set_xlabel('Year')
        axs[0,o].grid(False)
        
        axs[1,o].hist((obj/tot_income)[tplot], bins=bins, color='0.5', edgecolor='k', lw=1.5)
        axs[1,o].set_title(titles[o] + ' fraction')
        axs[1,o].set_xlabel('Year')
        axs[1,o].grid(False)
    axs[1,0].set_ylabel('Frequency')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'income_dists{}.png'.format(ext))

def labor_dists(m, savedir, tplot, ext):
    ## agent labor distribution
    bins = np.arange(0,1.1,0.1)
    fig, axs = plt.subplots(2,4,figsize=(15,8))
    objs = [m.agents.ag_labor, m.agents.ls_labor, m.agents.salary_labor, m.agents.wage_labor]
    tot_labor = np.sum(np.array(objs), axis=0)
    titles = ['Farm labor','Livestock labor','Salary labor', 'Wage labor']
    for o, obj in enumerate(objs):
        axs[0,o].plot(obj/tot_labor, color='k',lw=0.5)
        axs[0,o].plot(np.nanmedian(obj/tot_labor, axis=1), color='b',lw=4)
        axs[0,o].set_title(titles[o] + ' fraction')
        axs[0,o].set_xlabel('Year')
        axs[0,o].grid(False)
        
        axs[1,o].hist((obj/tot_labor)[tplot], bins=bins, color='0.5', edgecolor='k', lw=1.5)
        axs[1,o].set_title(titles[o] + ' fraction')
        axs[1,o].set_xlabel('Year')
        axs[1,o].grid(False)
    axs[1,0].set_ylabel('Frequency')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'labor_dists{}.png'.format(ext))