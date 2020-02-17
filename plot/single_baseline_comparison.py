'''
compare model outputs to baseline
for a single simulation
'''
import matplotlib as mpl
# mpl.use('Agg')
import scipy.stats as stat
import numpy as np
import pandas as pd
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

def main(mods, exp_name, relative=False, save=True):
    if save == True:
        savedir = '../outputs/{}/'.format(exp_name)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        savedir = False

    area_comparison(mods, exp_name, savedir)
    timeline(mods, exp_name, savedir, relative)

def area_comparison(mods, exp_name, savedir):
    '''
    compare the land area under each scenario
    '''
    ## land composition before and after
    fig, ax = plt.subplots(1,1,figsize=(30,8))
    farm_pre = mods['baseline'].land.tot_area
    common_pre = mods['baseline'].rangeland.size_ha

    dt = {'Commons' : [common_pre], 'Farm (unaffected)' : [farm_pre], 'Farm (affected)' : [0], 
        'LSLA' : [0]}
    names = ['Baseline']
    for name, mod in mods.items():
        if name=='baseline':
            continue
        names.append(name)
        dt['Farm (unaffected)'].append(mod.agents.land_area[~mod.lsla.lost_land].sum())
        dt['Farm (affected)'].append(mod.agents.land_area[mod.lsla.lost_land].sum())
        dt['LSLA'].append(mod.lsla.area)
        dt['Commons'].append(mod.rangeland.size_ha)

    df = pd.DataFrame(dt, index=names)

    df.plot.bar(rot=0, ax=ax, stacked=True, color=['#8DC8F6','#3FAC00','#2C6908','#000000'])
    ax.grid(False)
    ax.set_ylabel('ha')
    ax.set_title('Area pre- and post-LSLA')
    ax.legend(bbox_to_anchor=[1,0.5], loc='center left')

    fig.savefig(savedir+'land_changes.png')
    plt.close('all')

    # code.interact(local=dict(globals(), **locals()))

def timeline(mods, exp_name, savedir, relative):
    fig, ax_all = plt.subplots(2,4,figsize=(6*4,7), gridspec_kw={'height_ratios':[1,0.05]})
    axs = ax_all[0]
    [axi.remove() for axi in ax_all[1,:]]
    n = len(mods)-1 if relative else len(mods)
    if n < 5:
        lss = [':','-.','--','-']
    else:
        lss = ['-']*n
    lw = 3
    pre = 'Change in ' if relative else ''
    titles = ['E[livestock]','E[income]','E[SOM]','rangeland biomass']
    ylabs = ['head','birr','kg N/ha','kg/ha']

    # get the baseline data
    baseline = {
        'livestock' : mods['baseline'].agents.livestock,
        'income' : mods['baseline'].agents.income,
        'SOM' : mods['baseline'].land.organic,
        'rangeland' : mods['baseline'].rangeland.R,
    }

    m=0
    plot_mods = []
    for name, mod in mods.items():
        if relative:
            if name == 'baseline':
                continue
            axs[0].plot(np.mean(mod.agents.livestock - baseline['livestock'], axis=1), lw=lw, label=name, ls=lss[m])
            axs[1].plot(np.mean(mod.agents.income - baseline['income'], axis=1), lw=lw, label=name, ls=lss[m])
            axs[2].plot(np.mean(mod.land.organic - baseline['SOM'], axis=1), lw=lw, label=name, ls=lss[m])
            axs[3].plot(mod.rangeland.R - baseline['rangeland'], lw=lw, label=name, ls=lss[m])
        else:
            axs[0].plot(np.mean(mod.agents.livestock, axis=1), lw=lw, label=name, ls=lss[m])
            axs[1].plot(np.mean(mod.agents.income, axis=1), lw=lw, label=name, ls=lss[m])
            axs[2].plot(np.mean(mod.land.organic, axis=1), lw=lw, label=name, ls=lss[m])
            axs[3].plot(mod.rangeland.R, lw=lw, label=name, ls=lss[m])

        m += 1
        plot_mods.append(name)

    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabs[a])
        # ax.legend()
        if relative:
            ax.axhline(0, color='k', lw=lw*3/4)
        ax.set_title('{}{}'.format(pre, titles[a]))
        ax.axvline(mod.lsla.tstart, lw=1, color='k', ls=':')

    lg = fig.legend(plot_mods, loc=10, bbox_to_anchor=(0.5, 0.05), ncol=len(plot_mods), frameon=False)


    if isinstance(savedir, bool):
        return
    else:
        ext = '_relative' if relative else ''
        fig.savefig('{}combined_effect{}.png'.format(savedir, ext), bbox_extra_artists=(lg,))
        plt.close('all')

