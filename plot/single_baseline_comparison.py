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

def main(mods, exp_name, ext, relative=False, save=True):
    if save == True:
        savedir = '../outputs/{}/'.format(exp_name)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        savedir = False

    timeline(mods, exp_name, savedir, relative, ext)
    area_comparison(mods, exp_name, savedir, ext)

def area_comparison(mods, exp_name, savedir, ext):
    '''
    compare the land area under each scenario
    '''
    ## land composition before and after
    fig, ax = plt.subplots(1,1,figsize=(30,8))
    farm_pre = mods['baseline'][0].land.tot_area
    common_pre = mods['baseline'][0].rangeland.size_ha

    dt = {'Commons' : [common_pre], 'Farm (unaffected)' : [farm_pre], 'Farm (affected)' : [0], 
        'LSLA' : [0]}
    names = ['Baseline']
    for name, mods_i in mods.items():
        mod = mods_i[0]
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

    fig.savefig(savedir+ext+'land_changes.png', dpi=200)
    plt.close('all')

    # code.interact(local=dict(globals(), **locals()))

def timeline(mods, exp_name, savedir, relative, ext):
    fig, ax_all = plt.subplots(2,4,figsize=(6*4,7), gridspec_kw={'height_ratios':[1,0.05]})
    axs = ax_all[0]
    [axi.remove() for axi in ax_all[1,:]]
    n = len(mods)-1 if relative else len(mods)
    if n < 5:
        lss = [':','-.','--','-']
    else:
        lss = ['-']*n
    lw = 3
    lss = ['-']*n
    pre = 'Change in ' if relative else ''
    titles = ['E[livestock]','E[income]','E[SOM]','rangeland biomass']
    ylabs = ['head','birr','kg N/ha','kg/ha']
    colors = brewer2mpl.get_map('Paired', 'Qualitative', n).mpl_colors

    ## format the data for the plot
    # datas = {'livestock' : ,'income','SOM','rangeland'}
    m=0
    safeguard = False
    plot_mods = []
    for name, mods_i in mods.items():
        ls = []
        inc = []
        som = []
        rng = []
        if relative:
            if name == 'baseline':
                continue
            for r in range(len(mods['baseline'])):
                ls.append(mods_i[r].agents.livestock - mods['baseline'][r].agents.livestock)
                inc.append(mods_i[r].agents.income - mods['baseline'][r].agents.income)
                som.append(mods_i[r].land.organic - mods['baseline'][r].land.organic)
                rng.append(mods_i[r].rangeland.R - mods['baseline'][r].rangeland.R)
        else:
            for r in range(len(mods['baseline'])):
                ls.append(mods_i[r].agents.livestock)
                inc.append(mods_i[r].agents.income)
                som.append(mods_i[r].land.organic)
                rng.append(mods_i[r].rangeland.R)
        
        ls_mean = np.mean(np.array(ls), axis=(0,2))
        inc_mean = np.mean(np.array(inc), axis=(0,2))
        som_mean = np.mean(np.array(som), axis=(0,2))
        rng_mean = np.mean(np.array(rng), axis=(0))
        objs = [ls_mean,inc_mean,som_mean,rng_mean]

        # plot
        for oi, obj in enumerate(objs):
            axs[oi].plot(obj, lw=lw, label=name, ls=lss[m], color=colors[m])
        m += 1
        plot_mods.append(name)
        
        if mods_i[r].adaptation_option != 'none':
            safeguard = True
            yr_safeguard = mods_i[r].all_inputs['adaptation']['burnin_period']

    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabs[a])
        # ax.legend()
        if relative:
            ax.axhline(0, color='k', lw=lw*3/4)
        ax.set_title('{}{}'.format(pre, titles[a]))
        ax.axvline(mods_i[0].lsla.tstart, lw=1, color='k', ls=':')
        if safeguard:
            ax.axvline(yr_safeguard, lw=1, color='k', ls=':')


    lg = fig.legend(plot_mods, loc=10, bbox_to_anchor=(0.5, 0.05), ncol=len(plot_mods), frameon=False)

    if isinstance(savedir, bool):
        return
    else:
        ext2 = '_relative' if relative else ''
        fig.savefig('{}{}combined_effect{}.png'.format(savedir, ext, ext2), bbox_extra_artists=(lg,), dpi=200)
        plt.close('all')

