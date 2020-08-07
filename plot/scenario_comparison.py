import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import code
import brewer2mpl
import os
import matplotlib.transforms as transforms
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

    combined_plots(mods, savedir)
    agent_type_plots(mods, savedir)

    # soil_wealth(mods, savedir)
    # coping(mods, savedir)

def combined_plots(mods, savedir):
    has_shock = True if 'shock' in mods.keys() else False
    # plot wealth for an agent of type 2 only
    fig, axs = plt.subplots(3,1,figsize=(7,8), gridspec_kw={'height_ratios':[1,1,0.3]})
    ax = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    for m, mod in mods.items():
        lands = mod.agents.land_area
        uniq_land = np.unique(lands)
        burnin = mod.adap_properties['burnin_period']
        if has_shock:
            T = 8
            yr = mods['shock'].climate.shock_years[0]
            xs = np.arange(yr-2,yr+T)
            ls = ':' if m == 'shock' else '-'
            labl = 'no shock' if m == 'baseline' else m
            ax.plot(xs, mod.agents.wealth[:,lands==uniq_land[1]][xs].mean(1), label=labl, color='k', lw=2, ls=ls)
            ax2.plot(xs, mod.land.organic[:,lands==uniq_land[1]][xs+1].mean(1), label=labl, color='k', lw=2, ls=ls)
            ax3.plot(xs, mod.climate.rain[xs], label=labl, color='k', lw=2, ls=ls)
        else:
            T = 21
            xs = np.arange(burnin, burnin+T)
            col = 'k' if m == 'baseline' else 'r' if m == 'cover_crop' else 'b'
            ls = '-' if m=='baseline' else '--' if m=='cover_crop' else '-.'
            ax.plot(xs, mod.agents.income[:,lands==uniq_land[1]][xs].mean(1), label=m, lw=1.5, color=col, ls=ls)
            ax2.plot(xs, mod.land.organic[:,lands==uniq_land[1]][xs+1].mean(1), label=m, lw=1.5, color=col, ls=ls)
            ax3.plot(xs, mod.climate.rain[xs], label=m, color=col, lw=1.5, ls=ls)
        if mod.shock:
            for yr in mod.climate.shock_years:
                for a, axi in enumerate(axs):
                    axi.axvline(x=yr, color='k', lw=1, ls=':')
                    if a==1:
                        trans = transforms.blended_transform_factory(
                                axi.transData, axi.transAxes)
                        axi.text(yr, 0.05, 'SHOCK', ha='right', va='bottom', rotation=90, transform=trans)
                
        if m == 'insurance':
            # add the shock years
            shock_yrs = np.where(mod.climate.rain <= mod.adap_properties['magnitude'])[0]
            for yr in shock_yrs:
                if yr <= max(xs):
                    for a, axi in enumerate(axs):
                        axi.axvline(x=yr, color='k', lw=1, ls=':')
                        if a==1:
                            trans = transforms.blended_transform_factory(
                                    axi.transData, axi.transAxes)
                            axi.text(yr, 0.05, 'PAYOUT', ha='right', va='bottom', rotation=90, transform=trans)

            # add the climate index
            idx = mod.adap_properties['magnitude']
            ax3.axhline(y=idx, color='k', lw=1, ls=':')
            trans = transforms.blended_transform_factory(
                                    ax3.transAxes, axi.transData)
            ax3.text(0.02, idx, 'THRESHOLD', ha='left', va='bottom', transform=trans)

            
    if has_shock:
        T = 8
        # for axi in axs:
        #     axi.set_xlim([yr-3, yr+T])
        #     axi.set_xticks(np.arange(yr-2,yr+T+1,2))
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        ax.set_ylim([0,ax.get_ylim()[1]])
        # ax2.set_xticklabels(np.arange(yr-2,yr+T+1,2))
    else:
        T = 21
        for axi in axs:
            axi.set_xlim([burnin, burnin+T])
            axi.set_xticks(np.arange(burnin+1, burnin+T+1,2))
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels(np.arange(0,T,2))
        ax.axhline(y=0, color='k', lw=1)
    
    for axi in axs:
        axi.grid(False)
            
    ax2.set_ylim([0,6000])
    ax.legend()
    ax.set_ylabel('Wealth (birr)' if has_shock else 'Income (birr)')
    ax2.set_ylabel('SOM (kg N/ha)')
    ax3.set_ylabel('Climate\ncondition')
    ax3.set_xlabel('Year')
    ax3.set_ylim([0,1])
    text = 'B: Effect of shock' if has_shock else 'C: Resilience strategies'
    # ax.text(-0.2,1.1, text, transform=ax.transAxes, fontsize=22)
    ax.set_title(text, fontsize=22)

    if isinstance(savedir, bool):
        return fig
    else:
        ext = 'shock' if has_shock else 'resilience'
        fig.savefig(savedir + 'combined_{}.png'.format(ext))

def agent_type_plots(mods, savedir):
    '''
    plot each agent type (land area) separately
    this assumes there's 3 agent types
    '''
    colors = ['b','r','k','g','y']
    fig = plt.figure(figsize=(18,6))
    axs = [fig.add_subplot(131),fig.add_subplot(132),fig.add_subplot(133)]
    fig9 = plt.figure(figsize=(18,5))
    ax9s = [fig9.add_subplot(131),fig9.add_subplot(132),fig9.add_subplot(133)]
    fig6 = plt.figure(figsize=(12,20))
    ax6s = [fig6.add_subplot(311),fig6.add_subplot(312),fig6.add_subplot(313)]
    fig7 = plt.figure(figsize=(12,20))
    ax7s = [fig7.add_subplot(311),fig7.add_subplot(312),fig7.add_subplot(313)]
    fig2 = plt.figure(figsize=(12,4))
    ax2s = [fig2.add_subplot(131),fig2.add_subplot(132),fig2.add_subplot(133)]
    fig3 = plt.figure(figsize=(18,5))
    ax3s = [fig3.add_subplot(131),fig3.add_subplot(132),fig3.add_subplot(133)]
    fig4 = plt.figure(figsize=(12,20))
    ax4s = [fig4.add_subplot(311),fig4.add_subplot(312),fig4.add_subplot(313)]
    fig5 = plt.figure(figsize=(12,20))
    ax5s = [fig5.add_subplot(311),fig5.add_subplot(312),fig5.add_subplot(313)]
    fig8 = plt.figure(figsize=(18,6))
    ax8s = [fig8.add_subplot(131),fig8.add_subplot(132),fig8.add_subplot(133)]
    fig10 = plt.figure(figsize=(18,6))
    ax10s = [fig10.add_subplot(131),fig10.add_subplot(132),fig10.add_subplot(133)]
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
            ax10s[a].plot(mod.agents.fert_choice[:,ag].mean(1), color=colors[ii], label=m)#, lw=0.5)
            ax6s[a].plot(np.median(mod.agents.income[:,ag], axis=1), label=m, color=colors[ii])
            ax9s[a].plot(np.median(mod.agents.income[:,ag], axis=1), label=m, color=colors[ii])
            ax7s[a].plot(np.median(mod.land.yields[:,ag], axis=1), label=m, color=colors[ii])
            ax8s[a].plot(mod.climate.rain, np.median(mod.land.yields[:,ag], axis=1), 'o', label=m, color=colors[ii])

        ii += 1

    # some formatting
    for a in range(3):
        axs[a].set_title('Wealth : agent type {}'.format(a+1))
        ax4s[a].set_title('Wealth : agent type {}'.format(a+1))
        ax2s[a].set_title('Coping : agent type {}'.format(a+1))
        ax3s[a].set_title('SOM : agent type {}'.format(a+1))
        ax5s[a].set_title('SOM : agent type {}'.format(a+1))
        ax6s[a].set_title('Income : agent type {}'.format(a+1))
        ax9s[a].set_title('Income : agent type {}'.format(a+1))
        ax7s[a].set_title('Yield : agent type {}'.format(a+1))
        ax8s[a].set_title('Yield : agent type {}'.format(a+1))
        ax10s[a].set_title('P(choose fert) : agent type {}'.format(a+1))
        axs[a].set_ylabel('Birr')
        ax4s[a].set_ylabel('Birr')
        ax2s[a].set_ylabel('P(coping rqd)')
        ax3s[a].set_ylabel('kg/ha')
        ax5s[a].set_ylabel('kg/ha')
        ax6s[a].set_ylabel('Birr')
        ax9s[a].set_ylabel('Income')
        ax7s[a].set_ylabel('kg/ha')
        ax8s[a].set_ylabel('kg/ha')
        ax10s[a].set_ylabel('Probability')
        ax8s[a].set_xlabel('Rainfall')
        axs[a].legend()
        ax2s[a].legend()
        ax3s[a].legend()
        ax6s[a].legend()
        ax9s[a].legend()
        ax10s[a].legend()
        axs[a].axhline(y=0, color='k', ls=':')
        ax4s[a].axhline(y=0, color='k', ls=':')
        ax3s[a].axhline(y=0, color='k', ls=':')
        ax5s[a].axhline(y=0, color='k', ls=':')
        ax6s[a].axhline(y=0, color='k', ls=':')
        ax9s[a].axhline(y=0, color='k', ls=':')
        for axx in [axs[a], ax2s[a], ax3s[a], ax4s[a], ax5s[a], ax6s[a], ax7s[a], ax9s[a]]:
            axx.grid(False)
            axx.set_xlabel('Time (yrs)')
            # show the shock on the plot, if necessary
            if shock:
                for yr in shock_years:
                    axx.axvline(x=yr, color='k', ls=':')
                    axx.text(yr, axx.get_ylim()[0]+(axx.get_ylim()[1]-axx.get_ylim()[0])*0.1, 'SHOCK', ha='center', va='bottom', rotation=90)
            # show the burnin period
            burnin = mod.adap_properties['burnin_period']
            axx.fill_between([0, burnin], [axx.get_ylim()[0],axx.get_ylim()[0]],[axx.get_ylim()[1],axx.get_ylim()[1]], color='0.5', alpha=0.3)
            # axx.text(burnin/2, axx.get_ylim()[1], '\nBURN-IN', ha='center', va='top')

    if isinstance(savedir, bool):
        # return fig, fig2, fig3, fig4, fig5
        return fig4, fig5
    else:
        fig.savefig(savedir + 'type_wealth.png')
        fig4.savefig(savedir + 'type_wealth_all.png')
        fig2.savefig(savedir + 'type_coping.png')
        fig3.savefig(savedir + 'type_SOM.png')
        fig5.savefig(savedir + 'type_SOM_all.png')
        fig6.savefig(savedir + 'type_income.png')
        fig9.savefig(savedir + 'type_income_horiz.png')
        fig7.savefig(savedir + 'type_yields.png')
        fig10.savefig(savedir + 'type_fertilizer.png')

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