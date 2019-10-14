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

def main(mods, shock_data, save):
    if save == True:
        savedir = '../outputs/{}/'.format(mod.exp_name)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
    else:
        savedir = False

    coping_comparison(shock_data, savedir)

def coping_comparison(d, savedir):
    '''
    plot the number of agents coping over time in the shock simulations
    for different model scenarios/conditions
    '''
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    for k, v in d.items():
        xs = np.arange(v['coping'].shape[0])
        ys = np.mean(v['coping'], axis=1)
        ax.plot(xs, ys, label=k)

    ax.legend()
    ax.grid(False)
    ax.set_xlabel('Time of shock (years)')
    ax.set_ylabel('Fraction of population')
    ax.set_title('Fraction of population coping under shock conditions')

    if isinstance(savedir, bool):
        return fig
    else:
        fig.savefig(savedir + 'coping_rqd.png')
