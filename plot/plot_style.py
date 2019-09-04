'''
Plotting style function for green space

README:
    In each plot function, import this module

    Have code
    ``
    import matplotlib as mpl
    with styles['context']:
        mpl.rcParams.update(styles['params'])
    ``
'''
from math import *
import matplotlib.pyplot as plt
from cycler import cycler


def create():
    columns = 1
    fig_width = 40#15.24#6.5 # if columns==1 else 6.9 # width in cm
    font_size = 14
    #
    golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_height = fig_width*golden_mean # height in cm,

    styles = {
        'context' : 'fivethirtyeight',
        ## styles for figures on paper
        'paper' : {
            'font.size': font_size, # was 10

            'figure.autolayout' : True,
            'figure.figsize': [fig_width/2.54,fig_height/2.54],
            'figure.facecolor' : 'white',
            'patch.facecolor' : 'white',
            
            'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
            'axes.titlesize' : font_size * 1.3,
            'axes.edgecolor' : 'k',
            'axes.linewidth' : 1.5,
            'axes.facecolor' : 'white',
            
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            
            'lines.linewidth' : 2,
            'grid.color' : '#CACFD2',

            # 'legend.framealpha' : None,
            'legend.fontsize': font_size * 0.8, # was 10
            'legend.facecolor' : 'white',
            'savefig.transparent' : False,
            'savefig.facecolor' : 'white',
            'savefig.dpi' : 100,
            # 'axes.spines.color' : 'k'
            'axes.prop_cycle'    : cycler('color', 'brkgcmy')

            },

        ## styles for figures in a presentation (black background)
        'presentation_black_bg' : {
            'font.size': font_size, # was 10
            
            'figure.autolayout' : True,
            'figure.figsize': [fig_width/2.54,fig_height/2.54],
            'figure.facecolor' : 'none',

            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'ytick.color' : 'w',
            'xtick.color' : 'w',

            'grid.color' : '#ababab',
            'grid.alpha' : 0.3,
            
            'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
            'axes.titlesize' : font_size * 1.3,
            'axes.linewidth' : 0.5,
            'axes.labelcolor' : 'w',
            'axes.edgecolor' : 'w',
            
            'lines.linewidth' : 2,
            'text.color' : 'w',
            'grid.color' : 'w',
            
            'legend.fontsize': font_size * 1.2, # was 10
            'legend.facecolor' : 'k',
            'savefig.transparent' : True,
            'savefig.dpi' : 100,
            
            },

        'spine_color' : 'w', # can't be changed directly with rcparams
        'fig_width' : fig_width/2.54,
        'fig_height' : fig_height/2.54,
        }

    return styles