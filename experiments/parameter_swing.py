'''
explore the effect of a systematic change in LSLA parameters
on a QoI
'''
from model.model import Model
import model.base_inputs as base_inputs
import plot.single as plot_single
import plot.single_baseline_comparison as plot_compare
import code
import experiments.POM as POM
import time
import pickle
import numpy as np
import sys
import copy
import os
import pickle
import logging
import logging.config
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
logging.config.fileConfig('logger.conf', defaults={'logfilename' : 'logs/{}.log'.format(os.path.basename(__file__)[:-3])})
logger = logging.getLogger('sLogger')


import matplotlib as mpl
# mpl.use('Agg')
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import plot.plot_style as plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

def main():
    #### define LSLA parameters ####
    baseline_params = {'model' : {'lsla_simulation' : False}}
    lsla_base_params = {
        'model' : {'lsla_simulation' : True},
        'LSLA' : {
        'tstart' : 5, # e.g. 5 means start of 6th year of simulation
        'employment' : 2, # jobs/ha taken
        'LUC' : 'farm', # 'farm' or 'commons'' or ?'none'?
        'encroachment' : 'farm', # where do displaced HHs encroach on? 'farm' or 'commons'
        'frac_retain' : 0.5, # fraction of land that was originally taken that HHs retain (on average)
        'land_distribution_type' : 'amt_lost', # current_holdings: proportional to current holdings, 'equal_hh' : equal per hh, "equal_pp" : equal per person
        'land_taking_type' : 'random', # random or equalizing
        }
    }

    exp_name = '2020_2_12_11'
    pom_nreps = '100000_10reps'
    # types of lslas
    lsla_types = {
        'farm-LUC_farm-displacement' : {'LSLA' : {'LUC' : 'farm', 'encroachment' : 'farm'}},
        'farm-LUC_common-displacement' : {'LSLA' : {'LUC' : 'farm', 'encroachment' : 'commons'}},
        'common-LUC' : {'LSLA' : {'LUC' : 'commons', 'encroachment' : 'farm'}},
    }

    lsla_type = 'farm-LUC_farm-displacement'
    # parameter sweeps -- these should be of length two
    swing_names = ['employment','frac_retain']
    swing_names_pretty = ['Employment offered (jobs/ha)', 'Fraction of land retained']
    # swing_values = [np.arange(0, 50, 2), np.arange(0.05,1,0.05)]
    swing_values = [np.arange(0, 120, 20), np.round(np.arange(0.05,1,0.2),2)]
    qoi_type = 'exp_income'
    # qoi_name = r'$E[\Delta income]$'
    qoi_name = 'E[decrease in income]'
    T_eval = 10 # how many years from the end of sim to evaluate over

    # load inputs
    f = '../outputs/{}/POM/{}/input_params_0.pkl'.format(exp_name, pom_nreps)
    inputs_pom = pickle.load(open(f, 'rb'))
    # params not in POM
    baseline_inputs = base_inputs.compile()
    for k, v in inputs_pom.items():
        for k2, v2 in v.items():
            baseline_inputs[k][k2] = v2

    nreps = 40
    ncores = 40
    baseline_inputs['model']['T'] = 30
    baseline_inputs['model']['n_agents'] = 400
    
    rep_chunks = POM.chunkIt(np.arange(nreps), ncores)

    # loop over the different types of lslas
    qoi_all = {}
    for lsla_type, lsla_params2 in lsla_types.items():
        logger.info('TYPE OF LSLA : {}.......'.format(lsla_type))
        outdir = '../outputs/{}/QOI_data/{}/'.format(exp_name, lsla_type)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        # set the params for this lsla type
        lsla_type_params = copy.deepcopy(lsla_base_params)
        for k, v in lsla_params2.items():
            for k2, v2 in v.items():
                lsla_type_params[k][k2] = v2

        # loop over the values
        QOI_type = np.full([len(swing_values[0]), len(swing_values[1])], np.nan)
        for i, pi in enumerate(swing_values[0]):
            logger.info('  i = {}/{}...'.format(i+1, len(swing_values[0])))
            for j, pj in enumerate(swing_values[1]):
                logger.info('    j = {}/{}...'.format(j+1, len(swing_values[1])))
                
                # search for the data
                outname = '{}{}_{}{}_{}{}.npz'.format(outdir, qoi_type, pi, swing_names[0], pj, swing_names[1])
                if os.path.isfile(outname):
                    QOIs = np.load(outname)['data']
                else:
                    # run if it doesn't exist
                    QOIs = []
                    tmp = Parallel(n_jobs=ncores)(delayed(qoi_chunk)(rep_chunks[i], 
                        swing_names, pi, pj, baseline_inputs, baseline_params, lsla_type_params, qoi_type, T_eval, exp_name) \
                        for i in range(len(rep_chunks)))
                    QOIs = np.array([li for subl in tmp for li in subl])

                    np.savez_compressed(outname, data=QOIs)

                QOI_type[i,j] = np.mean(QOIs)
        
        # append to the dictionary with all QOIs
        qoi_all[lsla_type] = QOI_type
        plot_grid(QOI_type, swing_names_pretty, swing_values, exp_name, qoi_type, qoi_name, lsla_type)
    
    plot_grid_combined(qoi_all, swing_names_pretty, swing_values, exp_name, qoi_type, qoi_name)

def qoi_chunk(reps, swing_names, pi, pj, baseline_inputs, baseline_params, lsla_params, qoi_type, T_eval, exp_name):
    '''
    calculate QoI for a chunk of reps
    '''
    QOIs_chunk = []
    for ri in reps:
        # baseline
        exp_base = copy.copy(baseline_params)
        exp_base['model']['seed'] = ri
        m_base = run_model(baseline_inputs, exp_base, exp_name)

        # scenario
        exp_lsla = copy.copy(lsla_params)
        exp_lsla['model']['seed'] = ri
        exp_lsla['LSLA'][swing_names[0]] = pi
        exp_lsla['LSLA'][swing_names[1]] = pj
        m_exp = run_model(baseline_inputs, exp_lsla, exp_name)    

        # calculate QoI
        QOIs_chunk.append(calculate_qoi(m_base, m_exp, qoi_type, T_eval))

    return QOIs_chunk

def run_model(baseline_inputs, change_inputs, exp_name):
    # change the baseline inputs
    params = copy.copy(baseline_inputs)
    for k, v in change_inputs.items():
        for k2, v2 in v.items():
            params[k][k2] = v2
    params['model']['exp_name'] = exp_name
    # initialize and run model
    m = Model(params)
    for t in range(m.T):
        m.step()
    return m

def calculate_qoi(m_base, m_exp, qoi_type, T_eval):
    if qoi_type == 'exp_income':
        return np.mean(m_base.agents.income[-T_eval:] - m_exp.agents.income[-T_eval:])
    else:
        print('ERROR: unknown QoI type')
        sys.exit()

def plot_grid(qois, names, values, exp_name, qoi_type, qoi_name, lsla_type):
    '''
    plot a grid of the qoi
    '''
    # fig = plt.figure(figsize=(6, 5))
    # axs = ImageGrid(fig, 111, nrows_ncols=(1,1) axes_pad=0.5, add_all=True, label_mode='L',
    #     cbar_mode='single',cbar_location='right', aspect=False)
    fig, axs = plt.subplots(1,1,figsize=(12,10))
    ax = axs
    mx_qoi = np.max(np.abs(qois))
    # qois = np.arange()
    # code.interact(local=dict(globals(), **locals()))
    hm = ax.imshow(qois, cmap='bwr',origin='lower',extent=[min(values[1]), max(values[1]), min(values[0]), max(values[0])],
                aspect='auto', vmin=-mx_qoi, vmax=mx_qoi)
    ax.set_ylabel(names[0])
    ax.set_xlabel(names[1])
    ax.grid(False)
    cb_ax = fig.add_axes([1, 0.22, 0.02, 0.5])
    cbar = fig.colorbar(hm, orientation='vertical', cax=cb_ax)
    cbar.set_label(qoi_name)

    fig.savefig('../outputs/{}/parameter_swings_{}_{}.png'.format(exp_name, qoi_type, lsla_type), bbox_inches='tight')


def plot_grid_combined(qois_all, names, values, exp_name, qoi_type, qoi_name):
    '''
    plot a grid of the qoi
    '''
    # fig = plt.figure(figsize=(6, 5))
    # axs = ImageGrid(fig, 111, nrows_ncols=(1,1) axes_pad=0.5, add_all=True, label_mode='L',
    #     cbar_mode='single',cbar_location='right', aspect=False)
    # code.interact(local=dict(globals(), **locals()))
    fig, axs = plt.subplots(1,len(qois_all),figsize=(6*len(qois_all),5))
    
    # get the maximum value so they have consistent colors
    mx_qoi = -999
    for qoi_type, qois in qois_all.items():
        mx_qoi = max(mx_qoi, np.max(np.abs(qois)))

    axi=0
    for qoi_type, qois in qois_all.items():
        # qois = np.arange()
        hm = axs[axi].imshow(qois, cmap='bwr',origin='lower',extent=[min(values[1]), max(values[1]), min(values[0]), max(values[0])],
                    aspect='auto', vmin=-mx_qoi, vmax=mx_qoi)
        axs[axi].set_title(qoi_type)
        axi += 1
    
    for a, ax in enumerate(axs):
        ax.set_ylabel(names[0])
        ax.set_xlabel(names[1])
        ax.grid(False)

    cb_ax = fig.add_axes([1, 0.22, 0.02, 0.5])
    cbar = fig.colorbar(hm, orientation='vertical', cax=cb_ax)
    cbar.set_label(qoi_name)

    fig.savefig('../outputs/{}/parameter_swings_{}_all.png'.format(exp_name, qoi_type), bbox_inches='tight')


if __name__ == '__main__':
    main()