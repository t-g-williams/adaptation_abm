'''
Conduct a sensitivity analysis over the baseline ranges of the parameters
used in the POM
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import code
import pickle
import os
import copy
import sys
import matplotlib.transforms as transforms
from plot import plot_style
plot_type = 'paper'#'presentation_black_bg'
styles = plot_style.create() # get the plotting styles
styles['plot_type'] = plot_type
plt.style.use('fivethirtyeight')
plt.style.use(styles[plot_type])

import scipy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
import sklearn.model_selection as mod_sel
from sklearn.compose import TransformedTargetRegressor

def main(fits, calib_vars, rvs, threshs, exp_name):
    '''
    Run GBRF models to assess:
        1. parameter sensitivity (variable importance)
        2. parameter influence (PDP)
        3. meta-model accuracy (predictive accuracy)
    '''
    ## A: construct the outcomes
    # # social - P(S+)
    # Y_s = np.mean((fits[:,:,4]>=threshs[0]) * (fits[:,:,5]>=threshs[0]), axis=0)
    # # environmental - P(E+)
    # Y_e = np.mean((fits[:,:,6]>=threshs[1]) * (fits[:,:,7]>=threshs[1]), axis=0)
    # # feasibility
    # Y_f = np.mean(fits[:,:,0], axis=0)
    # social - P(S+)
    Y_s = ((fits[:,:,4]>=threshs[0]) * (fits[:,:,5]>=threshs[0])).flatten()
    # environmental - P(E+)
    Y_e = ((fits[:,:,6]>=threshs[1]) * (fits[:,:,7]>=threshs[1])).flatten()
    # social - P(S-)
    Y_s2 = ((fits[:,:,4]<threshs[0]) | (fits[:,:,5]<threshs[0])).flatten()
    # environmental - P(E-)
    Y_e2 = ((fits[:,:,6]<threshs[1]) | (fits[:,:,7]<threshs[1])).flatten()
    # feasibility
    Y_f = fits[:,:,0].flatten().astype(bool)
    # tile the RVs (i.e. there are multiple reps of each outcome)
    rvs = np.tile(rvs, (fits.shape[0],1))
    # combine
    Ys = [Y_s, Y_e, Y_f, Y_s2, Y_e2]
    Y_names = ['P(S+)', 'P(E+)', 'P(feas)', 'P(S-)', 'P(E-)']

    ## B: run the regression models
    rf_data = []
    for yi, Y in enumerate(Ys):
        rf_data.append(cross_val_random_forest(Y, Y_names[yi], rvs, calib_vars, exp_name))
    

    ## C: plot it
    # (develop function in jupyter)
    code.interact(local=dict(globals(), **locals()))

def cross_val_random_forest(y, y_name, X, calib_vars, exp_name):
    '''
    run the random forest and PDP calculations n_holdout times
    '''
    n_holdout = 10
    outdir = '../outputs/{}/pom_sensitivity/'.format(exp_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outname = outdir + 'baseline_pom_sensitivity_{}.pkl'.format(y_name)
    if os.path.isfile(outname):
        return pickle.load(open(outname, 'rb'))

    # init outputs
    var_imps = {}
    pdp_datas = {}
    fits = []
    for v in calib_vars.key2:
        pdp_datas[v] = {'x' : [], 'y' : []}
        var_imps[v] = []

    print(y_name)
    for h in range(n_holdout):
        print('   {}...'.format(h))
        # create random sample
        X_tr, X_te, y_tr, y_te = mod_sel.train_test_split(X, y, test_size=0.2, random_state=h)
        # run the random forest
        var_imp, pdp_data, fit = random_forest(y_tr, y_te, X_tr, X_te, calib_vars.key2, calib_vars.key1)

        fits.append(fit)
        if h == 0:
            var_imps_df = var_imp
        else:
            # take mean importance
            var_imps_df['importance'] = pd.merge(var_imps_df, var_imp['importance'], left_index=True, right_index=True, how='outer').sum(axis=1)
        for v in calib_vars.key2:
            pdp_datas[v]['x'].append(pdp_data[v][1])
            pdp_datas[v]['y'].append(pdp_data[v][0])
            var_imps[v].append(var_imp['importance'][var_imp['variable']==v].values[0])

    # combine results
    var_imps_df['importance'] /= n_holdout
    # write
    combined = {'var_imps_df' : var_imps_df, 'var_imps' : var_imps, 'pdp_datas' : pdp_datas, 'fit' : fits}
    with open(outname, 'wb') as f:
        pickle.dump(combined, f, pickle.HIGHEST_PROTOCOL)

    return combined

def random_forest(y_tr, y_te, X_tr, X_te, varz, keys):
    # fit gradient boosted forest
    # tt = TransformedTargetRegressor(regressor=GradientBoostingRegressor(random_state=0), 
    #                     func=scipy.special.logit, inverse_func=scipy.special.expit)    
    # gb = tt.fit(X_tr, y_tr)

    gb = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)

    # extract importances
    var_imp = pd.DataFrame({'importance' : gb.feature_importances_, 'key' : keys, 
        'variable' : varz}).sort_values(['key','importance'], ascending=False)

    # var_imp = pd.DataFrame(gb.feature_importances_, index = varz, 
    #         columns=['importance']).sort_values('importance', ascending=False)

    # partial dependence
    pdp_data = {}
    for xi, var in enumerate(varz):
        tmp = partial_dependence(gb, features=[xi], X=X_tr)
        # transform to probability
        pdp_data[varz[xi]] = [scipy.special.expit(tmp[0][0]), tmp[1][0]]

    # predictive accuracy (in-sample)
    fit = gb.score(X_te,y_te)

    return var_imp, pdp_data, fit