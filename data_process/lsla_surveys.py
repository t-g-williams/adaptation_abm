'''
processing of LSLA survey data for ABM input
'''
import numpy as np
import pandas as pd
import code

def main():
    ## FROM PROCESSED DATA
    d = pd.read_csv('../inputs/lsla_combined.csv', index_col=0)
    d['land_area_init'] = np.maximum(d['total_ha_owned'],d['own_ha_cult'])
    d_abm = d[['site','treat','hh_size','land_area_init']]
    d_abm.to_csv('../inputs/lsla_for_abm.csv')
    code.interact(local=dict(globals(), **locals()))


    ## FROM RAW DATA
    d = pd.read_csv('../inputs/LSLT Ethiopia Survey.csv')
    d = d.loc[8:]
    create_hh_characteristics(d)

def create_raw_characteristics(d):
    ## household size
    hh_size = d['MHTOT']

    ## LAND ##
    # land owned
    land_subs = d['ALANDOC_x1_1'].astype(float)
    land_comm = d['ALANDOC_x2_1'].astype(float)
    land_rent_out = d['ALANDOC_x7_1'].astype(float)
    land_subs[np.isnan(land_subs)] = 0
    land_comm[np.isnan(land_comm)] = 0
    land_rent_out[np.isnan(land_rent_out)] = 0
    land_tot = land_subs + land_comm + land_rent_out
    # land cultivated
    cult_family = d['ACULTINFO_x5_AEA_1'].astype(float)
    cult_own = d['ACULTINFO_x7_AEA_1'].astype(float)
    cult_family[np.isnan(cult_family)] = 0
    cult_own[np.isnan(cult_own)] = 0
    # combine -- there are inconsistencies in the survey
    land_ha_comb = np.maximum(land_tot, cult_own)
    # fix bad values
    xx

    df = pd.DataFrame({'hh_size' : hh_size, 'land_ha' : land_ha_comb})
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()