'''
processing of LSLA survey data for ABM input
'''
import numpy as np
import pandas as pd
import code

def main():
    d = pd.read_csv('../inputs/LSLT Ethiopia Survey.csv')
    d = d.loc[8:]
    create_hh_characteristics(d)

def create_hh_characteristics(d):
    hh_size = d['MHTOT']
    land_subs = d['ALANDOC_x1_1'].astype(float)
    land_comm = d['ALANDOC_x2_1'].astype(float)
    land_subs[np.isnan(land_subs)] = 0
    land_comm[np.isnan(land_comm)] = 0
    land_tot = land_subs + land_comm

    df = pd.DataFrame({'hh_size' : hh_size, 'land_ha' : land_tot})
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()