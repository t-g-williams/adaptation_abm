'''
convergence analysis
'''
import os
import model.model as mod
import imp
import copy
import code
import tqdm
import numpy as np
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main(params):
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()