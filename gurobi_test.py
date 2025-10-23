import os.path
import pickle
import multiprocessing
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from helper import get_BG_from_GRB
import time
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def collect(ins_dir, topodir, bg_dir):
    #get bipartite graph , binary variables' indices
    print(ins_dir)
    A2,v_map2,v_nodes2,c_nodes2,b_vars2=get_BG_from_GRB(ins_dir, topodir)
    BG_data=[A2,v_map2,v_nodes2,c_nodes2,b_vars2]
    # save data
    pickle.dump(BG_data, open(os.path.join(bg_dir, filename+'.bg'), 'wb'))



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    task=f'test'  # 任务名称
    BG_DIR =f'./dataset/{task}/BG'
    INS_DIR = f'./instance/{task}'
    TOPO_DIR = f'./Topology/{task}'
    filenames = os.listdir(INS_DIR)
    toponames = sorted(os.listdir(TOPO_DIR), key=natural_sort_key)
    print(filenames)
    i = 0
    for filename in filenames:
        print(filename)
        filepath = os.path.join(INS_DIR, filename)
        begin = time.time()
        collect(filepath, toponames[0], BG_DIR)
        i += 1
    end = time.time()
    print(f'Instance {filename} processed in {end - begin:.2f} seconds.')
    # add ins
    print('done')
