import os.path
import pickle
import multiprocessing
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from helper import get_BG_from_GRB
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def solve_grb(filepath,log_dir,settings):

    gp.setParam('LogToConsole', 0)
    m = gp.read(filepath)
    m.Params.PoolSolutions = settings['maxsol']
    m.Params.PoolSearchMode = settings['mode']
    m.Params.TimeLimit = settings['maxtime']
    m.Params.Threads = settings['threads']
    log_path = os.path.join(log_dir, os.path.basename(filepath)+'.log')
    with open(log_path,'w'):
        pass

    m.Params.LogFile = log_path
    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr('SolCount') 
    mvars = m.getVars()
    #get variable name,
    oriVarNames = [var.varName for var in mvars]

    # varInds=np.arange(0, len(oriVarNames))

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        
   
        num_obj = m.NumObj if hasattr(m, 'NumObj') else 1
        if num_obj > 1:
          
            obj_vals = [m.getObjective(i).getValue() for i in range(num_obj)]
            objs.append(obj_vals)
        else:
           
            objs.append(m.PoolObjVal)

    sols = np.array(sols,dtype=np.float32)
    objs = np.array(objs,dtype=np.float32)

    sol_data = {
        'var_names': oriVarNames,
        'sols': sols,
        'objs': objs,
    }
    return sol_data

def collect(ins_dir, topodir, q, r, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        toponame = r.get()
        print(filename)
        print(toponame)
        if not filename:
            break
        topopath = os.path.join(topodir, toponame)
        filepath = os.path.join(ins_dir, filename)        
        sol_data = solve_grb(filepath, log_dir, settings)
        #get bipartite graph , binary variables' indices
        A2,v_map2,v_nodes2,c_nodes2,b_vars2=get_BG_from_GRB(filepath, topopath)
        BG_data=[A2,v_map2,v_nodes2,c_nodes2,b_vars2]
        
        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename+'.bg'), 'wb'))



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=1)
    parser.add_argument('--maxTime', type=int, default=36000)
    parser.add_argument('--maxStoredSol', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    task=f'train' 

    dataDir = args.dataDir
    INS_DIR = os.path.join(dataDir,f'instance/{task}')
    if not os.path.isdir(f'./dataset/{task}'):
        os.mkdir(f'./dataset/{task}')
    if not os.path.isdir(f'./dataset/{task}/solution'):
        os.mkdir(f'./dataset/{task}/solution')
    if not os.path.isdir(f'./dataset/{task}/logs'):
        os.mkdir(f'./dataset/{task}/logs')
    if not os.path.isdir(f'./dataset/{task}/BG'):
        os.mkdir(f'./dataset/{task}/BG')

    SOL_DIR =f'./dataset/{task}/solution'
    LOG_DIR =f'./dataset/{task}/logs'
    BG_DIR =f'./dataset/{task}/BG'
    TOPO_DIR = f'./Topology/{task}'
    os.makedirs(SOL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)
    os.makedirs(TOPO_DIR, exist_ok=True)
    N_WORKERS = args.nWorkers

    # gurobi settings
    SETTINGS = {
        'maxtime': args.maxTime,
        'mode': 2,
        'maxsol': args.maxStoredSol,
        'threads': args.threads,
    }

    filenames = os.listdir(INS_DIR)
    toponames = os.listdir(TOPO_DIR)
    toponames = sorted(toponames, key=natural_sort_key)
    # print(filenames)
    # print(toponames)
    q = Queue()
    r = Queue()
    # add ins
    for filename in filenames:
        # if not os.path.exists(os.path.join(BG_DIR,filename+'.bg')):
            # print(filename)
        q.put(filename)
    for toponame in toponames:
        r.put(toponame)
    # add stop signal
    for i in range(N_WORKERS):
        q.put(None)
    ps = []
    for i in range(N_WORKERS):
        p = Process(target=collect, args=(INS_DIR, TOPO_DIR, q, r, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    print('done')

