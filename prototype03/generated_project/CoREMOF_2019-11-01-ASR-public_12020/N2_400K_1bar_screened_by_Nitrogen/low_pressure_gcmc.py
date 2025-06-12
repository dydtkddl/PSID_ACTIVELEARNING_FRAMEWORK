#!/usr/bin/env python3
# low_pressure_gcmc.py

import argparse
import json
import os
import random
import subprocess
import sys
import sqlite3
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyrascont
import pandas as pd
from tqdm import tqdm

# Constants
DB_PATH = Path.cwd() / 'mof_project.db'
TABLE = 'low_pressure_gcmc'
FIRST_COL = None

# Prepare logs directory and rotate if exists
def _prepare_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        idx = 1
        while True:
            new_path = path.with_name(f"{path.stem}.{idx}{path.suffix}")
            if not new_path.exists():
                path.rename(new_path)
                break
            idx += 1

# Logging setup
# ensure logs directory exists and rotate existing logs
logs_dir = Path('logs/low_pressure_gcmc')
logs_dir.mkdir(parents=True, exist_ok=True)

def _rotate(path: Path):
    if path.exists():
        idx = 1
        while True:
            new = path.with_name(f"{path.stem}.{idx}{path.suffix}")
            if not new.exists():
                path.rename(new)
                break
            idx += 1

progress_log = logs_dir / 'progress.log'
complete_log = logs_dir / 'completed.log'
_rotate(progress_log)
_rotate(complete_log)

progress_logger = logging.getLogger('progress')
progress_handler = logging.FileHandler(progress_log)
progress_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
progress_logger.addHandler(progress_handler)
progress_logger.setLevel(logging.INFO)

complete_logger = logging.getLogger('complete')
complete_handler = logging.FileHandler(complete_log)
complete_handler.setFormatter(logging.Formatter('%(message)s'))
complete_logger.addHandler(complete_handler)
complete_logger.setLevel(logging.INFO)

def get_connection():
    if not DB_PATH.exists():
        print(f"✖ DB 파일이 없습니다: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def _make_simulation_input(mof, base_tpl, params, out_root, raspa_dir):
    cif = raspa_dir / 'share/raspa/structures/cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(str(cif), float(params["CUTOFFVDW"]), Display=False)
        uc = ' '.join(map(str, ucell))
    except Exception as e:
        print(f"⚠ CIF 실패: {mof}: {e}")
        return mof
    try:
        content = base_tpl.format(**{
            'NumberOfCycles': params['NumberOfCycles'],
            'NumberOfInitializationCycles': params['NumberOfInitializationCycles'],
            'PrintEvery': params['PrintEvery'],
            'UseChargesFromCIFFile': params['UseChargesFromCIFFile'],
            'Forcefield': params['Forcefield'],
            'TEMP': params['ExternalTemperature'],
            'PRESSURE': float(params['ExternalPressure'])*1e5,
            'GAS': params['GAS'],
            'MoleculeDefinition': params.get('MoleculeDefinition',''),
            'MOF': mof,
            'UNITCELL': uc
        })
        d = out_root / mof
        d.mkdir(exist_ok=True)
        with open(d/'simulation.input','w') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠ 시뮬생성 실패: {mof}: {e}")
    return mof


def parse_data_file(mof_dir: Path):
    root = mof_dir/'Output'/'System_0'
    files = list(root.glob('*.data'))
    if not files: raise FileNotFoundError(files)
    text = files[0].read_text()
    keys = [
        'Average loading absolute [mol/kg framework]'
    ]
    for k in keys:
        try:
            val = float(text.split(k)[1].split('+/-')[0].split()[-1])
            return val
        except:
            pass
    raise ValueError('parse 실패')


def run_one(task):
    mof, idx, raspa = task
    d = Path('low_pressure_gcmc')/mof
    cmd = f"{raspa}/bin/simulate simulation.input"
    start=time.time(); subprocess.run(cmd, shell=True, cwd=d, check=True)
    t=time.time()-start
    uptake=parse_data_file(d)
    return mof, uptake, t


def cmd_create(n):
    conn=get_connection(); df=pd.read_sql(f'SELECT * FROM {TABLE}',conn); conn.close()
    global FIRST_COL; FIRST_COL=df.columns[0]
    cfg=json.load(open('gcmcconfig.json')); raspa=Path(cfg['RASPA_DIR'])
    base=open('base.input').read(); out=Path('low_pressure_gcmc'); out.mkdir(exist_ok=True)
    ms=list(df[FIRST_COL].astype(str)); print(f"▶ CREATE {len(ms)} MOFs on {n} CPUs")
    with ProcessPoolExecutor(n) as e:
        list(tqdm(e.map(lambda m: _make_simulation_input(m,base,cfg,out,raspa),ms),total=len(ms)))
    print('✅ CREATE 완료')


def cmd_run(n):
    conn=get_connection(); df=pd.read_sql(f'SELECT * FROM {TABLE}',conn); conn.close()
    global FIRST_COL; FIRST_COL=df.columns[0]
    pending=list(df[df['completed'].isna()][FIRST_COL].astype(str)); total=len(pending)
    print(f"▶ RUN {total} MOFs on {n} CPUs")
    cfg=json.load(open('gcmcconfig.json')); raspa=cfg['RASPA_DIR']
    tasks=[(m,i,raspa) for i,m in enumerate(pending,1)]
    done=0; tot_t=0
    with ProcessPoolExecutor(n) as e:
        for mof,upt,t in tqdm(e.map(run_one,tasks),total=total):
            done+=1; tot_t+=t; avg=tot_t/done; rem=total-done; eta=avg*rem; peta=eta/n
            conn2=sqlite3.connect(DB_PATH)
            conn2.execute(f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?,calculation_time=?,completed=1 WHERE {FIRST_COL}=?",(upt,t,mof)); conn2.commit(); conn2.close()
            progress_logger.info(f"{done}/{total} | time={t:.1f}s | avg={avg:.1f}s | rem={rem} | ETA={peta:.1f}s")
            complete_logger.info(mof)
            print(f"✔ {mof} t={t:.1f}s uptake={upt}")
    print('✅ RUN 완료')

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('create').add_argument('-n','--ncpus',type=int,default=1)
    s.add_parser('run').add_argument('-n','--ncpus',type=int,default=1)
    args=p.parse_args(); cmd_create(args.ncpus) if args.cmd=='create' else cmd_run(args.ncpus)
