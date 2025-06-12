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
import datetime
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyrascont
import pandas as pd
from tqdm import tqdm
from threading import Lock

# Constants
DB_PATH = Path.cwd() / 'mof_project.db'
TABLE = 'low_pressure_gcmc'
FIRST_COL = None

# Thread-safe file append
write_lock = Lock()
def append_to_file(filename, text):
    """파일에 text 한 줄을 안전하게(락 사용) 추가합니다."""
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

def load_completed_list(completed_file):
    """completed_file에서 이미 완료된 MOF 리스트를 읽어 세트로 반환합니다."""
    if not os.path.exists(completed_file):
        return set()
    with open(completed_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

# Prepare logs directory and rotate if exists
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

# Configure multiprocessing logging
log_queue = multiprocessing.Queue(-1)
queue_handler = QueueHandler(log_queue)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(queue_handler)

# Handlers for writing to files
gress_handler = logging.FileHandler(progress_log)
gress_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
gress_handler.addFilter(lambda rec: rec.name=='progress')

complete_handler = logging.FileHandler(complete_log)
complete_handler.setFormatter(logging.Formatter('%(message)s'))
complete_handler.addFilter(lambda rec: rec.name=='complete')

listener = QueueListener(log_queue, gress_handler, complete_handler)
listener.start()

# Convenience loggers
progress_logger = logging.getLogger('progress')
complete_logger = logging.getLogger('complete')


def get_connection():
    """SQLite 연결 반환"""
    if not DB_PATH.exists():
        print(f"✖ DB 파일이 없습니다: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def _make_simulation_input(mof, base_tpl, params, out_root, raspa_dir):
    cif = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(str(cif), float(params['CUTOFFVDW']), Display=False)
        uc = ' '.join(map(str, ucell))
    except Exception as e:
        print(f"⚠ CIF 변환 실패: {mof}: {e}")
        return mof
    try:
        content = base_tpl.format(
            NumberOfCycles=params['NumberOfCycles'],
            NumberOfInitializationCycles=params['NumberOfInitializationCycles'],
            PrintEvery=params['PrintEvery'],
            UseChargesFromCIFFile=params['UseChargesFromCIFFile'],
            Forcefield=params['Forcefield'],
            TEMP=params['ExternalTemperature'],
            PRESSURE=float(params['ExternalPressure'])*1e5,
            GAS=params['GAS'],
            MoleculeDefinition=params.get('MoleculeDefinition',''),
            MOF=mof,
            UNITCELL=uc
        )
        d = out_root / mof
        d.mkdir(exist_ok=True)
        with open(d/'simulation.input','w') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠ 시뮬 입력 파일 생성 실패: {mof}: {e}")
    return mof


def parse_data_file(mof_dir: Path):
    root = mof_dir/'Output'/'System_0'
    files = list(root.glob('*.data'))
    if not files:
        raise FileNotFoundError(f".data 파일 없음: {root}")
    text = files[0].read_text()
    key = 'Average loading absolute [mol/kg framework]'
    try:
        val = float(text.split(key)[1].split('+/-')[0].split()[-1])
        return val
    except Exception:
        raise ValueError(f"데이터 파싱 실패 for {mof_dir}")


def run_one(task):
    mof, idx, raspa = task
    d = Path('low_pressure_gcmc')/mof
    cmd = f"{raspa}/bin/simulate simulation.input"
    start = time.time()
    subprocess.run(cmd, shell=True, cwd=d, check=True)
    t = time.time()-start
    uptake = parse_data_file(d)
    return mof, uptake, t


def cmd_create(n):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    global FIRST_COL
    FIRST_COL = df.columns[0]
    cfg = json.load(open('gcmcconfig.json'))
    raspa = Path(cfg['RASPA_DIR'])
    base_tpl = open('base.input').read()
    out = Path('low_pressure_gcmc'); out.mkdir(exist_ok=True)
    ms = list(df[FIRST_COL].astype(str))
    print(f"▶ CREATE: {len(ms)} MOFs on {n} threads")
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend('threading', n_jobs=n):
        Parallel()(delayed(_make_simulation_input)(m, base_tpl, cfg, out, raspa) for m in ms)
    print('✅ CREATE 완료')


def cmd_run(n):
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    global FIRST_COL
    FIRST_COL = df.columns[0]

    all_mofs = list(df[FIRST_COL].astype(str))
    completed = load_completed_list(str(complete_log))
    pending = [m for m in all_mofs if m not in completed]
    total = len(all_mofs)
    to_run = len(pending)
    print(f"▶ RUN: {to_run}/{total} MOFs on {n} threads")
    if to_run == 0:
        print("✔ 처리할 MOF가 없습니다.")
        return

    start_time = time.time()
    now0 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    append_to_file(str(progress_log), f"---- START at {now0}, remaining={to_run}/{total} ----")

    cfg = json.load(open('gcmcconfig.json'))
    raspa = cfg['RASPA_DIR']
    tasks = [(m, i, raspa) for i, m in enumerate(pending,1)]

    done = len(completed)
    tot_t = 0.0

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend('threading', n_jobs=n):
        results = Parallel()(delayed(run_one)(task) for task in tasks)

    for mof, uptake, t in results:
        done += 1
        tot_t += t
        avg = tot_t/done
        rem = total - done
        eta = avg*rem
        peta = eta/n

        conn2 = sqlite3.connect(DB_PATH)
        conn2.execute(
            f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, completed=1 WHERE {FIRST_COL}=?",
            (uptake, t, mof)
        )
        conn2.commit()
        conn2.close()

        append_to_file(str(complete_log), mof)
        append_to_file(str(progress_log),
            f"[{done}/{total}] {mof} | time={t:.1f}s | avg={avg:.1f}s | rem={rem} | ETA={int(peta)}s"
        )
        print(f"✔ {mof}: uptake={uptake}, time={t:.1f}s")

    now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    append_to_file(str(progress_log), f"---- ALL DONE at {now1} ----")
    print('✅ RUN 완료')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog="low_pressure_gcmc.py")
    sub=parser.add_subparsers(dest='cmd',required=True)
    p1=sub.add_parser('create');p1.add_argument('-n','--ncpus',type=int,default=1)
    p2=sub.add_parser('run');p2.add_argument('-n','--ncpus',type=int,default=1)
    args=parser.parse_args()
    if args.cmd=='create':
        cmd_create(args.ncpus)
    else:
        cmd_run(args.ncpus)
    listener.stop()