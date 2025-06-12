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
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
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
logs_dir = Path('logs/low_pressure_gcmc')
logs_dir.mkdir(parents=True, exist_ok=True)

def _prepare_log(path: Path):
    if path.exists():
        idx = 1
        while True:
            new = path.with_name(f"{path.stem}.{idx}{path.suffix}")
            if not new.exists():
                path.rename(new)
                break
            idx += 1

# Rotate existing logs
progress_log = logs_dir / 'progress.log'
complete_log = logs_dir / 'completed.log'
_prepare_log(progress_log)
_prepare_log(complete_log)

# Configure multiprocessing logging
log_queue = multiprocessing.Queue(-1)
queue_handler = QueueHandler(log_queue)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(queue_handler)

# Handlers for writing to files
progress_handler = logging.FileHandler(progress_log)
progress_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
progress_handler.addFilter(lambda record: record.name == 'progress')

complete_handler = logging.FileHandler(complete_log)
complete_handler.setFormatter(logging.Formatter('%(message)s'))
complete_handler.addFilter(lambda record: record.name == 'complete')

listener = QueueListener(log_queue, progress_handler, complete_handler)
listener.start()

# Convenience loggers for modules
progress_logger = logging.getLogger('progress')
complete_logger = logging.getLogger('complete')


def get_connection():
    """
    SQLite 연결을 반환
    """
    if not DB_PATH.exists():
        print(f"✖ DB 파일이 없습니다: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def _make_simulation_input(mof, base_tpl, params, out_root, raspa_dir):
    """
    단일 MOF 폴더에 simulation.input 생성
    """
    cif = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(str(cif), float(params["CUTOFFVDW"]), Display=False)
        uc = ' '.join(map(str, ucell))
    except Exception as e:
        print(f"⚠ CIF 실패: {mof}: {e}")
        return mof
    try:
        content = base_tpl.format(
            NumberOfCycles=params['NumberOfCycles'],
            NumberOfInitializationCycles=params['NumberOfInitializationCycles'],
            PrintEvery=params['PrintEvery'],
            UseChargesFromCIFFile=params['UseChargesFromCIFFile'],
            Forcefield=params['Forcefield'],
            TEMP=params['ExternalTemperature'],
            PRESSURE=float(params['ExternalPressure']) * 1e5,
            GAS=params['GAS'],
            MoleculeDefinition=params.get('MoleculeDefinition', ''),
            MOF=mof,
            UNITCELL=uc
        )
        d = out_root / mof
        d.mkdir(exist_ok=True)
        with open(d / 'simulation.input', 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠ 시뮬입력 생성 실패: {mof}: {e}")
    return mof


def parse_data_file(mof_dir: Path):
    """
    .data 파일에서 uptake 정보 추출
    """
    root = mof_dir / 'Output' / 'System_0'
    files = list(root.glob('*.data'))
    if not files:
        raise FileNotFoundError(f".data 파일 없음 in {root}")
    text = files[0].read_text()
    key = 'Average loading absolute [mol/kg framework]'
    try:
        val = float(text.split(key)[1].split('+/-')[0].split()[-1])
        return val
    except Exception:
        raise ValueError(f"parse 실패 for {mof_dir}")


def run_one(task):
    """
    GCMC 실행 및 결과 반환
    """
    mof, idx, raspa = task
    d = Path('low_pressure_gcmc') / mof
    cmd = f"{raspa}/bin/simulate simulation.input"
    start = time.time()
    subprocess.run(cmd, shell=True, cwd=d, check=True)
    t = time.time() - start
    uptake = parse_data_file(d)
    return mof, uptake, t


def cmd_create(n):
    conn = get_connection()
    df = pd.read_sql(f'SELECT * FROM {TABLE}', conn)
    conn.close()
    global FIRST_COL
    FIRST_COL = df.columns[0]
    cfg = json.load(open('gcmcconfig.json'))
    raspa = Path(cfg['RASPA_DIR'])
    base_tpl = open('base.input').read()
    out = Path('low_pressure_gcmc')
    out.mkdir(exist_ok=True)
    ms = df[FIRST_COL].astype(str).tolist()
    print(f"▶ CREATE: {len(ms)} MOFs on {n} CPUs")
    with ProcessPoolExecutor(max_workers=n) as exe:
        list(tqdm(exe.map(lambda m: _make_simulation_input(m, base_tpl, cfg, out, raspa), ms), total=len(ms)))
    print('✅ CREATE 완료')


def cmd_run(n):
    conn = get_connection()
    df = pd.read_sql(f'SELECT * FROM {TABLE}', conn)
    conn.close()
    global FIRST_COL
    FIRST_COL = df.columns[0]
    pending = df[df['completed'].isna()][FIRST_COL].astype(str).tolist()
    total = len(pending)
    print(f"▶ RUN: {total} MOFs on {n} CPUs")
    cfg = json.load(open('gcmcconfig.json'))
    raspa = cfg['RASPA_DIR']
    tasks = [(m, i, raspa) for i, m in enumerate(pending, 1)]
    done = 0
    tot_t = 0.0
    with ProcessPoolExecutor(max_workers=n) as exe:
        for mof, upt, t in tqdm(exe.map(run_one, tasks), total=total):
            done += 1
            tot_t += t
            avg = tot_t / done
            rem = total - done
            eta = avg * rem
            peta = eta / n
            conn2 = sqlite3.connect(DB_PATH)
            conn2.execute(
                f"UPDATE {TABLE} SET `uptake[mol/kg framework]` = ?, calculation_time = ?, completed = 1 WHERE {FIRST_COL} = ?",
                (upt, t, mof)
            )
            conn2.commit()
            conn2.close()
            # log via queue
            logging.getLogger('progress').info(f"{done}/{total} | time={t:.1f}s | avg={avg:.1f}s | rem={rem} | ETA={peta:.1f}s")
            logging.getLogger('complete').info(mof)
            print(f"✔ {mof} t={t:.1f}s uptake={upt}")
    print('✅ RUN 완료')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="low_pressure_gcmc.py")
    sub = parser.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('create', help='simulation.input 생성')
    p1.add_argument('-n', '--ncpus', type=int, default=1)
    p2 = sub.add_parser('run', help='GCMC 실행')
    p2.add_argument('-n', '--ncpus', type=int, default=1)
    args = parser.parse_args()
    if args.cmd == 'create':
        cmd_create(args.ncpus)
    elif args.cmd == 'run':
        cmd_run(args.ncpus)
    listener.stop()  # Stop the log listener
