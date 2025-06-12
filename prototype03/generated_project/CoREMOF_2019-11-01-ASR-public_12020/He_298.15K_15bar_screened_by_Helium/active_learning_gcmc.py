#!/usr/bin/env python3
"""
Active Learning GCMC Simulation Workflow
- Supports phases: initial_gcmc (create/run), init_model, active_gcmc (future)
- Benchmarked from low_pressure_gcmc.py
"""
import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from collections import deque
import logging
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import pyrascont
import random

# Constants
TABLE = 'active_learning_gcmc'
LOG_DIR = Path.cwd() / 'logs'
LOG_DIR.mkdir(exist_ok=True)
WINDOW_SIZE = 10 
# Logging setup
run_logger = logging.getLogger('active_run')
uptake_logger = logging.getLogger('active_uptake')
error_logger = logging.getLogger('active_error')

for logger, name in [(run_logger, 'run.log'), (uptake_logger, 'uptake.log'), (error_logger, 'error.log')]:
    handler = logging.FileHandler(LOG_DIR / name)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if logger != error_logger else logging.ERROR)

# DB connection
def get_db_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn

# Simulation input generator

def make_simulation_input(mof: str, base_template: str, params: dict, out_root: Path, raspa_dir: Path, gcfg : dict) -> None:
    cif_path = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(str(cif_path), float(params.get('CUTOFFVDW', 12.8)), Display=False)
        unitcell_str = ' '.join(map(str, ucell))
    except Exception as e:
        error_logger.error(f"Input gen failed for {mof}: {e}", exc_info=True)
        return

    content = base_template.format(
        NumberOfCycles=gcfg['NumberOfCycles'],
        NumberOfInitializationCycles=gcfg['NumberOfInitializationCycles'],
        PrintEvery=gcfg['PrintEvery'],
        UseChargesFromCIFFile=gcfg['UseChargesFromCIFFile'],
        Forcefield=gcfg['Forcefield'],
        TEMP=gcfg['ExternalTemperature'],
        PRESSURE=float(gcfg['ExternalPressure']) * 1e5,
        GAS=gcfg['GAS'],
        MoleculeDefinition=gcfg.get('MoleculeDefinition', ''),
        MOF=mof,
        UNITCELL=unitcell_str
    )
    dest = out_root / mof
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'simulation.input').write_text(content, encoding='utf-8')

# Simulation runner
def parse_output(mof_dir: Path) -> dict:
    """
    Parse .data file for uptake values.
    """
    data_dir = mof_dir / 'Output' / 'System_0'
    files = list(data_dir.glob('*.data'))
    if not files:
        raise FileNotFoundError(f"No .data file in {data_dir}")
    text = files[0].read_text(encoding='utf-8')
    keys = [
        "Average loading absolute [mol/kg framework]",
        "Average loading excess [mol/kg framework]",
        "Average loading absolute [molecules/unit cell]",
        "Average loading excess [molecules/unit cell]"
    ]
    res = {}
    for k in keys:
        try:
            part = text.split(k)[1]
            val = float(part.split('+/-')[0].split()[-1])
            res[k] = val
        except Exception:
            continue
    if "Average loading absolute [mol/kg framework]" not in res:
        raise ValueError(f"Parse failed for {mof_dir.name}")
    return res

def run_simulation(mof: str, raspa_dir: Path):
    base_dir = Path('initial_gcmc')
    d = base_dir / mof
    start = time.time()
    subprocess.run(f"{raspa_dir}/bin/simulate simulation.input", shell=True, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - start
    uptake = parse_output(d)["Average loading absolute [mol/kg framework]"]
    return mof, uptake, elapsed

# Initial GCMC phase

def initial_create(db_path: Path, config_path: Path, base_input: Path, mode: str, gcfg : Path):
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    gcfg = json.loads(gcfg.read_text(encoding='utf-8'))
    raspa = Path(gcfg['RASPA_DIR'])
    tpl = base_input.read_text(encoding='utf-8')
    out = Path('initial_gcmc')
    frac = cfg['initial_fraction']

    completed_initial = df[df['iteration'] == 0]
    total_needed = int(len(df) * frac)
    remaining = total_needed - len(completed_initial)

    if remaining <= 0 or len(df[df["iteration"] > 0 ]) > 0:
        print("✅ Enough initial samples already exist. Skipping create.")
        return

    eligible = df[df['iteration'].isna()]
    if mode == 'random':
        sample = eligible.sample(n=remaining)
    elif mode == 'quantile':
        sample = eligible.groupby(pd.qcut(eligible['pld'], remaining)).sample(n=1)
    else:
        raise ValueError("Unsupported mode")

    for mof in sample[sample.columns[0]]:
        make_simulation_input(mof, tpl, cfg, out, raspa,gcfg)
        conn.execute(f"UPDATE {TABLE} SET initial_sample=1 WHERE {eligible.columns[0]}=?", (mof,))

    conn.commit()
    conn.close()
    print(f"✅ Added {remaining} new initial inputs (mode={mode}).")


def initial_run(db_path: Path, config_path: Path, ncpus: int, gcfg : Path):
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    gcfg = json.loads(gcfg.read_text(encoding='utf-8'))
    raspa = Path(gcfg['RASPA_DIR'])
    base_dir = Path("initial_gcmc")


    targets = df[(df['initial_sample'] == str(1)) & (df['iteration'].isna())][df.columns[0]].tolist()

    total = len(df[df["initial_sample"] == str(1)])
    to_run = len(targets)
    print(f"▶ RUN: {to_run}/{total} pending using {ncpus} CPUs")
    run_logger.info(f"Start run: {to_run}/{total}, CPUs={ncpus}")
    results = Parallel(n_jobs=ncpus)(
        delayed(run_simulation)(mof, raspa) for mof in targets
    )
    time.sleep(1)
    if to_run == 0:
        return
    recent = deque(maxlen=WINDOW_SIZE)
    done = 0

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpus) as executor:
        futures = {executor.submit(run_simulation, mof, raspa): mof for mof in targets}
        for future in concurrent.futures.as_completed(futures):
            mof = futures[future]
            try:
                mof, uptake, elapsed = future.result()
                done += 1
                recent.append(elapsed)
                win_avg = sum(recent) / len(recent)
                remain = to_run - done
                eta_sec = win_avg * remain / ncpus
                eta_min = eta_sec / 60

                msg = (
                    f"{done}/{to_run} completed: {mof} took {elapsed:.2f}s | "
                    f"win_avg({len(recent)})={win_avg:.2f}s | "
                    f"ETA@{ncpus}cpus={eta_sec:.2f}s({eta_min:.2f}min)"
                )
                run_logger.info(msg)
                print(msg)
                uptake_logger.info(f"{mof}, uptake: {uptake:.6f}")

                # update DB
                conn = get_db_connection(db_path)
                conn.execute(
                    f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, iteration=0 WHERE {targets.columns[0]}=?",
                    (uptake, elapsed, mof)
                )
                conn.commit()
                conn.close()

            except Exception as e:
                error_logger.error(f"Error processing {mof}: {e}", exc_info=True)
    print("✅ Simulations and DB update complete.")

# Main dispatcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['initial_gcmc', 'init_model', 'active_gcmc'])
    parser.add_argument('action', nargs='?', default=None)
    parser.add_argument('--initial_mode', default='random')
    parser.add_argument('-n', '--ncpus', type=int, default=1)
    args = parser.parse_args()

    dbp = Path('mof_project.db')
    cfg = Path('active_learning_config.json')
    gcfg = Path('gcmcconfig.json')
    binput = Path('base.input')
    print(args.phase) 
    if args.phase == 'initial_gcmc':
        if args.action == 'create':
            initial_create(dbp, cfg, binput, args.initial_mode,gcfg)
        elif args.action == 'run':
            initial_run(dbp, cfg, args.ncpus, gcfg)

    elif args.phase == 'init_model':
        print("🔧 Placeholder: init_model not implemented yet.")

    elif args.phase == 'active_gcmc':
        print("🔧 Placeholder: active_gcmc not implemented yet.")

if __name__ == '__main__':
    main()
