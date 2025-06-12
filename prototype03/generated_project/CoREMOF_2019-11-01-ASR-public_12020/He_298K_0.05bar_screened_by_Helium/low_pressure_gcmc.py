#!/usr/bin/env python3
"""
Low-pressure GCMC simulation workflow using RASPA and SQLite.
Generates input files, runs simulations in parallel, parses outputs,
and updates a SQLite database with uptake results.
"""
import argparse
import json
import logging
import multiprocessing
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from threading import Lock

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

import pyrascont

def setup_logging(logs_dir: Path):
    """
    Configure logging with a multiprocessing queue, rotating existing logs.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_log = logs_dir / 'progress.log'
    complete_log = logs_dir / 'completed.log'

    def rotate(path: Path):
        if path.exists():
            idx = 1
            while True:
                new_path = path.with_name(f"{path.stem}.{idx}{path.suffix}")
                if not new_path.exists():
                    path.rename(new_path)
                    break
                idx += 1

    rotate(progress_log)
    rotate(complete_log)

    queue = multiprocessing.Queue(-1)
    handler = QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    fmt = logging.Formatter('%(asctime)s | %(message)s')
    prog_handler = logging.FileHandler(progress_log)
    prog_handler.setFormatter(fmt)
    prog_handler.addFilter(lambda rec: rec.name == 'progress')

    comp_handler = logging.FileHandler(complete_log)
    comp_handler.setFormatter(logging.Formatter('%(message)s'))
    comp_handler.addFilter(lambda rec: rec.name == 'complete')

    listener = QueueListener(queue, prog_handler, comp_handler)
    listener.start()

    return listener, progress_log, complete_log


def append_safe(filename: Path, text: str, lock: Lock):
    """Thread-safe append to text file."""
    with lock:
        filename.write_text(filename.read_text() + text + '\n', encoding='utf-8')


def load_completed(completed_file: Path) -> set:
    """Read completed MOFs from log."""
    if not completed_file.exists():
        return set()
    return set(line.strip() for line in completed_file.read_text(encoding='utf-8').splitlines() if line.strip())


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection, raising if missing."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(db_path)


def make_simulation_input(mof: str,
                          base_template: str,
                          params: dict,
                          out_root: Path,
                          raspa_dir: Path,
                          logger: logging.Logger) -> None:
    """
    Convert CIF to unit cell, format RASPA input, and write to directory.
    """
    cif_path = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(str(cif_path), float(params['CUTOFFVDW']), Display=False)
        unitcell_str = ' '.join(map(str, ucell))
    except Exception as exc:
        logger.warning(f"Failed to convert CIF for {mof}: {exc}")
        return

    content = base_template.format(
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
        UNITCELL=unitcell_str
    )

    dest_dir = out_root / mof
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / 'simulation.input').write_text(content, encoding='utf-8')
    logger.info(f"Created input for {mof}")


def parse_output(mof_dir: Path) -> dict:
    """
    Parse .data file for uptake values.
    """
    data_root = mof_dir / 'Output' / 'System_0'
    data_files = list(data_root.glob('*.data'))
    if not data_files:
        raise FileNotFoundError(f"No .data file in {data_root}")
    text = data_files[0].read_text()
    keys = [
        "Average loading absolute [mol/kg framework]",
        "Average loading excess [mol/kg framework]",
        "Average loading absolute [molecules/unit cell]",
        "Average loading excess [molecules/unit cell]"
    ]
    results = {}
    for key in keys:
        try:
            part = text.split(key)[1]
            val = float(part.split("+/-")[0].split()[-1])
            results[key] = val
        except Exception:
            continue
    if 'Average loading absolute [mol/kg framework]' not in results:
        raise ValueError(f"Failed to parse data for {mof_dir.name}")
    return results


def run_simulation(task, db_path: Path, lock: Lock, progress_log: Path, complete_log: Path):
    mof, raspa_dir = task
    mof_dir = Path('low_pressure_gcmc') / mof
    simulate_cmd = f"{raspa_dir}/bin/simulate simulation.input"
    start = time.time()
    subprocess.run(simulate_cmd, shell=True, check=True, cwd=mof_dir)
    elapsed = time.time() - start

    uptake_data = parse_output(mof_dir)
    uptake = uptake_data["Average loading absolute [mol/kg framework]"]

    # Update database
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, completed=1 WHERE {FIRST_COL}=?",
            (uptake, elapsed, mof)
        )
        conn.commit()

    # Log progress
    append_safe(complete_log, mof, lock)
    msg = f"{mof} | uptake={uptake:.4f} mol/kg | time={elapsed:.1f}s"
    append_safe(progress_log, msg, lock)
    return mof, uptake, elapsed


def cmd_create(db_path: Path, config_path: Path, base_input: Path, threads: int):
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    mofs = df.iloc[:, 0].astype(str).tolist()

    config = json.loads(config_path.read_text(encoding='utf-8'))
    raspa_dir = Path(config['RASPA_DIR'])
    base_tpl = base_input.read_text(encoding='utf-8')
    out_root = Path('low_pressure_gcmc')

    logger = logging.getLogger('progress')
    print(f"▶ CREATE: {len(mofs)} MOFs using {threads} threads")
    with parallel_backend('threading', n_jobs=threads):
        Parallel()(delayed(make_simulation_input)(m, base_tpl, config, out_root, raspa_dir, logger)
                   for m in tqdm(mofs, desc="Generating inputs"))
    print("✅ Input generation complete.")


def cmd_run(db_path: Path, config_path: Path, threads: int, progress_log: Path, complete_log: Path):
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    mofs = df.iloc[:, 0].astype(str).tolist()
    completed = load_completed(complete_log)
    pending = [m for m in mofs if m not in completed]

    total = len(mofs)
    to_run = len(pending)
    print(f"▶ RUN: {to_run}/{total} simulations pending")
    if to_run == 0:
        print("No simulations to run.")
        return

    # Log start
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    append_safe(progress_log, f"---- START {now} | Remaining: {to_run}/{total} ----", threading.Lock())

    config = json.loads(config_path.read_text(encoding='utf-8'))
    raspa_dir = Path(config['RASPA_DIR'])
    tasks = [(m, raspa_dir) for m in pending]
    lock = Lock()

    with parallel_backend('threading', n_jobs=threads):
        results = Parallel()(delayed(run_simulation)(task, db_path, lock, progress_log, complete_log)
                              for task in tqdm(tasks, desc="Running sims"))

    print("✅ Simulations complete.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    create_p = sub.add_parser('create', help='Generate simulation inputs')
    create_p.add_argument('-n', '--threads', type=int, default=1)
    run_p = sub.add_parser('run', help='Execute simulations and parse results')
    run_p.add_argument('-n', '--threads', type=int, default=1)

    args = parser.parse_args()

    BASE_DIR = Path.cwd()
    DB_PATH = BASE_DIR / 'mof_project.db'
    CONFIG = BASE_DIR / 'gcmcconfig.json'
    BASE_INPUT = BASE_DIR / 'base.input'
    LOG_DIR = BASE_DIR / 'logs' / 'low_pressure_gcmc'

    listener, prog_log, comp_log = setup_logging(LOG_DIR)

    try:
        if args.cmd == 'create':
            cmd_create(DB_PATH, CONFIG, BASE_INPUT, args.threads)
        elif args.cmd == 'run':
            cmd_run(DB_PATH, CONFIG, args.threads, prog_log, comp_log)
    except Exception as e:
        logging.getLogger().error(f"Error: {e}")
        sys.exit(1)
    finally:
        listener.stop()


if __name__ == '__main__':
    main()
