#!/usr/bin/env python3
"""
Low-pressure GCMC simulation workflow using RASPA and SQLite.
Generates input files, runs simulations in parallel, parses outputs,
updates a SQLite database with uptake results, and logs detailed progress.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
from collections import deque

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import pyrascont

# Constants
TABLE = 'low_pressure_gcmc'
FIRST_COL = None
WINDOW_SIZE = 10  # for moving average ETA

# Setup log directory
LOG_DIR = Path.cwd() / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Main run logger (run.log)
LOG_FILE = LOG_DIR / 'run.log'
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
import socket

hostname = socket.gethostname()
LOG_FILE = LOG_DIR / f'run_{hostname}.log'
uptake_log_file = LOG_DIR / f'uptake_{hostname}.log'
error_log_file = LOG_DIR / f'error_{hostname}.log'
# Uptake logger (uptake.log)
run_logger = logging.getLogger('gcmc_run')
uptake_logger = logging.getLogger('gcmc_uptake')
uptake_handler = logging.FileHandler(uptake_log_file)
uptake_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
uptake_logger.addHandler(uptake_handler)
uptake_logger.setLevel(logging.INFO)

# Error logger (error.log)
error_logger = logging.getLogger('gcmc_error')
error_handler = logging.FileHandler(error_log_file)
error_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """
    Return a SQLite connection that waits up to 30s for locks,
    uses WAL mode, and sets busy_timeout to 30s.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def make_simulation_input(mof: str,
                          base_template: str,
                          params: dict,
                          out_root: Path,
                          raspa_dir: Path) -> None:
    """
    Convert CIF to unit cell, format RASPA input, and write to directory.
    """
    cif_path = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    try:
        ucell = pyrascont.cif2Ucell(
            str(cif_path), float(params.get('CUTOFFVDW', 12.8)), Display=False
        )
        unitcell_str = ' '.join(map(str, ucell))
    except Exception as e:
        error_logger.error(f"Input gen failed for {mof}: {e}", exc_info=True)
        return

    content = base_template.format(
        NumberOfCycles=params['NumberOfCycles'],
        NumberOfInitializationCycles=params['NumberOfInitializationCycles'],
        PrintEvery=params['PrintEvery'],
        UseChargesFromCIFFile=params['UseChargesFromCIFFile'],
        Forcefield=params['Forcefield'],
        TEMP=params['ExternalTemperature'],
        PRESSURE=float(params['ExternalPressure_LOW']) * 1e5,
        GAS=params['GAS'],
        MoleculeDefinition=params.get('MoleculeDefinition', ''),
        MOF=mof,
        UNITCELL=unitcell_str
    )

    dest = out_root / mof
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'simulation.input').write_text(content, encoding='utf-8')


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
    """
    Run RASPA simulate and return uptake and elapsed time.
    """
    d = Path('low_pressure_gcmc') / mof
    start = time.time()
    subprocess.run(
        f"{raspa_dir}/bin/simulate simulation.input",
        shell=True, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    elapsed = time.time() - start
    uptake = parse_output(d)["Average loading absolute [mol/kg framework]"]
    return mof, uptake, elapsed


def cmd_create(db_path: Path, config_path: Path, base_input: Path, ncpus: int):
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()

    global FIRST_COL
    FIRST_COL = df.columns[0]

    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    raspa = Path(cfg['RASPA_DIR'])
    tpl = base_input.read_text(encoding='utf-8')
    out = Path('low_pressure_gcmc')

    print(f"▶ CREATE: {len(df)} MOFs using {ncpus} CPUs")
    with parallel_backend('threading', n_jobs=ncpus):
        Parallel()(delayed(make_simulation_input)(m, tpl, cfg, out, raspa)
                   for m in pd.Series(df[FIRST_COL].astype(str)).tolist())
    print("✅ Input generation complete.")


def cmd_run(db_path: Path, config_path: Path, ncpus: int, node_map: dict, mof_list: Path):
    """
    Run GCMC simulations either locally or in distributed mode.
    node_map: dict mapping node_name to cpu_count.
    """
    import subprocess
    from pathlib import Path
    import json
    import time
    from collections import deque
    import concurrent.futures
    import socket

    hostname = socket.gethostname()

    # Load database and config
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    gcfg = json.loads(config_path.read_text(encoding='utf-8'))
    raspa = Path(gcfg['RASPA_DIR'])

    # Prepare full target list
    all_targets = df[df['completed'].isna()][df.columns[0]].astype(str).tolist()

    if not all_targets:
        print(f"▶ No pending MOFs to run .")
        return

    # Distributed mode
    if node_map:
        total_cpus = sum(node_map.values())
        assigned = {}
        start = 0
        for node, cpus in node_map.items():
            count = round(len(all_targets) * cpus / total_cpus)
            assigned[node] = all_targets[start:start + count]
            start += count
        if start < len(all_targets):
            last_node = list(node_map.keys())[-1]
            assigned[last_node] += all_targets[start:]

        procs = []
        project_dir = Path.cwd()
        for node, cpus in node_map.items():
            mofs = assigned[node]
            list_file = project_dir / f'mofs_{node}.txt'
            list_file.write_text("\n".join(mofs), encoding='utf-8')
            ssh_cmd = (
                f"ssh {node} 'cd {project_dir} && "
                f"python low_pressure_gcmc.py run "
                f"--pal_nodes none --ncpus {cpus} --mof_list {list_file}'"
            )
            procs.append(subprocess.Popen(ssh_cmd, shell=True))
        for p in procs:
            p.wait()
        print("✅ Distributed run complete.")
        return

    # Local mode
    if mof_list is not None:
        targets = Path(mof_list).read_text().splitlines()
    else:
        targets = all_targets

    total = len(df)
    to_run = len(targets)

    print(f"▶ RUN ({hostname}): {to_run}/{total} pending using {ncpus} CPUs")
    run_logger.info(f"Start run ({hostname}): {to_run}/{total}, CPUs={ncpus}")
    time.sleep(1)
    if to_run == 0:
        return

    recent = deque(maxlen=WINDOW_SIZE)
    done = 0
    batch_updates = []
    BATCH_SIZE = 40

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
                    f"({hostname}) {done}/{to_run} completed: {mof} took {elapsed:.2f}s | "
                    f"win_avg({len(recent)})={win_avg:.2f}s | "
                    f"ETA@{ncpus}cpus={eta_sec:.2f}s({eta_min:.2f}min)"
                )
                run_logger.info(msg)
                print(msg)
                uptake_logger.info(f"{mof}, uptake: {uptake:.6f}")

                batch_updates.append((uptake, elapsed, mof))

                if len(batch_updates) >= BATCH_SIZE:
                    conn = get_db_connection(db_path)
                    conn.executemany(
                        f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, completed=1 "
                        f"WHERE {df.columns[0]}=?",
                        batch_updates
                    )
                    conn.commit()
                    conn.close()
                    batch_updates.clear()

            except Exception as e:
                error_logger.error(f"Error processing {mof}: {e}", exc_info=True)

    # 최종 남은 업데이트 실행
    if batch_updates:
        conn = get_db_connection(db_path)
        conn.executemany(
            f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, completed=1 "
            f"WHERE {df.columns[0]}=?",
            batch_updates
        )
        conn.commit()
        conn.close()

    print("✅ Simulations and DB update complete.")
def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest='cmd', required=True)

    c = sp.add_parser('create', help='Generate simulation inputs')
    c.add_argument('-n', '--ncpus', type=int, default=1)

    r = sp.add_parser('run', help='Run simulations and update DB')
    r.add_argument('-n', '--ncpus', type=int, default=1)
    r.add_argument('--pal_nodes', type=str, default=None,
                  help='JSON map of node:cpu_count for distributed run')
    r.add_argument('--mof_list', type=Path, default=None,
                  help='Path to file listing MOFs to run (for distributed or testing)')

    args = parser.parse_args()

    node_map = {}
    if hasattr(args, 'pal_nodes') and args.pal_nodes and args.pal_nodes.lower() != "none":
        node_map = json.loads(args.pal_nodes)

    mof_list = getattr(args, 'mof_list', None)

    base = Path.cwd()
    dbp = base / 'mof_project.db'
    cfg = base / 'gcmcconfig.json'
    binput = base / 'base.input'

    if args.cmd == 'create':
        cmd_create(dbp, cfg, binput, args.ncpus)
    else:
        cmd_run(dbp, cfg, args.ncpus, node_map, mof_list)

if __name__ == '__main__':
    main()
