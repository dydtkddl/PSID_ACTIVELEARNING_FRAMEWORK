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
import logging
import random
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed, parallel_backend
import pyrascont
import sys

# Constants
TABLE = 'active_learning_gcmc'
LOG_DIR = Path.cwd() / 'logs'
LOG_DIR.mkdir(exist_ok=True)
WINDOW_SIZE = 10 


###############################################################################################
###############################################################################################
###############################################################################################
#
# 
# 
#                                   general 관련 함수 
# 
# 
# 
###############################################################################################
###############################################################################################
###############################################################################################
def get_db_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn










###############################################################################################
###############################################################################################
###############################################################################################
#
# 
# 
#                                   initial_gcmc 관련 함수 
# 
# 
# 
###############################################################################################
###############################################################################################

###############################################################################################
# Active GCMC Logger
active_logger = logging.getLogger('active_gcmc')
active_handler = logging.FileHandler(LOG_DIR / 'active_gcmc.log')
active_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
active_logger.addHandler(active_handler)
active_logger.setLevel(logging.INFO)

run_logger = logging.getLogger('active_run')
uptake_logger = logging.getLogger('active_uptake')
error_logger = logging.getLogger('active_error')
for logger, name in [(run_logger, 'run.log'), (uptake_logger, 'uptake.log'), (error_logger, 'error.log')]:
    handler = logging.FileHandler(LOG_DIR / name)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if logger != error_logger else logging.ERROR)
def make_simulation_input(mof: str, base_template: str, params: dict, out_root: Path, raspa_dir: Path, gcfg : dict) -> None:
    cif_path = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / mof
    print(cif_path)
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
    print(dest)
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
def run_simulation(mof: str, simulation_root: Path, raspa_dir: Path):
    """
    Run RASPA simulate for any given iteration directory:
      simulation_root/<MOF>/simulation.input
    Returns (mof, uptake, elapsed).
    """
    d = simulation_root / mof
    start = time.time()
    result = subprocess.run(
        f"{raspa_dir}/bin/simulate simulation.input",
        shell=True, cwd=d,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    elapsed = time.time() - start

    uptake = None
    if result.returncode != 0:
        stderr_output = result.stderr.strip()
        if "core dumped" in stderr_output.lower() or "malloc(): corrupted" in stderr_output.lower():
            error_logger.error(f"[CORE DUMP] Simulation failed for {mof}: {stderr_output}")
        else:
            error_logger.error(f"Simulation failed for {mof} (code={result.returncode}): {stderr_output}")
    else:
        uptake = parse_output(d)["Average loading absolute [mol/kg framework]"]

    return mof, uptake, elapsed


def initial_create(db_path: Path, config_path: Path, base_input: Path, mode: str, gcfg : Path):
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    gcfg = json.loads(gcfg.read_text(encoding='utf-8'))
    raspa = Path(gcfg['RASPA_DIR'])
    tpl = base_input.read_text(encoding='utf-8')
    out = Path('initial_gcmc')
    frac = cfg['initial_fraction']
    # 먼저 실패한 MOFs에 대해 active_learning_gcmc 테이블도 -1로 설정
    failed_mofs = conn.execute(
        f"SELECT {df.columns[0]} FROM low_pressure_gcmc WHERE completed = -1"
    ).fetchall()
    for (mof,) in failed_mofs:
        conn.execute(
            f"UPDATE active_learning_gcmc SET iteration = -1 WHERE {df.columns[0]} = ?",
            (mof,)
        )
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
def initial_run(db_path: Path, config_path: Path, ncpus: int, gcfg_path: Path, node_map: dict,mof_list:Path):
    """
    Run initial GCMC locally or distributed across nodes via SSH using node_map keys.
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
    gcfg = json.loads(gcfg_path.read_text(encoding='utf-8'))
    raspa = Path(gcfg['RASPA_DIR'])

    # Prepare full target list
    all_targets = df[(df['initial_sample'] == str(1)) & (df['iteration'].isna())][df.columns[0]].tolist()
    total = len(all_targets)
    if total == 0:
        print("▶ No pending MOFs to run.")
        return

    # Distributed mode
    if node_map:
        total_cpus = sum(node_map.values())
        assigned = {}
        start = 0
        # partition all_targets based on cpu ratios
        for node, cpus in node_map.items():
            count = round(total * cpus / total_cpus)
            assigned[node] = all_targets[start:start+count]
            start += count
        # assign any remainder to last node
        if start < total:
            last_node = list(node_map.keys())[-1]
            assigned[last_node] += all_targets[start:]

        procs = []
        project_dir = Path.cwd()
        # dispatch SSH commands
        for node, cpus in node_map.items():
            mofs = assigned[node]
            # inline command passes MOF subset via environment var or file
            list_file = project_dir / f'mofs_{node}.txt'
            list_file.write_text("\n".join(mofs), encoding='utf-8')
            ssh_cmd = (
                f"ssh {node} 'cd {project_dir} && "
                f"python active_learning_gcmc.py initial_gcmc run "
                f"--pal_nodes none --ncpus {cpus} --mof_list {list_file}'"
            )
            procs.append(subprocess.Popen(ssh_cmd, shell=True))
        for p in procs:
            p.wait()
        print("✅ Distributed initial_run complete.")
        return
    # Local mode
    if mof_list is not None:
        targets = Path(mof_list).read_text().splitlines()
    else:
        targets = all_targets
    cpus = ncpus
    print(f"▶ Local   {len(targets)}/{total} MOFs → {cpus} CPUs")
    run_logger.info(f"Start run ({hostname}): {len(targets)}/{total}, CPUs={cpus}")

    recent = deque(maxlen=WINDOW_SIZE)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpus) as executor:
        futures = {executor.submit(run_simulation, mof,Path("initial_gcmc"), raspa): mof for mof in targets}
        for future in concurrent.futures.as_completed(futures):
            mof = futures[future]
            try:
                mof, uptake, elapsed = future.result()
                done += 1
                recent.append(elapsed)
                win_avg = sum(recent) / len(recent)
                remain = len(targets) - done
                eta_sec = win_avg * remain / cpus
                eta_min = eta_sec / 60
                msg = (
                    f"({hostname}) {done}/{len(targets)} completed: {mof} took {elapsed:.2f}s | "
                    f"win_avg({len(recent)})={win_avg:.2f}s | "
                    f"ETA@{cpus}cpus={eta_sec:.2f}s({eta_min:.2f}min)"
                )
                run_logger.info(msg)
                print(msg)
                uptake_logger.info(f"{mof}, uptake: {uptake:.6f}")
                # update DB
                conn = get_db_connection(db_path)
                conn.execute(
                    f"UPDATE {TABLE} SET `uptake[mol/kg framework]`=?, calculation_time=?, iteration=0 WHERE {df.columns[0]}=?",
                    (uptake, elapsed, mof)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                error_logger.error(f"Error processing {mof}: {e}", exc_info=True)
    print("✅ Local initial_run complete.")








###############################################################################################
###############################################################################################
###############################################################################################
#
# 
# 
#                                   init_model 관련 함수 
# 
# 
# 
###############################################################################################
###############################################################################################
###############################################################################################
# Logging
logger = logging.getLogger('init_model')
handler = logging.FileHandler('logs/init_model.log')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Phase completion check
def check_phase_completion(db_path: Path, table: str, flag_column: str, flag_value) -> (bool, int, int):
    """
    Returns (is_complete, count_matching, total_rows)
    """
    conn = get_db_connection(db_path)
    df = pd.read_sql(f"SELECT {flag_column} FROM {table}", conn)
    conn.close()
    total = len(df)
    count = (df[flag_column] == flag_value).sum()
    return (count == total, count, total)

# Dataset loader
def load_active_learning_dataset(db_path: Path, al_cfg: Path, gcfg: Path) -> pd.DataFrame:
    """
    Build DataFrame: MOF name, iteration, low & high pressure uptakes, input features.
    """
    gcfg = json.loads(gcfg.read_text(encoding='utf-8'))
    al_cfg = json.loads(al_cfg.read_text(encoding='utf-8'))
    conn = get_db_connection(db_path)
    df = pd.read_sql("SELECT * FROM active_learning_gcmc", conn)
    df = df[df["iteration"] != "-1"]
    low_df = pd.read_sql(
        "SELECT * FROM low_pressure_gcmc",
        conn
    )
    low_df = low_df[["filename",'uptake[mol/kg framework]']]
    conn.close()
    lp = gcfg['ExternalPressure_LOW']
    hp = gcfg['ExternalPressure']
    lp_col = f"{lp}bar uptake"
    hp_col = f"{hp}bar uptake"
    df.rename(columns={'uptake[mol/kg framework]': hp_col}, inplace=True)
    df[lp_col] = df['filename'].map(
        low_df.set_index('filename')['uptake[mol/kg framework]']
    )
    features = al_cfg['input_features']
    cols = [df.columns[0], 'iteration', lp_col] + features + [hp_col]
    df = df[cols]
    return df
# Split datasets
def split_datasets(df: pd.DataFrame, model_dir: Path) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split into full, labeled (iteration==0), unlabeled (iteration is null).
    Save CSVs.
    """
    labeled = df[df['iteration'] == 0]
    unlabeled = df[df['iteration'].isna()]
    model_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_dir/'full_dataset.csv', index=False)
    labeled.to_csv(model_dir/'labeled.csv', index=False)
    unlabeled.to_csv(model_dir/'unlabeled.csv', index=False)
    return df, labeled, unlabeled
# Neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list):
        super().__init__()
        layers=[]; prev=input_dim
        for spec in hidden_layers:
            layers.append(nn.Linear(prev, spec['hidden_dim']))
            layers.append(getattr(nn, spec['activation_func'])())
            layers.append(nn.Dropout(spec['dropout']))
            prev = spec['hidden_dim']
        layers.append(nn.Linear(prev,1))
        self.net=nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# Training routine
def train_model(labeled: pd.DataFrame, al_cfg: Path, model_dir: Path) -> nn.Module:
    """
    Train NN with early stopping. Save model.pth & training_log.csv.
    """
    al_cfg = json.loads(al_cfg.read_text(encoding='utf-8'))
    ups_col = labeled.columns[-1]
    X = labeled.drop(columns=[labeled.columns[0],'iteration', ups_col])
    y = labeled[ups_col].values
    dropped = set(labeled.columns) - set([labeled.columns[0],'iteration', ups_col]) - set(labeled.select_dtypes(include='number').columns)
    if dropped: logger.warning(f"Dropped non-numeric columns: {dropped}")
    X_train, y_train = X, y
    X_arr = np.asarray(X_train, dtype=np.float32)
    y_arr = np.asarray(y_train, dtype=np.float32)
    X_tensor = torch.from_numpy(X_arr)            # shape (N, n_features)
    y_tensor = torch.from_numpy(y_arr).unsqueeze(1)  # shape (N, 1)
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=al_cfg['neural_network']['dataset']['BATCH_SIZE'], shuffle=True)
    spec = al_cfg['neural_network']['model_spec']['hidden_layers']
    model = FeedForwardNN(X.shape[1], spec)
    optimizer = optim.Adam(model.parameters(), lr=al_cfg['neural_network']['training']['learning_rate'])
    criterion = nn.MSELoss()
    patience = al_cfg['neural_network']['training']['patience']
    min_delta = al_cfg['neural_network']['training'].get('min_delta',0.0)
    best_loss=float('inf'); wait=0; logs=[]
    for epoch in range(al_cfg['neural_network']['training']['max_epoch']):
        model.train(); total_loss=0
        for xb,yb in loader:
            optimizer.zero_grad(); preds=model(xb); loss=criterion(preds,yb)
            loss.backward(); optimizer.step(); total_loss+=loss.item()
        avg_loss=total_loss/len(loader)
        X_arr = np.asarray(X_train, dtype=np.float32)
        # metrics
        X_tensor = torch.from_numpy(X_arr)
        with torch.no_grad():
            output = model(X_tensor)                  # Tensor
            preds_all = output.detach().cpu().numpy() # NumPy ndarray
            preds_all = preds_all.reshape(-1)
        r2 = r2_score(y_train,preds_all)
        mae = mean_absolute_error(y_train,preds_all)
        mse = mean_squared_error(y_train,preds_all)
        logs.append((epoch,avg_loss,r2,mae,mse))
        if avg_loss+min_delta < best_loss:
            best_loss=avg_loss; wait=0; torch.save(model.state_dict(), model_dir/'model.pth')
        else:
            wait+=1
            if wait>=patience: break

    pd.DataFrame(logs,columns=['epoch','loss','r2','mae','mse']).to_csv(model_dir/'training_log.csv',index=False)
    return model

# Prediction routine
def predict_with_model(df: pd.DataFrame, model: nn.Module, al_cfg: Path, model_dir: Path) -> pd.DataFrame:
    """
    MC Dropout predictions. Save predictions.csv.
    """
    al_cfg = json.loads(al_cfg.read_text(encoding='utf-8'))
    feat = [c for c in df.columns if c not in [df.columns[0],'iteration', df.columns[-1]] ]
    X_df = df[feat].apply(pd.to_numeric, errors='raise')
    X = X_df.values
    tx = torch.from_numpy(X)  # 이미 float32 여서 dtype 지정 불필요
    tx=torch.tensor(X,dtype=torch.float32)
    mcd=al_cfg["neural_network"]['prediction']['mcd_numbers']; 
    all_preds=[]
    for _ in range(mcd):
        with torch.no_grad(): all_preds.append(model(tx).numpy().reshape(-1))
    arr=np.stack(all_preds); mean=arr.mean(0); std=arr.std(0)
    out=df[[df.columns[0],'iteration']].copy();
    out['pred_mean']=mean; 
    out['pred_std']=std
    actual_col = df.columns[-1]
    actual = pd.to_numeric(df[actual_col], errors='coerce')
    out[actual_col] = actual
    out["absolute_error"] = (out['pred_mean'] - actual).abs()
    out.to_csv(model_dir/'predictions.csv', index=False)
    return out

import concurrent.futures
# ---------------- Active GCMC Functions ----------------

def get_iteration_status(conn: sqlite3.Connection, model_base: Path, n_samples: int, target_fraction: float) -> (int, str):
    # 1) Fetch max iteration from DB, exclude failures (-2)
    raw = conn.execute(f"SELECT MAX(iteration) FROM {TABLE} WHERE iteration NOT IN (-2)").fetchone()[0]
    I_db = int(raw) if raw is not None else 0
    # 2) Determine max iteration from model directories
    dirs = [p for p in model_base.iterdir() if p.name.startswith('iteration')]
    I_dir = 0
    for d in dirs:
        try:
            idx = int(d.name.replace('iteration', ''))
            I_dir = max(I_dir, idx)
        except:
            continue
    I = max(I_db, I_dir)
    # 3) Count GCMC-completed MOFs at iteration I
    count_gcmc = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE} WHERE iteration = ?", (I,)
    ).fetchone()[0]
    # 4) Check if model weights exist for iteration I
    model_exists = (model_base / f'iteration{I:05d}' / 'model.pth').exists()
    # 5) Check overall labeled count vs target
    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    count_labeled = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE} WHERE iteration NOT NULL AND iteration != -2"
    ).fetchone()[0]
    if count_labeled >= total * target_fraction:
        return I, 'finished'
    if count_gcmc < n_samples:
        return I, 'gcmc_pending'
    if count_gcmc == n_samples and not model_exists:
        return I, 'training_pending'
    return I, 'ready_next'


def select_top_uncertain_mofs(model_dir: Path, n_samples: int) -> list:
    df = pd.read_csv(model_dir / 'predictions.csv')
    pending = df[df['iteration'].isna()]
    return pending.sort_values('pred_std', ascending=False).head(n_samples)['filename'].tolist()


def create_active_inputs(mofs: list, tpl: str, params: dict, out_root: Path, raspa_dir: Path, gcfg: dict):
    print(out_root,1)
    for mof in mofs:
        print(mof, '\n\n\n')
        make_simulation_input(mof, tpl, params, out_root, Path(raspa_dir), gcfg)

def run_active_simulations(db_path: Path,
                           raspa_dir: Path,
                           node_map: dict,
                           ncpus: int,
                           iteration: int,
                           mof_list: list,
                           out_root: Path):
    """
    Run Active GCMC locally or distributed across nodes via SSH using node_map keys.
    Writes results into DB with given iteration.
    """
    import socket
    hostname = socket.gethostname()
    conn = get_db_connection(db_path)

    total = len(mof_list)
    if total == 0:
        print("▶ No pending MOFs to run.")
        return

    # Distributed mode
    if node_map:
        total_cpus = sum(node_map.values())
        assigned = {}
        start = 0
        for node, cpus in node_map.items():
            count = round(total * cpus / total_cpus)
            assigned[node] = mof_list[start:start+count]
            start += count
        if start < total:
            assigned[list(node_map.keys())[-1]] += mof_list[start:]

        procs = []
        project_dir = Path.cwd()
        for node, cpus in node_map.items():
            mofs = assigned[node]
            list_file = project_dir / f'active_mofs_{node}.txt'
            list_file.write_text("\n".join(mofs), encoding='utf-8')
            ssh_cmd = (
                f"ssh {node} 'cd {project_dir} && "
                f"python active_learning_gcmc.py active_gcmc "
                f"--pal_nodes none --ncpus {cpus} "
                f"--mof_list {list_file}'"
            )
            procs.append(subprocess.Popen(ssh_cmd, shell=True))
        for p in procs:
            p.wait()
        print("✅ Distributed active_run complete.")
        return

    # Local mode
    cpus = ncpus
    print(f"▶ Local   {total}/{total} MOFs → {cpus} CPUs")
    active_logger.info(f"Start active run ({hostname}): {total}/{total}, CPUs={cpus}")

    recent = deque(maxlen=WINDOW_SIZE)
    done = 0

    def worker(mof):
        return run_simulation(mof, out_root, raspa_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpus) as executor:
        futures = {executor.submit(worker, mof): mof for mof in mof_list}
        for fut in concurrent.futures.as_completed(futures):
            mof = futures[fut]
            try:
                mof, uptake, elapsed = fut.result()
                done += 1
                recent.append(elapsed)
                win_avg = sum(recent) / len(recent)
                remain = total - done
                eta_sec = win_avg * remain / cpus
                eta_min = eta_sec / 60
                msg = (
                    f"({hostname}) {done}/{total} completed: {mof} took {elapsed:.2f}s | "
                    f"win_avg({len(recent)})={win_avg:.2f}s | "
                    f"ETA@{cpus}cpus={eta_sec:.2f}s({eta_min:.2f}min)"
                )
                active_logger.info(msg)
                print(msg)

                # Update DB
                if uptake is None:
                    conn.execute(f"UPDATE {TABLE} SET iteration=-2 WHERE filename=?", (mof,))
                else:
                    conn.execute(
                        f"UPDATE {TABLE} "
                        f"SET iteration=?, `uptake[mol/kg framework]`=?, calculation_time=? "
                        f"WHERE filename=?",
                        (iteration, uptake, elapsed, mof)
                    )
                conn.commit()
            except Exception as e:
                error_logger.error(f"Error processing {mof}: {e}", exc_info=True)

    print("✅ Local active_run complete.")


def retrain_model_for_iteration(iteration: int, al_cfg: Path, gcfg: Path):
    prev = Path('nn_model') / f'iteration{iteration:05d}'
    next_dir = Path('nn_model') / f'iteration{iteration+1:05d}'
    full, labeled, unlabeled = split_datasets(
        load_active_learning_dataset(Path('mof_project.db'), al_cfg, gcfg), next_dir
    )
    model = FeedForwardNN(labeled.shape[1]-3, json.loads(al_cfg.read_text())['neural_network']['model_spec']['hidden_layers'])
    model.load_state_dict(torch.load(prev/'model.pth'))
    trained = train_model(labeled, al_cfg, next_dir)
    predict_with_model(full, trained, al_cfg, next_dir)

# ---------------- End Active GCMC Functions ----------------



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['initial_gcmc', 'init_model', 'active_gcmc'])
    parser.add_argument('action', nargs='?', default=None)
    parser.add_argument('--initial_mode', default='random')
    parser.add_argument('-n', '--ncpus', type=int, default=1)

    ################################################################################################
    parser.add_argument('--pal_nodes', type=str, default=None,
                    help='JSON map of node:cpu_count for distributed run')
    parser.add_argument('--mof_list', type=Path, default=None,
                    help='Path to file listing MOFs to run (for distributed or testing)')

    ##################################################################################################

    args = parser.parse_args()

    dbp = Path('mof_project.db')
    cfg = Path('active_learning_config.json')
    gcfg = Path('gcmcconfig.json')
    binput = Path('base.input')
    node_map = {}
    if args.pal_nodes and args.pal_nodes.lower() != "none":
        node_map = json.loads(args.pal_nodes)
    else:   
        node_map = {}
    mof_list = args.mof_list
    print(args.phase) 
    if args.phase == 'initial_gcmc':
        if args.action == 'create':
            initial_create(dbp, cfg, binput, args.initial_mode,gcfg)
        elif args.action == 'run':
            initial_run(dbp, cfg, args.ncpus, gcfg, node_map,mof_list)

    elif args.phase == 'init_model':
        # print("🔧 Placeholder: init_model not implemented yet.")
    # 1) Check previous phases
        _, c1, t1 = check_phase_completion(dbp, 'active_learning_gcmc', 'initial_sample', 1)
        _, c2, t2 = check_phase_completion(dbp, 'active_learning_gcmc', 'iteration', 0)
        _, c3, t3 = check_phase_completion(dbp, 'low_pressure_gcmc', 'completed', 1)
        _, c4, t4 = check_phase_completion(dbp, 'low_pressure_gcmc', 'completed', -1)

        if not (c1 == c2) and (t4 == c3 + c4):
            # 상세한 오류 메시지 구성
            error_msg = (
                f"선행 단계 미완료: "
                f"초기 샘플 {c1}/{t1}, "
                f"초기 반복 {c2}/{t2}, "
                f"저압 GCMC {c3}/{t3}"
            )
            logger.error(error_msg)
            print("오류: 선행 단계가 완료되지 않았습니다. 자세한 내용은 로그를 확인하세요.")
            sys.exit(1)


        # 2) Load and prepare data
        df = load_active_learning_dataset(dbp, cfg, gcfg)
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')

        # 2) NaN 행 삭제
        iterations_list = df['iteration'].dropna()
        # 3) float→int 캐스팅
        iterations_list = iterations_list.astype(int).to_list()
        # 3) Check for existing training
        flag = 0
        for i in iterations_list:
            if i >= 1:
                flag = 1
                break 
        if flag == 1:
            ans = input("Existing training detected. Reinitialize? (y/n): ")
            if ans.lower() != 'y': sys.exit(0)

        # 4) Split and save datasets
        model_dir = Path('nn_model') / 'iteration00000'
        full, labeled, unlabeled = split_datasets(df, model_dir)

        # 5) Train model
        model = train_model(labeled, cfg, model_dir)

        # 6) Predict and save results
        pred = predict_with_model(full, model, cfg, model_dir)
        print('init_model phase completed successfully.')

    elif args.phase == 'active_gcmc':
        conn = get_db_connection(dbp)
        al_cfg = json.loads(cfg.read_text())
        gcfg_d = json.loads(gcfg.read_text())
        tpl = binput.read_text()
        raspa = Path(gcfg_d['RASPA_DIR'])
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

        while True:
            I, status = get_iteration_status(conn, Path('nn_model'), al_cfg['n_samples'], al_cfg['target_fraction'])
            active_logger.info(f"Iteration {I}, status={status}")
            if status == 'finished':
                active_logger.info("Target fraction reached, exiting.")
                print("✅ Active learning complete.")
                break
            if status == 'gcmc_pending':
                prev_iter = I - 1 if I > 0 else 0
                model_dir = Path('nn_model') / f'iteration{prev_iter:05d}'
                mofs = select_top_uncertain_mofs(model_dir, al_cfg['n_samples'])
                out_root = Path('active_gcmc') / f'iteration{I:05d}'
                out_root.mkdir(parents=True, exist_ok=True)
                create_active_inputs(mofs, tpl, al_cfg, out_root, raspa, gcfg_d)
                run_active_simulations(
                        dbp,                # 1) DB 파일 경로
                        raspa,              # 2) RASPA_DIR
                        node_map,           # 3) 노드 맵
                        args.ncpus,         # 4) 로컬 CPU 수
                        I,                  # 5) 이번 iteration 번호
                        mofs,               # 6) 샘플링된 MOF 리스트
                        out_root            # 7) 입력/출력 루트 디렉터리
                    )
                status = 'training_pending'

            if status in ('training_pending', 'ready_next'):
                retrain_model_for_iteration(I, cfg, gcfg)

if __name__ == '__main__':
    main()
