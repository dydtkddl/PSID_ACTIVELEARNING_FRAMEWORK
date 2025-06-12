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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyrascont
import pandas as pd
from tqdm import tqdm

# Constants
DB_PATH = Path.cwd() / 'mof_project.db'
TABLE = 'low_pressure_gcmc'
CSV_NAME = './low_pressure_gcmc.csv'  # kept for fallback or legacy reference
FIRST_COL = None  # will be set after loading table


def get_connection():
    """
    SQLite 연결을 반환
    """
    if not DB_PATH.exists():
        print(f"✖ DB 파일이 없습니다: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    return conn


def _make_simulation_input(mof, base_tpl, gcmc_params, out_root, raspa_dir):
    """
    단일 MOF 폴더에 simulation.input 생성
    """
    cif_file = raspa_dir / 'share' / 'raspa' / 'structures' / 'cif' / f"{mof}"
    try:
        res_ucell = pyrascont.cif2Ucell(str(cif_file), float(gcmc_params.get("CUTOFFVDW", 12.8)), Display=False)
        unitcell_str = ' '.join(map(str, res_ucell))
    except Exception as e:
        print(f"⚠ CIF 변환 실패 for {mof}: {e}")
        return mof

    try:
        sim_content = base_tpl.format(
            NumberOfCycles=gcmc_params["NumberOfCycles"],
            NumberOfInitializationCycles=gcmc_params["NumberOfInitializationCycles"],
            PrintEvery=gcmc_params["PrintEvery"],
            UseChargesFromCIFFile=gcmc_params["UseChargesFromCIFFile"],
            Forcefield=gcmc_params["Forcefield"],
            TEMP=gcmc_params["ExternalTemperature"],
            PRESSURE=float(gcmc_params["ExternalPressure"]) * 100000,
            GAS=gcmc_params["GAS"],
            MoleculeDefinition=gcmc_params.get("MoleculeDefinition", ""),
            MOF=mof,
            UNITCELL=unitcell_str
        )
        mof_dir = out_root / mof
        mof_dir.mkdir(exist_ok=True)
        with open(mof_dir / 'simulation.input', 'w') as fw:
            fw.write(sim_content)
    except Exception as e:
        print(f"⚠ simulation.input 생성 실패 for {mof}: {e}")
    return mof


def cmd_create(ncpus):
    """
    DB에서 MOF 목록을 불러와 simulation.input 생성
    """
    conn = get_connection()
    # pandas로 전체 테이블 로드
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()

    global FIRST_COL
    FIRST_COL = df.columns[0]

    # 설정 파일 로드
    cfg_path = Path('gcmcconfig.json')
    if not cfg_path.exists():
        print("✖ gcmcconfig.json 파일이 없습니다.")
        sys.exit(1)
    with open(cfg_path) as f:
        gcmc_params = json.load(f)
    raspa_dir = Path(gcmc_params.get("RASPA_DIR", ""))

    # 템플릿 로드
    base_tpl_path = Path('base.input')
    if not base_tpl_path.exists():
        print("✖ base.input 파일이 없습니다.")
        sys.exit(1)
    base_tpl = base_tpl_path.read_text()

    # 출력 폴더
    out_root = Path('low_pressure_gcmc')
    out_root.mkdir(exist_ok=True)

    mo_list = df[FIRST_COL].astype(str).tolist()
    total = len(mo_list)
    print(f"▶ CREATE: {total}개 MOF, {ncpus}개 프로세스로 simulation.input 생성 시작")

    if ncpus > 1:
        with ProcessPoolExecutor(max_workers=ncpus) as exe:
            futures = {exe.submit(_make_simulation_input, mof, base_tpl, gcmc_params, out_root, raspa_dir): mof for mof in mo_list}
            for fut in tqdm(as_completed(futures), total=total, desc="📁 CREATE 시뮬레이션 입력 파일 생성 중"):
                mof = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"✖ {mof} 생성 실패: {e}")
    else:
        for mof in tqdm(mo_list, desc="📁 CREATE 시뮬레이션 입력 파일 생성 중"):
            try:
                _make_simulation_input(mof, base_tpl, gcmc_params, out_root, raspa_dir)
            except Exception as e:
                print(f"✖ {mof} 생성 실패: {e}")

    print("✅ CREATE 완료")


def parse_data_file(mof_dir: Path):
    """
    .data 파일에서 uptake 정보 추출
    """
    data_root = mof_dir / 'Output' / 'System_0'
    files = [f for f in os.listdir(data_root) if f.endswith('.data')]
    if not files:
        raise FileNotFoundError(".data 파일을 찾을 수 없습니다")
    text = (data_root / files[0]).read_text()
    dic = {}
    keys = [
        "Average loading absolute [mol/kg framework]",
        "Average loading excess [mol/kg framework]",
        "Average loading absolute [molecules/unit cell]",
        "Average loading excess [molecules/unit cell]"
    ]
    for key in keys:
        try:
            val = float(text.split(key)[1].split("+/-")[0].split()[-1])
            dic[key] = val
        except:
            pass
    if not dic:
        raise ValueError("데이터 파싱 실패")
    return dic

def run_one(mof_idx):
        # gcmcconfig.json 로드
    cwd = Path.cwd()
    gcmcconf_path = cwd / 'gcmcconfig.json'
    if not gcmcconf_path.exists():
        print("✖ gcmcconfig.json 파일이 없습니다.")
        sys.exit(1)
    with open(gcmcconf_path) as f:
        gcmc_params = json.load(f)
    print(gcmc_params)
    raspa_dir = Path(gcmc_params["RASPA_DIR"])
    mof, idx = mof_idx
    mof_dir = Path('low_pressure_gcmc') / mof
    cmd = f"{raspa_dir}/bin/simulate simulation.input"
    start = time.time()
    subprocess.run(cmd, shell=True, check=True, cwd=mof_dir)
    elapsed = time.time() - start
    uptake_dic = parse_data_file(mof_dir)
    print(uptake_dic)
    return mof, uptake_dic["Average loading absolute [mol/kg framework]"], elapsed


def cmd_run(ncpus):
    """
    DB에서 completed IS NULL인 MOF 실행 후 결과 DB 업데이트
    """
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()

    global FIRST_COL
    FIRST_COL = df.columns[0]

    pending = df[df['completed'].isna()][FIRST_COL].astype(str).tolist()
    total = len(pending)
    print(f"▶ RUN: {total}개 MOF, {ncpus}개 CPU로 실행 시작")
    if total == 0:
        print("✔ 처리할 MOF가 없습니다.")
        return

    random.shuffle(pending)
    # gcmc 파라미터
    with open('gcmcconfig.json') as f:
        gcmc_params = json.load(f)
    raspa_dir = Path(gcmc_params.get("RASPA_DIR", ""))

    # 병렬 실행
    with ProcessPoolExecutor(max_workers=ncpus) as exe:
        futures = {exe.submit(run_one, (mof, i+1)): mof for i, mof in enumerate(pending)}
        for fut in as_completed(futures):
            mof = futures[fut]
#        try:
            mof_name, uptake, ctime = fut.result()
            conn2 = sqlite3.connect(DB_PATH)
            conn2.execute(
                    f"UPDATE {TABLE} SET `uptake[mol/kg framework]` = ?, calculation_time = ?, completed = 1 WHERE {FIRST_COL} = ?",
                    (uptake, ctime, mof_name)
                )
            conn2.commit()
            conn2.close()
            print(f"✔ {mof_name}: uptake={uptake}, time={ctime:.1f}s")
 #           except Exception as e:
            print(f"✖ {mof} 실패: {e}")

    print("✅ RUN 완료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="low_pressure_gcmc.py")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_create = sub.add_parser('create', help='simulation.input 생성')
    p_create.add_argument('-n', '--ncpus', type=int, default=1,
                          help='simulation.input 생성 병렬 프로세스 수')

    p_run = sub.add_parser('run', help='GCMC 실행')
    p_run.add_argument('-n', '--ncpus', type=int, default=1,
                       help='GCMC 실행 병렬 프로세스 수')

    args = parser.parse_args()
    if args.cmd == 'create':
        cmd_create(args.ncpus)
    elif args.cmd == 'run':
        cmd_run(args.ncpus)
    else:
        parser.print_help()
