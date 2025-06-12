#!/usr/bin/env python3
# low_pressure_gcmc.py

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyrascont
import pandas as pd
from pathlib import Path
from tqdm import tqdm  
from multiprocessing import Manager
import time 
CSV_NAME = './low_pressure_gcmc.csv'
FIRST_COL = None  # CSV 로드 후 설정

## create 관련
def _make_simulation_input(mof, base_tpl, gcmc_params, out_root,raspa_dir):
    """
    단일 MOF 폴더에 simulation.input 생성 함수
    """
    cifpath = os.path.join( raspa_dir / "share" / "raspa" / "structures" / "cif" / mof)
    try:
            res_ucell = pyrascont.cif2Ucell(cifpath, float(gcmc_params["CUTOFFVDW"]), Display=False)
            unitcell_str = ' '.join(map(str, res_ucell))
            # print(f"  ✅ UNITCELL 계산 성공: {unitcell_str}")
    except Exception as e:
            df = pd.read_csv(CSV_NAME)            
            fir = df.columns[0]
            df.loc[df[fir] == mof, "sim_created"] = False
            df.to_csv(CSV_NAME, index = False)
    # None인 값은 빈 문자열로 대체
    try : 
        sim_content = base_tpl.format(
                NumberOfCycles=gcmc_params["NumberOfCycles"],
                NumberOfInitializationCycles=gcmc_params["NumberOfInitializationCycles"],
                PrintEvery=gcmc_params["PrintEvery"],
                UseChargesFromCIFFile=gcmc_params["UseChargesFromCIFFile"],
                Forcefield=gcmc_params["Forcefield"],
                TEMP=gcmc_params["ExternalTemperature"],
                PRESSURE=float(gcmc_params["ExternalPressure"]) * 100000,
                GAS=gcmc_params["GAS"],
                MoleculeDefinition=gcmc_params["MoleculeDefinition"],
                MOF=mof,
                UNITCELL=unitcell_str
        )
        mof_dir = out_root / mof
        mof_dir.mkdir(exist_ok=True)
        sim_path = mof_dir / 'simulation.input'
        with open(sim_path, 'w') as fw:
            fw.write(sim_content)
    except Exception as e:
            df = pd.read_csv(CSV_NAME)            
            fir = df.columns[0]
            df.loc[df[fir] == mof, "sim_created"] = False
            df.to_csv(CSV_NAME, index = False)
    return mof
def cmd_create(ncpus):
    """
    1) low_pressure_gcmc/ 하위 폴더 생성
    2) CSV 첫 번째 열의 MOF 이름별 서브폴더 생성
    3) 각 폴더에 base.input 템플릿 + gcmcconfig.json 치환 → simulation.input
       병렬 처리 (max_workers=ncpus)
    """
    cwd = Path.cwd()
    csv_path = cwd / CSV_NAME
    if not csv_path.exists():
        print(f"✖ CSV 파일이 없습니다: {CSV_NAME}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    global FIRST_COL
    FIRST_COL = df.columns[0]

    # gcmcconfig.json 로드
    gcmcconf_path = cwd / 'gcmcconfig.json'
    if not gcmcconf_path.exists():
        print("✖ gcmcconfig.json 파일이 없습니다.")
        sys.exit(1)
    with open(gcmcconf_path) as f:
        gcmc_params = json.load(f)
    print(gcmc_params)
    raspa_dir = Path(gcmc_params["RASPA_DIR"])
    # base.input 템플릿 로드
    base_tpl_path = cwd / 'base.input'
    if not base_tpl_path.exists():
        print("✖ base.input 파일이 없습니다.")
        sys.exit(1)
    base_tpl = base_tpl_path.read_text()
    print(base_tpl)
    # 출력 루트 폴더 생성
    out_root = cwd / 'low_pressure_gcmc'
    out_root.mkdir(exist_ok=True)

    mo_list = df[FIRST_COL].astype(str).tolist()
    print(f"▶ CREATE: {len(mo_list)}개 MOF, {ncpus}개 프로세스로 simulation.input 생성 시작")
    # import time 
    if ncpus > 1:
        with ProcessPoolExecutor(max_workers=ncpus) as exe:
            futures = {exe.submit(_make_simulation_input,
                                mof, base_tpl, gcmc_params, out_root,raspa_dir): mof for mof in mo_list}
            
            # tqdm으로 진행률 표시 (변경 부분)
            for fut in tqdm(as_completed(futures), total=len(mo_list), desc="📁 MOF 시뮬레이션 입력 파일 생성 중"):
                mof = futures[fut]
                try:
                    fut.result()
                    # print(f"✔ {mof} simulation.input 생성 완료")
                except Exception as e:
                    print(f"✖ {mof} 생성 실패: {e}")
    else:
        # 단일 스레드 처리에도 tqdm 적용 (변경 부분)
        for mof in tqdm(mo_list, desc="📁 MOF 시뮬레이션 입력 파일 생성 중"):
            try:
                _make_simulation_input(mof, base_tpl, gcmc_params, out_root,raspa_dir)
                # print(f"✔ {mof} simulation.input 생성 완료")
            except Exception as e:
                print(f"✖ {mof} 생성 실패: {e}")

    print("✅ CREATE 완료")


def parse_data_file(gcmc_sim_root: Path):
    """
    Output/.data 파일에서 uptake, calculation time 추출
    """
    uptake = None
    gcmc_data_root = gcmc_sim_root / "Output" / "System_0"
    gcmc_datas = [ x for x in os.listdir(gcmc_data_root) if ".data" in x]
    if len(gcmc_datas) == 0 :
        raise FileNotFoundError(f"{gcmc_datas} .data file not found")
    if len(gcmc_datas) > 1 : 
        print(">> warining : there are two data file exist")
    datafile = gcmc_data_root / gcmc_datas[0]
    with open(datafile, "r") as f:
        data = f.read()
        dic = {}
        uptake_absolute = float(data.split("Average loading absolute [mol/kg framework]")[1].split(" +/-")[0].split()[0])
        dic["Average loading absolute [mol/kg framework]"] = uptake_absolute
        uptake_excess = float(data.split("Average loading excess [mol/kg framework]")[1].split(" +/-")[0].split()[0])
        dic["Average loading excess [mol/kg framework]"] = uptake_excess
        uptake_absolute_per_unitcell = float(data.split("Average loading absolute [molecules/unit cell]")[1].split(" +/-")[0].split()[0])
        dic["Average loading absolute [molecules/unit cell]"] = uptake_absolute_per_unitcell
        uptake_excess_per_unitcell = float(data.split("Average loading excess [molecules/unit cell]")[1].split(" +/-")[0].split()[0])
        dic["Average loading excess [molecules/unit cell]"] = uptake_excess_per_unitcell
    if dic == {}:
        raise ValueError(f"데이터 파싱 실패: {datafile}")
    return dic

from threading import Lock
write_lock = Lock()

def append_to_file(filename, text):
    """파일에 text 한 줄을 안전하게(락 사용) 추가합니다."""
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

def load_completed_list(completed_file):
    """98_complete.txt에서 이미 완료된 디렉터리 목록(줄 단위)을 세트로 읽어옵니다."""
    if not os.path.exists(completed_file):
        return set()
    with open(completed_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}
def run_one(mof: str, idx : int, total : int):
    """
    한 MOF에 대해 GCMC 실행 및 결과 파싱
    """
    cwd = Path.cwd()
    mof_dir = cwd / 'low_pressure_gcmc' / mof

    gcmcconf_path = cwd / 'gcmcconfig.json'
    if not gcmcconf_path.exists():
        print("✖ gcmcconfig.json 파일이 없습니다.")
        sys.exit(1)
    with open(gcmcconf_path) as f:
        gcmc_params = json.load(f)
    raspa_dir = Path(gcmc_params["RASPA_DIR"])

    try:
        print(f"[{idx+1}/{total}] Starting simulation in {mof_dir}")

        command = f"{raspa_dir}/bin/simulate simulation.input"
        start_t = time.time()
        subprocess.run(command, shell=True, check=True, cwd=mof_dir)
        end_t = time.time()

        ctime = end_t - start_t
        # elapsed_total = end_t - start_time
        # avg_time_each = elapsed_total / completed_count if completed_count > 0 else 0
        # remain_count = total - completed_count
        # est_remain_time = avg_time_each * remain_count
        # eta = datetime.datetime.now() + datetime.timedelta(seconds=est_remain_time)

        # log_text = (f"[{idx+1}/{total}] {sim_dir} Done. "
        #             f"TimeForThis={elapsed_for_this:.1f}s, "
        #             f"Completed={completed_count}/{total}, "
        #             f"ETA={eta.strftime('%Y-%m-%d %H:%M:%S')}")
        # append_to_file(progress_file, log_text)

        print(f"[{idx+1}/{total}] Simulation completed in {mof_dir}.") # Elapsed: {elapsed_for_this:.1f}s")

    except subprocess.CalledProcessError as e:
        error_msg = f"[{idx+1}/{total}] {mof_dir} FAILED: {str(e)}"
        # append_to_file(progress_file, error_msg)
        print(error_msg)
    except Exception as e:
        error_msg = f"[{idx+1}/{total}] {mof_dir} Unexpected Error: {str(e)}"
        # append_to_file(progress_file, error_msg)
        print(error_msg)

    uptake_dic = parse_data_file(mof_dir)
    return mof, uptake_dic, ctime


def cmd_run(ncpus: int):
    """
    1) CSV에서 completed != True MOF 목록
    2) shuffle → 병렬 실행 → 결과마다 CSV 업데이트
    """
    cwd = Path.cwd()
    csv_path = cwd / CSV_NAME
    if not csv_path.exists():
        print(f"✖ CSV 파일이 없습니다: {CSV_NAME}")
        sys.exit(1)

#    df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, dtype={'completed': 'Int64'})
    global FIRST_COL
    FIRST_COL = df.columns[0]
    pending = df[df["completed"].isna()][FIRST_COL].astype(str).tolist()
    print(f"{len(pending)} 개 처리줌 ( {len(df)} completed ) ")
    if not pending:
        print("✔ 처리할 MOF가 없습니다.")
        return

    random.shuffle(pending)
    print(f"▶ RUN: {len(pending)}개 MOF, {ncpus}개 CPU로 실행 시작")
        
    manager = Manager()
    lock = manager.Lock()  # 🔒 공유 Lock 생성

    def _safe_update_csv(mof, uptake, ctime):
        """Lock으로 보호된 CSV 업데이트 함수"""
        with lock:
#            df = pd.read_csv(csv_path)
            df = pd.read_csv(csv_path, dtype={'completed': 'Int64'})
            df.loc[df[FIRST_COL] == mof, 'uptake[mol/kg framework]'] = uptake
            df.loc[df[FIRST_COL] == mof, 'calculation_time'] = ctime
            df.loc[df[FIRST_COL] == mof, 'completed'] = 1
            df.to_csv(csv_path, index=False)
            print(csv_path)
            print(df.loc[df[FIRST_COL] == mof, 'completed'] )
            print(df.head())
            print(df.loc[df[FIRST_COL] == mof])
            print(mof)
    with ProcessPoolExecutor(max_workers=ncpus) as exe:
       # futures = {exe.submit(run_one, mof): mof for mof in pending}
        total = len(pending)
        futures = { exe.submit(run_one, mof, idx, total): (mof, idx ) for idx, mof in enumerate(pending, 1)  }
        for fut in as_completed(futures):
            mof = futures[fut]
            try:
                _, uptake_dic, ctime = fut.result()
                _safe_update_csv(mof[0], uptake_dic["Average loading absolute [mol/kg framework]"], ctime)  # 🔒 Lock으로 보호
                print(f"✔ {mof}: Average loading absolute [mol/kg framework]={uptake_dic["Average loading absolute [mol/kg framework]"]}, time={ctime}")
            except Exception as e:
                print(f"✖ {mof} 실패: {e}")
    print("✅ RUN 완료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="low_pressure_gcmc.py")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_create = sub.add_parser('create', help='simulation.input 생성')
    p_create.add_argument('--ncpus', '-n', type=int, default=1,
                          help='simulation.input 생성 병렬 프로세스 수')

    p_run = sub.add_parser('run', help='GCMC 병렬 실행')
    p_run.add_argument('--ncpus', '-n', type=int, default=1,
                       help='GCMC 실행 병렬 프로세스 수')

    args = parser.parse_args()
    if args.cmd == 'create':
        cmd_create(args.ncpus)
    elif args.cmd == 'run':
        cmd_run(args.ncpus)
    else:
        parser.print_help()


