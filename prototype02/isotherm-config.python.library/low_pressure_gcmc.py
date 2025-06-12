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
CSV_NAME = './low_pressure_gcmc.csv'
FIRST_COL = None  # CSV 로드 후 설정

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


def parse_data_file(data_path: Path):
    """
    Output/.data 파일에서 uptake, calculation time 추출
    """
    uptake = None
    ctime = None
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    for line in data_path.read_text().splitlines():
        ll = line.lower()
        if 'uptake' in ll:
            uptake = float(ll.split()[-2])
        if 'calculation time' in ll:
            ctime = float(ll.split()[-2])
    if uptake is None or ctime is None:
        raise ValueError(f"데이터 파싱 실패: {data_path}")
    return uptake, ctime


def run_one(mof: str):
    """
    한 MOF에 대해 GCMC 실행 및 결과 파싱
    """
    cwd = Path.cwd()
    mof_dir = cwd / 'low_pressure_gcmc' / mof

    cmds = [
        ["python", "make_simulations.py", mof],   # 실제 스크립트로 교체
        ["python", "run_all.py"],                 
        ["python", "crop_simulations.py"]         
    ]
    for c in cmds:
        subprocess.run(c, cwd=mof_dir, check=True)

    data_file = mof_dir / 'Output' / 'results.data'  # 실제 파일 경로로 조정
    uptake, ctime = parse_data_file(data_file)
    return mof, uptake, ctime


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

    df = pd.read_csv(csv_path)
    global FIRST_COL
    FIRST_COL = df.columns[0]

    pending = df.loc[df['completed'] != True, FIRST_COL].astype(str).tolist()
    if not pending:
        print("✔ 처리할 MOF가 없습니다.")
        return

    random.shuffle(pending)
    print(f"▶ RUN: {len(pending)}개 MOF, {ncpus}개 CPU로 실행 시작")

    with ProcessPoolExecutor(max_workers=ncpus) as exe:
        futures = {exe.submit(run_one, mof): mof for mof in pending}
        for fut in as_completed(futures):
            mof = futures[fut]
            try:
                _, uptake, ctime = fut.result()
                df.loc[df[FIRST_COL] == mof, 'uptake'] = uptake
                df.loc[df[FIRST_COL] == mof, 'calculation_time'] = ctime
                df.loc[df[FIRST_COL] == mof, 'completed'] = True
                df.to_csv(csv_path, index=False)
                print(f"✔ {mof}: uptake={uptake}, time={ctime}")
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

