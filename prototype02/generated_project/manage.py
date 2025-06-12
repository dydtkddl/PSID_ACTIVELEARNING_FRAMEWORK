# python manage.py create
# > 298
# > 1
# > MOF DB csv 이름 선택 ( ./mof_database내부에 있는 csv의 리스트를 보여주고 cli로 선택할 수 있게)
# > 가스선택 (RASPA_DIR share molecule에 있는 것들 이름, RASPA_DIR정보는 ./config.json의 RASPA_DIR 에 있음 )
# > kinetic-diameter 가스 선택 (kinetic-diameter.json 안에 있는 것들 이름{    
#     "kinetic_diameters": [
#         {"name": "Hydrogen", "formula": "H2", "molecular_mass": 2, "kinetic_diameter_A": 2.89},
#         {"name": "Helium", "formula": "He", "molecular_mass": 4, "kinetic_diameter_A": 2.60},
#         {"name": "Methane", "formula": "CH4", "molecular_mass": 16, "kinetic_diameter_A": 3.80},
#         {"name": "Ammonia", "formula": "NH3", "molecular_mass": 17, "kinetic_diameter_A": 2.60},
#         {"name": "Water", "formula": "H2O", "molecular_mass": 18, "kinetic_diameter_A": 2.65},)... 이런식임
# # 작동 세부사항
# 0) 선택한 MOFDB CSV를 pd로 읽어옴
# if PLD 열이 있으면
# 1) 선택한 MOF DB csv의 MOF들을 PLD기준으로 kinetic diameter와 비교하여 스크리닝
# else:
# 1) PLD정보가 없는데 괜찮냐고 물어보고 스크리닝 안된다는 경고한다음에 y/n선택하게 cli 이후 y면 pld스크리닝 없이 CSV그대로 사용
# > 
# 2) 현재 디렉토리에 MOF DB csv 이름의 폴더 작성
# 3) MOF DB csv 이름 폴더 내부에 가스_온도_압력_screened_by_kinetic_diameter가스이름 폴더 생성
# 4) 해당 폴더에 screened 된 MOF DB csv를 low_pressure_gcmc.csv / active_learning_gcmc.csv로 만듦
# >> 이때 low_pressure_gcmc.csv에는 completed/uptake/calculation_time 열이 존재해야함 (추가될때 null로)
# >> active_learning_gcmc.csv에는 iteration/uptake/calculation_time 열이 존재해야함 (추가될떄 null로)
# 5) 해당 폴더에 isotherm-config.bat파일이 있는 폴더에서 low_pressure_gcmc.py와 active_learning_gcmc.py를 가져옴 (isotherm-config.bat은 config.json의 isotherm-config.bat 항목에 있음음)
# 6) 해당 폴더에 isotherm-config.bat파일이 있는 폴더에서 gcmcconfig.json과 base.input을 가져옴

#!/usr/bin/env python3
# manage.py

import sys
import os
import json
import shutil
from pathlib import Path

import pandas as pd

CONFIG_FILE = Path(__file__).parent / 'config.json'
MOF_DB_DIR   = Path(__file__).parent / 'mof_database'
KD_FILE      = Path(__file__).parent / 'kinetic-diameter.json'


def load_config():
    if not CONFIG_FILE.exists():
        print(f"ERROR: config.json not found at {CONFIG_FILE}")
        sys.exit(1)
    return json.loads(CONFIG_FILE.read_text())


def choose_from_list(prompt, items):
    """번호로 선택하게 한 뒤, 선택된 아이템 반환"""
    for i, name in enumerate(items, 1):
        print(f"  {i}. {name}")
    idx = input(f"{prompt} [1–{len(items)}]: ").strip()
    try:
        n = int(idx)
        assert 1 <= n <= len(items)
    except:
        print("❌ 잘못된 입력입니다. 프로그램을 종료합니다.")
        sys.exit(1)
    return items[n-1]


def prompt_yes_no(msg):
    ans = input(f"{msg} (y/n): ").strip().lower()
    return ans == 'y'


def main():
    if len(sys.argv) != 2 or sys.argv[1] != 'create':
        print("사용법: python manage.py create")
        sys.exit(0)

    cfg = load_config()

    # 1) MOF DB CSV 선택
    csv_files = sorted([f.name for f in MOF_DB_DIR.glob('*.csv')])
    if not csv_files:
        print("❌ ./mof_database 에 CSV 파일이 하나도 없습니다.")
        sys.exit(1)
    csv_name = choose_from_list("MOF DB CSV를 선택하세요", csv_files)
    csv_path = MOF_DB_DIR / csv_name
    df = pd.read_csv(csv_path)

    # 2) 시뮬레이션용 가스 선택
    raspa_dir = Path(cfg['RASPA_DIR'])
    mol_dir   = raspa_dir / 'share' / 'raspa' / 'molecules' / 'ExampleDefinitions'
    gas_list  = sorted([p.stem for p in mol_dir.iterdir() if p.is_file()])
    sim_gas   = choose_from_list("시뮬레이션할 가스를 선택하세요", gas_list)

    # 3) kinetic-diameter용 가스 선택
    kd_data = json.loads(KD_FILE.read_text())
    kd_list = [item['name'] for item in kd_data['kinetic_diameters']]
    kd_name = choose_from_list("kinetic-diameter 가스를 선택하세요", kd_list)
    kd_entry = next(item for item in kd_data['kinetic_diameters'] if item['name']==kd_name)
    kd_value = kd_entry['kinetic_diameter_A']

    # 3.5) 온도·압력 입력
    T = input("온도를 입력하세요 (예: 298.15) [K]: ").strip()
    P = input("압력을 입력하세요 (예: 1) [bar]: ").strip()

    # 4) PLD 컬럼 유무 및 screening
    if 'PLD' in df.columns:
        screened = df[df['PLD'] >= kd_value].copy()
        print(f"✅ PLD ≥ {kd_value} Å 인 MOF만 남깁니다 ({len(screened)}/{len(df)}).")
    else:
        print("⚠️ PLD 정보가 없습니다. Screening 없이 진행하면, 모든 MOF가 포함됩니다.")
        if prompt_yes_no("계속하시겠습니까?"):
            screened = df.copy()
        else:
            print("사용자가 취소하였습니다.")
            sys.exit(0)

    # 5) 폴더 구조 생성
    base_name = csv_path.stem
    out_root  = Path.cwd() / base_name
    tag       = f"{sim_gas}_{T}K_{P}bar_screened_by_{kd_name}"
    out_dir   = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CIF 파일 경로 오타 수정 ('stuctures' → 'structures')
    cifpath = Path(cfg['RASPA_DIR']) / "share" / "raspa" / "structures" / "cif"

    # 2. CIF 파일 리스트 생성 로직 개선
    cifs = [
        Path(x).stem  # 확장자 제거 (예: 'MOF1.cif' → 'MOF1')
        for x in os.listdir(cifpath) 
        if x.endswith(".cif")  # .cif로 끝나는 파일만 필터링
    ]
    if sum(screened.iloc[:, 0].isin(cifs)) != len(screened) : 
        notin = len(screened) - sum(screened.iloc[:, 0].isin(cifs))
        print(f">> error : 스크리닝된 {len(screened)}개의 MOF들중 {notin}개의 cif파일이 {cifpath}에 존재하지 않습니다. \n>> solution : csv내에 있는 모든 MOF들의 cif구조 파일을 {cifpath}에 넣으십시오")
        sys.exit(1)
    # 3. low_pressure_gcmc.csv 생성 (벡터화 연산 사용)
    lp = screened.copy()
    lp.insert(1, 'cif_file_exist', 
            lp.iloc[:, 0].isin(cifs))  # ✅ 첫 번째 열 기준 존재 여부
    lp.insert(2, 'completed', pd.NA)
    lp.insert(3, 'uptake[mol/kg framework]', pd.NA)
    lp.insert(4, 'calculation_time', pd.NA)
    lp.to_csv(out_dir/'low_pressure_gcmc.csv', index=False)

    # 4. active_learning_gcmc.csv 생성 (동일 로직 적용)
    al = screened.copy()
    al.insert(1, 'cif_file_exist', 
            al.iloc[:, 0].isin(cifs))  # ✅ 첫 번째 열 기준 존재 여부
    al.insert(2, 'iteration', pd.NA)
    al.insert(3, 'uptake[mol/kg framework]', pd.NA)
    al.insert(4, 'calculation_time', pd.NA)
    al.to_csv(out_dir/'active_learning_gcmc.csv', index=False)

    # 8) 스크립트·설정 파일 복사
    bat_path = Path(cfg['isotherm-config.bat'])
    bat_dir  = bat_path

    for fname in [
        'low_pressure_gcmc.py',
        'active_learning_gcmc.py',
        'gcmcconfig.json',
        'base.input',
        "active_learning_config.json",
        "prediction_gcmc.py",
        "pyrascont.py"
    ]:
        src = bat_dir / fname
        if not src.exists():
            print(f"⚠️ 복사할 파일이 없습니다: {src}")
            continue
        
        if fname == 'gcmcconfig.json':
            # JSON 로드 → 필드 업데이트 → 저장
            config = json.loads(src.read_text(encoding='utf-8'))
            config.update({
                "GAS": sim_gas,
                "ExternalPressure": float(P),
                "ExternalTemperature": float(T),
                "RASPA_DIR": str(Path(cfg['RASPA_DIR']).resolve())  # 절대 경로로 변환
            })
            (out_dir / fname).write_text(
                json.dumps(config, indent=4, ensure_ascii=False),
                encoding='utf-8'
            )
        else:
            shutil.copy(src, out_dir / fname)

    print(f"\n✅ 완료! 생성된 폴더: {out_dir}")

if __name__ == '__main__':
    main()
