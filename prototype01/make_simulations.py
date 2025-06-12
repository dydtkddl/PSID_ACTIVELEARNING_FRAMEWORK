import os
import argparse
import pandas as pd
import pyrascont
import logging
from datetime import datetime

def setup_logger():
    log_filename = "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
#            logging.StreamHandler()
        ]
    )

def main():
    setup_logger()
    logging.info("✅ GCMC input 파일 생성 시작")

    parser = argparse.ArgumentParser(description="GCMC Input 생성 인자")
    parser.add_argument("NumberOfCycles")
    parser.add_argument("NumberOfInitializationCycles")
    parser.add_argument("PrintEvery")
    parser.add_argument("UseChargesFromCIFFile")
    parser.add_argument("ExternalTemperature")
    parser.add_argument("ExternalPressure")
    parser.add_argument("Forcefield")
    parser.add_argument("GAS")
    parser.add_argument("MoleculeDefinition")
    parser.add_argument("CUTOFFVDW")
    parser.add_argument("CUTOFFCHARGECHARGE")
    parser.add_argument("CUTOFFCHARGEBONDDIPOLE")
    parser.add_argument("CUTOFFBONDDIPOLEBONDDIPOLE")
    parser.add_argument("cifs_path")
    parser.add_argument("max_cpu_fraction")
    args = parser.parse_args()
    max_cpu_fraction = args.max_cpu_fraction
    # 인자 확인 출력
    for k, v in vars(args).items():
        logging.info(f"Arg - {k}: {v}")

    # base.input 템플릿 로딩
    base_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.input")
    with open(base_input_path, "r") as f:
        template = f.read()
        logging.info(f"📄 base.input 템플릿 로드 완료: {base_input_path}")

    # MOF 리스트 읽기
    chunk = pd.read_csv("target_mofs.csv")
    logging.info(f"📊 MOF 수: {len(chunk)}개")

    for i, row in chunk.iterrows():
        mof = row["filename"]
        mof_dir = os.path.join("simulations", mof)
        os.makedirs(mof_dir, exist_ok=True)

        logging.info(f"[{i+1}/{len(chunk)}] ▶ {mof} 처리 시작")

        # UNITCELL 계산
        cif_path = os.path.join(args.cifs_path, mof)
        try:
            res_ucell = pyrascont.cif2Ucell(cif_path, float(args.CUTOFFVDW), Display=False)
            unitcell_str = ' '.join(map(str, res_ucell))
            logging.info(f"  ✅ UNITCELL 계산 성공: {unitcell_str}")
        except Exception as e:
            logging.warning(f"  ⚠️ {mof} 실패 - {e}")
            continue

        # input 파일 생성
        filled_input = template.format(
            NumberOfCycles=args.NumberOfCycles,
            NumberOfInitializationCycles=args.NumberOfInitializationCycles,
            PrintEvery=args.PrintEvery,
            UseChargesFromCIFFile=args.UseChargesFromCIFFile,
            Forcefield=args.Forcefield,
            TEMP=args.ExternalTemperature,
            PRESSURE=float(args.ExternalPressure) * 100000,
            GAS=args.GAS,
            MoleculeDefinition=args.MoleculeDefinition,
            MOF=mof,
            UNITCELL=unitcell_str
        )

        input_path = os.path.join(mof_dir, "simulation.input")
        with open(input_path, "w") as f_out:
            f_out.write(filled_input)
            logging.info(f"  📝 simulation.input 생성 완료: {input_path}")

    logging.info("✅ 전체 작업 완료!")

if __name__ == "__main__":
    main()
