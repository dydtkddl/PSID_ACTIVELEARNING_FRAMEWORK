import os
import json
import numpy as np
import pandas as pd

def run_remote_job(node, node_dir, args_list,make_simulations_py_path , pwd):
    # 명령어 생성
    arg_string = ' '.join(str(arg) for arg in args_list)
    cmd = f"ssh {node} 'cd {pwd} && python {make_simulations_py_path} {arg_string}'"
    print(f"🚀 {node}에서 원격 실행 중...")
    os.system(cmd)

def main():
    with open("./config.json", "r") as f:
        config = json.load(f)

    # 설정 값 가져오기
    G = config["GCMC"]
    low_config = config["Low_Pressure_GCMC"]

    execute_nodes = low_config["Parallel"]["nodes"]
    max_cpu_fraction = low_config["Parallel"]["max_cpu_fraction"]
    execute_nodes_len = len(execute_nodes)

    # MOF 분할
    target_mofs = pd.read_csv("target_mofs.csv")
    chunks = np.array_split(target_mofs, execute_nodes_len)

    # 환경 변수에서 cif 경로 가져오기
    raspa_dir = os.environ.get("RASPA_DIR")
    if not raspa_dir:
        raise EnvironmentError("RASPA_DIR not set")
    cifs_path = os.path.join(raspa_dir, "share", "raspa", "structures", "cif")
    python_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    make_simulations_py_path = os.path.join(python_file_dir_path, "make_simulations.py")
    pwd = os.path.abspath("./")
    for chunk, node in zip(chunks, execute_nodes):
        print(f"🔧 노드 {node} 처리 시작 ({len(chunk)}개 MOF)")

        # 디렉토리 생성 및 저장
        node_dir = os.path.join("low_pressure_gcmc", node)
        os.makedirs(node_dir, exist_ok=True)
        chunk = chunk[["filename"]].copy()
        chunk["node"] = node
        chunk.to_csv(os.path.join(node_dir, "target_mofs.csv"), index=False)

        # make_simulations.py에 전달할 인자
        args_list = [
            G["NumberOfCycles"],
            G["NumberOfInitializationCycles"],
            G["PrintEvery"],
            str(G["UseChargesFromCIFFile"]).lower(),
            G["ExternalTemperature"],
            G["ExternalPressure"],
            G["Forcefield"],
            G["GAS"],
            G["MoleculeDefinition"],
            G["CUTOFFVDW"],
            G["CUTOFFCHARGECHARGE"],
            G["CUTOFFCHARGEBONDDIPOLE"],
            G["CUTOFFBONDDIPOLEBONDDIPOLE"],
            cifs_path,
            max_cpu_fraction
        ]

        run_remote_job(node, node_dir, args_list,make_simulations_py_path, pwd)

if __name__ == "__main__":
    main()
