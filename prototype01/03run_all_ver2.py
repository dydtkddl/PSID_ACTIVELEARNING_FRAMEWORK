import os
import subprocess
import argparse
from joblib import Parallel, delayed, parallel_backend
from threading import Lock
import time
import datetime

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

def run_simulation_for_dir(sim_dir, raspa_dir, idx, total, start_time, 
                           completed_file, progress_file, verbose):
    """
    개별 디렉터리에 대해 시뮬레이션을 실행.
    - simulate 명령어를 shell=True로 호출, cwd=sim_dir 설정
    - 성공 시 completed_file, progress_file에 기록
    - ETA 계산 위해 현재까지 완료된 수를 세어 추정
    """
    try:
        if verbose:
            print(f"[{idx+1}/{total}] Starting simulation in {sim_dir}")

        command = f"{raspa_dir}/bin/simulate simulation.input"
        start_t = time.time()
        subprocess.run(command, shell=True, check=True, cwd=sim_dir)
        end_t = time.time()

        # 완료 기록
        append_to_file(completed_file, sim_dir)

        # 현재까지 완료된 디렉터리 수
        with write_lock:
            if os.path.exists(completed_file):
                with open(completed_file, 'r', encoding='utf-8') as cf:
                    completed_count = sum(1 for _ in cf)
            else:
                completed_count = 0

        elapsed_for_this = end_t - start_t
        elapsed_total = end_t - start_time
        avg_time_each = elapsed_total / completed_count if completed_count > 0 else 0
        remain_count = total - completed_count
        est_remain_time = avg_time_each * remain_count
        eta = datetime.datetime.now() + datetime.timedelta(seconds=est_remain_time)

        log_text = (f"[{idx+1}/{total}] {sim_dir} Done. "
                    f"TimeForThis={elapsed_for_this:.1f}s, "
                    f"Completed={completed_count}/{total}, "
                    f"ETA={eta.strftime('%Y-%m-%d %H:%M:%S')}")
        append_to_file(progress_file, log_text)

        if verbose:
            print(f"[{idx+1}/{total}] Simulation completed in {sim_dir}. Elapsed: {elapsed_for_this:.1f}s")

    except subprocess.CalledProcessError as e:
        error_msg = f"[{idx+1}/{total}] {sim_dir} FAILED: {str(e)}"
        append_to_file(progress_file, error_msg)
        if verbose:
            print(error_msg)
    except Exception as e:
        error_msg = f"[{idx+1}/{total}] {sim_dir} Unexpected Error: {str(e)}"
        append_to_file(progress_file, error_msg)
        if verbose:
            print(error_msg)

def main():
    parser = argparse.ArgumentParser(description="Run RASPA simulations in parallel across subdirectories with restart capability.")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs for parallel processing.")
    parser.add_argument("--verbose", action="store_true", default=False, help="If set, prints more detailed logs.")
    args = parser.parse_args()

    # 환경변수 RASPA_DIR 체크
    raspa_dir = os.getenv("RASPA_DIR")
    if not raspa_dir:
        raise EnvironmentError("RASPA_DIR 환경 변수가 설정되지 않았습니다.")

    # 기록용 파일들
    completed_file = "98_complete.txt"
    progress_file  = "99_progress.log"

    # 현재 디렉토리에 있는 하위 디렉터리 목록
    all_sim_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

    # 이미 완료된 디렉터리 목록 불러오기
    done_dirs = load_completed_list(completed_file)

    # 이번에 실제로 실행할 디렉터리
    remaining_dirs = [d for d in all_sim_dirs if d not in done_dirs]
    total = len(all_sim_dirs)
    to_run = len(remaining_dirs)

    # 로그에 재시작 여부 남기기
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(progress_file):
        append_to_file(progress_file, f"\n---- RESTART at {time_str}, Remain={to_run}/{total} ----")
    else:
        append_to_file(progress_file, f"---- START at {time_str}, Target={to_run}/{total} ----")

    print(f"전체 디렉터리: {total}개")
    print(f"이미 완료된 디렉터리: {len(done_dirs)}개")
    print(f"이번에 실행할 디렉터리: {to_run}개")

    if to_run == 0:
        print("모든 디렉터리가 이미 완료되었습니다. 종료합니다.")
        return

    start_time = time.time()

    # 병렬 실행 (threading backend 사용 -> 같은 프로세스 내에서 Lock 공유)
    with parallel_backend("threading", n_jobs=args.num_cpus):
        Parallel()(
            delayed(run_simulation_for_dir)(
                sim_dir=sim_dir,
                raspa_dir=raspa_dir,
                idx=idx,
                total=total,
                start_time=start_time,
                completed_file=completed_file,
                progress_file=progress_file,
                verbose=args.verbose
            )
            for idx, sim_dir in enumerate(remaining_dirs)
        )

    # 전부 완료 후 로그
    finish_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    append_to_file(progress_file, f"---- ALL DONE at {finish_time_str} ----")
    print("모든 시뮬레이션이 정상 종료되었습니다.")

if __name__ == "__main__":
    main()
