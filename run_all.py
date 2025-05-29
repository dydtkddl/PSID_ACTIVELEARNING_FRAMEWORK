import os
import subprocess
import argparse
from joblib import Parallel, delayed, parallel_backend
from threading import Lock
import time
import datetime

write_lock = Lock()

def append_to_file(filename, text):
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

def load_completed_list(completed_file):
    if not os.path.exists(completed_file):
        return set()
    with open(completed_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def run_simulation_for_dir(sim_dir, raspa_dir, idx, total, start_time,
                           completed_file, progress_file):
    try:
        command = f"{raspa_dir}/bin/simulate simulation.input"
        start_t = time.time()
        subprocess.run(command, shell=True, check=True, cwd=sim_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_t = time.time()

        append_to_file(completed_file, sim_dir)

        with write_lock:
            completed_count = 0
            if os.path.exists(completed_file):
                with open(completed_file, 'r', encoding='utf-8') as cf:
                    completed_count = sum(1 for _ in cf)

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

    except subprocess.CalledProcessError as e:
        append_to_file(progress_file, f"[{idx+1}/{total}] {sim_dir} FAILED: {str(e)}")
    except Exception as e:
        append_to_file(progress_file, f"[{idx+1}/{total}] {sim_dir} Unexpected Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run RASPA simulations in parallel")
    parser.add_argument("max_cpu_fraction", type=float, help="Max CPU fraction to use")
    args = parser.parse_args()

    raspa_dir = os.getenv("RASPA_DIR")
    if not raspa_dir:
        raise EnvironmentError("RASPA_DIR not set")

    completed_file = "98_complete.txt"
    progress_file = "99_progress.log"

    all_sim_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d)])
    done_dirs = load_completed_list(completed_file)
    remaining_dirs = [d for d in all_sim_dirs if d not in done_dirs]
    total = len(all_sim_dirs)
    to_run = len(remaining_dirs)

    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = (f"\n---- {'RESTART' if os.path.exists(progress_file) else 'START'} at {time_str}, "
              f"Remain={to_run}/{total} ----")
    append_to_file(progress_file, header)

    if to_run == 0:
        return

    start_time = time.time()
    n_jobs = max(1, int(os.cpu_count() * args.max_cpu_fraction))

    with parallel_backend("threading", n_jobs=n_jobs):
        Parallel()(
            delayed(run_simulation_for_dir)(
                sim_dir=sim_dir,
                raspa_dir=raspa_dir,
                idx=idx,
                total=total,
                start_time=start_time,
                completed_file=completed_file,
                progress_file=progress_file
            ) for idx, sim_dir in enumerate(remaining_dirs)
        )

    finish_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    append_to_file(progress_file, f"---- ALL DONE at {finish_time_str} ----")

if __name__ == "__main__":
    main()

