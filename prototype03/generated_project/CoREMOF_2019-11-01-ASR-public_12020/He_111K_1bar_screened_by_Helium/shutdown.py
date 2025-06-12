#!/usr/bin/env python3
"""
Safe shutdown utility for GCMC workflows.
Terminates all running workflow runner processes (low_pressure_gcmc or active_learning_gcmc)
and any lingering RASPA simulate processes, commits pending database transactions, and exits cleanly.

Usage:
    python safe_shutdown.py --workflow [low_pressure_gcmc|active_learning_gcmc] [--db-path path/to/mof_project.db]
"""
import sys
import time
import signal
import sqlite3
from pathlib import Path
import argparse

try:
    import psutil
except ImportError:
    print("psutil is required. Install via `pip install psutil`.")
    sys.exit(1)


def find_workflow_processes(name_snippet: str) -> list:
    matches = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info.get('cmdline') or [])
            if name_snippet in cmd:
                matches.append(proc)
        except Exception:
            continue
    if not matches:
        raise LookupError(f"No workflow process found matching '{name_snippet}'")
    return matches


def safe_terminate(process: psutil.Process, timeout: float = 5.0):
    children = process.children(recursive=True)
    to_kill = children + [process]
    for p in to_kill:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    gone, alive = psutil.wait_procs(to_kill, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    psutil.wait_procs(alive, timeout=timeout)


def kill_simulation_processes(cmd_snippet: str = 'simulate simulation.input', timeout: float = 5.0):
    to_terminate = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info.get('cmdline') or [])
            if cmd_snippet in cmd:
                to_terminate.append(proc)
        except Exception:
            continue
    if not to_terminate:
        return
    for p in to_terminate:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    gone, alive = psutil.wait_procs(to_terminate, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    psutil.wait_procs(alive, timeout=timeout)


def commit_db(db_path: Path):
    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        return
    conn = sqlite3.connect(str(db_path))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Safely shutdown GCMC workflow')
    parser.add_argument('--workflow', choices=['low_pressure_gcmc', 'active_learning_gcmc'], required=True,
                        help='Workflow script name to match and shut down')
    parser.add_argument('--db-path', type=Path, default=Path.cwd() / 'mof_project.db',
                        help='Path to SQLite database file')
    args = parser.parse_args()

    try:
        runners = find_workflow_processes(args.workflow)
        for runner in runners:
            print(f"Terminating workflow runner PID {runner.pid} ({args.workflow})")
            safe_terminate(runner)
    except LookupError as e:
        print(e)

    print("Terminating RASPA simulate processes...")
    kill_simulation_processes()

    time.sleep(1)

    print("Committing pending DB transactions...")
    commit_db(args.db_path)
    print("Safe shutdown complete.")


if __name__ == '__main__':
    main()
