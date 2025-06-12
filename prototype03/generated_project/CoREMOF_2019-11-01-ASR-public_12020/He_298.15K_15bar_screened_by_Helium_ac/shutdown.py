#!/usr/bin/env python3
"""
Safe shutdown utility for Low-pressure GCMC workflow.
Finds and terminates all running workflow runner processes and any lingering RASPA simulate processes,
commits pending database transactions, and exits cleanly.

Usage:
    python safe_shutdown.py [--db-path path/to/mof_project.db]
"""
import sys
import time
import signal
import sqlite3
from pathlib import Path

try:
    import psutil
except ImportError:
    print("psutil is required. Install via `pip install psutil`.")
    sys.exit(1)


def find_workflow_processes(name_snippet: str = 'low_pressure_gcmc') -> list:
    """
    Search for all processes whose command line contains the given snippet.
    Returns a list of matching psutil.Process. Raises LookupError if none found.
    """
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
    """
    Gracefully terminate the given process and its children.
    Sends SIGTERM, waits up to timeout, then SIGKILL if still alive.
    """
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
    """
    Terminate any RASPA simulate processes matching the given command snippet.
    """
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
    """
    Commit any pending SQLite transactions and close the connection.
    """
    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        return
    conn = sqlite3.connect(str(db_path))
    conn.commit()
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Safely shutdown GCMC workflow')
    parser.add_argument('--db-path', type=Path, default=Path.cwd() / 'mof_project.db',
                        help='Path to SQLite database file')
    args = parser.parse_args()

    # Step 1: terminate all workflow runners
    try:
        runners = find_workflow_processes()
        for runner in runners:
            print(f"Terminating workflow runner PID {runner.pid}")
            safe_terminate(runner)
    except LookupError as e:
        print(e)

    # Step 2: terminate any simulate processes
    print("Terminating RASPA simulate processes...")
    kill_simulation_processes()

    # Step 3: wait for cleanup
    time.sleep(1)

    # Step 4: commit DB
    print("Committing pending DB transactions...")
    commit_db(args.db_path)
    print("Safe shutdown complete.")


if __name__ == '__main__':
    main()
