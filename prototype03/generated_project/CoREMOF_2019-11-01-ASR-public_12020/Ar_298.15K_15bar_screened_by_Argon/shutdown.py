#!/usr/bin/env python3
"""
Safe shutdown utility for GCMC workflows.
Terminates workflow runner processes (low_pressure_gcmc or active_learning_gcmc),
and RASPA simulate processes, and commits DB.

Usage:
    python safe_shutdown.py [low|active] [--db-path path/to/db] [--pal_nodes node1 node2 ...]
"""
import sys
import time
import signal
import sqlite3
import argparse
import subprocess
from pathlib import Path

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
    return matches


def safe_terminate(process, timeout: float = 5.0):
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
    try:
        conn = sqlite3.connect(str(db_path))
        conn.commit()
        conn.close()
        print("Database commit complete.")
    except Exception as e:
        print(f"Database commit error: {e}")


def remote_shutdown(node: str, workflow: str, db_path: Path):
    script_path = Path(__file__).resolve()
    remote_dir = script_path.parent
    remote_cmd = f"cd {remote_dir} && python3 {script_path.name} {workflow} --db-path {db_path}"
    try:
        print(f"[{node}] Running remote shutdown...")
        subprocess.run(['ssh', node, remote_cmd], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{node}] Remote shutdown failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Safely shutdown GCMC workflow')
    parser.add_argument('workflow', choices=['low', 'active'],
                        help='Which workflow to shutdown')
    parser.add_argument('--db-path', type=Path, default=Path.cwd() / 'mof_project.db',
                        help='Path to SQLite database file')
    parser.add_argument('--pal_nodes', nargs='*', default=[],
                        help='List of additional nodes (hostnames) to shutdown remotely')
    args = parser.parse_args()

    wf_map = {
        'low': 'low_pressure_gcmc',
        'active': 'active_learning_gcmc'
    }
    wf_name = wf_map[args.workflow]

    # Local shutdown
    runners = find_workflow_processes(wf_name)
    if runners:
        for runner in runners:
            print(f"Terminating workflow runner PID {runner.pid} ({wf_name})")
            safe_terminate(runner)
    else:
        print(f"No local runner process found for '{wf_name}'")

    print("Terminating RASPA simulate processes locally...")
    kill_simulation_processes()

    print("Committing local DB...")
    commit_db(args.db_path)

    # Remote shutdown
    for node in args.pal_nodes:
        remote_shutdown(node, args.workflow, args.db_path)

    print("✅ Safe shutdown complete.")


if __name__ == '__main__':
    main()
