import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import psutil
import requests


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "server_state.json"


def owned_process(state):
    try:
        process = psutil.Process(state["pid"])
        if abs(process.create_time() - state["create_time"]) > 0.01:
            raise RuntimeError("PID was reused; refusing to manage this process")
        command = " ".join(process.cmdline())
        if str(ROOT) not in command or not any(
            name in command for name in ("server_entry.py", "launch_server.sh")
        ):
            raise RuntimeError(f"Refusing to manage an unrelated process: {command}")
        return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def stop_server():
    if not STATE_FILE.exists():
        print("No experiment server is registered.")
        return
    state = json.loads(STATE_FILE.read_text())
    parent = owned_process(state)
    if parent is not None:
        processes = parent.children(recursive=True) + [parent]
        state["stopping_processes"] = []
        for process in processes:
            try:
                state["stopping_processes"].append(
                    {"pid": process.pid, "create_time": process.create_time()}
                )
            except psutil.NoSuchProcess:
                pass
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    else:
        processes = []
        for saved in state.get("stopping_processes", []):
            try:
                process = psutil.Process(saved["pid"])
                if abs(process.create_time() - saved["create_time"]) < 0.01:
                    processes.append(process)
            except psutil.NoSuchProcess:
                pass
    if processes:
        for process in reversed(processes):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        surviving = wait_for_exit(processes, 40)
        for process in surviving:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        surviving = wait_for_exit(surviving, 20)
        if surviving:
            raise RuntimeError("Experiment workers did not stop")
    STATE_FILE.unlink()
    print("Stopped the experiment server and its workers.", flush=True)


def wait_for_exit(processes, timeout):
    deadline = time.monotonic() + timeout
    while processes:
        surviving = []
        for process in processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    surviving.append(process)
            except psutil.NoSuchProcess:
                pass
        if not surviving or time.monotonic() >= deadline:
            return surviving
        processes = surviving
        time.sleep(0.25)
    return []


def start_server(mode, run_directory):
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        if owned_process(state) is not None:
            raise RuntimeError("An experiment server is already running")
        STATE_FILE.unlink()
    port = int(os.environ.get("SERVER_PORT", "30180"))
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    run_directory = run_directory.resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    log_path = run_directory / "server.log"
    with log_path.open("ab", buffering=0) as logfile:
        process = subprocess.Popen(
            ["bash", str(ROOT / "launch_server.sh"), mode, str(run_directory)],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
    state = {
        "pid": process.pid,
        "create_time": psutil.Process(process.pid).create_time(),
        "mode": mode,
        "run_directory": str(run_directory),
        "base_url": f"http://127.0.0.1:{port}",
    }
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


def wait_server(timeout):
    state = json.loads(STATE_FILE.read_text())
    deadline = time.monotonic() + timeout
    with requests.Session() as session:
        session.trust_env = False
        while time.monotonic() < deadline:
            if owned_process(state) is None:
                raise RuntimeError(
                    f"Server exited; inspect {state['run_directory']}/server.log"
                )
            try:
                response = session.get(state["base_url"] + "/health", timeout=5)
                if response.status_code == 200:
                    response = session.get(
                        state["base_url"] + "/get_server_info", timeout=30
                    )
                    response.raise_for_status()
                    destination = Path(state["run_directory"]) / "server_info.json"
                    destination.write_text(json.dumps(response.json(), indent=2) + "\n")
                    print(
                        f"Server is ready; configuration saved to {destination}",
                        flush=True,
                    )
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
    raise TimeoutError(f"Server did not become healthy in {timeout} seconds")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument(
        "mode",
        choices=("bounded_on_l2", "bounded_off_host"),
    )
    start.add_argument("run_directory", type=Path)
    wait = commands.add_parser("wait")
    wait.add_argument("--timeout", type=float, default=1800)
    commands.add_parser("stop")
    arguments = parser.parse_args()
    if arguments.command == "start":
        start_server(arguments.mode, arguments.run_directory)
    elif arguments.command == "wait":
        wait_server(arguments.timeout)
    else:
        stop_server()


if __name__ == "__main__":
    main()
