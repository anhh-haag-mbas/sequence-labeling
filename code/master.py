import sys
import os
import subprocess
import time
import signal

if len(sys.argv) not in [2]:
    print("usage: master.py configs")
    exit(1)

processes = []
max_process_count = 33

def signal_handler(sig, frame):
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def load_configs(path):
    configs = []
    with open(path, "r") as f:
        for line in f:
            configs += [line.split(", ")]
    return configs

try:
    for config in load_configs(sys.argv[1]):
        framework = config[2]
        if len(processes) > max_process_count:
            print(f"Running the max {len(processes)} processes, now waiting...")
        while (len(processes) > max_process_count) or (framework == "tensorflow" and len(processes) >= 5) or (framework == "pytorch" and len(processes) >= 4):
            time.sleep(10)
            for process in processes:
                process.poll()
                if process.returncode is not None:
                    with open("log", "a", encoding = "utf-8") as of:
                        of.write(", ".join(config) + " == " + str(process.returncode) + "\n")
            processes = [p for p in processes if p.returncode is None]
        processes += [subprocess.Popen(config)]
        print(" ".join(config))
except:
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    raise sys.exc_info()[0]

