import sys
import os
import subprocess
import time
import signal
import requests

URL = os.environ["URL"]
assert URL is not None

processes = []
max_process_count = 60

def signal_handler(sig, frame):
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

try:
    config = requests.get(URL + "/configuration").json()
    while config is not None:
        framework = config[2]
        if len(processes) > max_process_count:
            print(f"Running the max {len(processes)} processes, now waiting...")
        while (len(processes) > max_process_count) or (framework == "tensorflow" and len(processes) >= 5):
            time.sleep(10)
            for process in processes:
                process.poll()
                if process.returncode is not None:
                    with open("log", "a", encoding = "utf-8") as of:
                        of.write(", ".join(config) + " == " + str(process.returncode) + "\n")
            processes = [p for p in processes if p.returncode is None]
        processes += [subprocess.Popen(config)]
        print(" ".join(config))
        config = requests.get(URL + "/configuration").json()
except:
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    raise sys.exc_info()[0]

