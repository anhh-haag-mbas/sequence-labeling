import sys
from pprint import pprint

if len(sys.argv) != 3:
    print("usage: find_failed.py <sent> <results>")
    exit(1)

sent_file = sys.argv[1]
results_file = sys.argv[2]

with open(sent_file, "r") as f:
    sent = []
    for line in f:
        item = eval(line)
        if item is None: continue
        sent += [",".join(item[2:])]

with open(results_file, "r") as f:
    results = []
    for line in f:
        results += [",".join(line.strip().split(",")[:8])]

sent = set(sent)
results = set(results)
missing = []

for item in sent:
    if item not in results:
        missing += [item]

for item in reversed(sorted(missing)):
    print(repr(["python3", "slave.py"] + item.split(",")))
# pprint(sent)
# pprint(results)
