import sys
from pprint import pprint

if len(sys.argv) != 2:
    print("usage: dedup.py <results>")
    exit(1)

results_file = sys.argv[1]

already_processed = set()

def to_conf(result_line):
    return ",".join(result_line.strip().split(",")[:8])

printed = 0
discarded = 0
total = 0
to_print = []
with open(results_file, "r") as f:
    results = []
    for line in f:
        total += 1
        conf = to_conf(line)

        if conf not in already_processed:
            to_print += [line]
            printed += 1
        else:
            discarded += 1

        already_processed.add(conf)

print(f"printed: {printed}", file=sys.stderr)
print(f"discarded: {discarded}", file=sys.stderr)
print(f"total: {total}", file=sys.stderr)
for item in to_print:
    print(item, end="")
