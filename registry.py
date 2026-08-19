"""Factor registry: every tested candidate is appended, pass or fail.
The line count is M -- the honest number of tries behind any 'discovery'.
"""
import json
import os


def append(path, record):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def count(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)
