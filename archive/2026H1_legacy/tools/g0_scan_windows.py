"""Offline scan: which (row, start) windows contain the most version pairs.

Pure text analysis, no GPU. Mirrors build_window's line coverage
(anchor 30 lines + 19 chunks x 20 lines) and counts same-(template,subject)
groups that occur >= 2 times inside the window.
"""

import re
import sys

import pandas as pd

sys.path.insert(0, "/share/latent/PRLM-C/tools")
import e6a_overfit as E  # noqa: E402

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"
WINDOW_LINES = 19 * 20  # 19 chunks x 20 lines
ANCHOR = 30


def main():
    frame = pd.read_parquet(PARQUET)
    print("row start groups pairs win_lo win_hi n_lines")
    for row in range(len(frame)):
        lines = E.read_fact_lines(row)
        for start in range(30, 46, 5):
            lo = start + ANCHOR
            hi = lo + WINDOW_LINES
            if hi > len(lines):
                continue
            window = range(lo, hi)
            groups = {}
            for index in window:
                found = E.first_match(lines[index])
                if found is None:
                    continue
                slot, subject, _a, _q = found
                groups.setdefault((slot, subject), []).append(index)
            multi = {k: v for k, v in groups.items() if len(v) >= 2}
            pairs = sum(len(v) - 1 for v in multi.values())
            print("%3d %5d %6d %5d %6d %6d %7d"
                  % (row, start, len(multi), pairs, lo, hi, len(lines)))


if __name__ == "__main__":
    main()
