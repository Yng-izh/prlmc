"""Surface-form reconnaissance of the Conflict_Resolution fact lines.

E6a has to synthesise its own probes: the benchmark's own questions are
multi-hop and paraphrased, so a fixed single-hop template scores 0/0 against
them. Rendering probes from a window's own fact lines needs the exact line
shapes first, and this prints them instead of guessing.
"""
import collections
import re

import pandas as pd

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"

df = pd.read_parquet(PARQUET)
ctx = df.iloc[0]["context"]
lines = [l for l in ctx.split("\n") if re.match(r"^[0-9]+\. ", l)]
print("n_lines", len(lines))

ENTITY = r"[A-Z][A-Za-z\u00c0-\u024f'\-]*(?: [A-Z][A-Za-z\u00c0-\u024f'\-]*)*"

pat = collections.Counter()
for l in lines:
    t = re.sub(r"^[0-9]+\. ", "", l).rstrip(".")
    t = re.sub(r"\[.*?\]", "X", t)
    t = re.sub(ENTITY, "<E>", t)
    t = re.sub(r"\b\d{4}\b", "<N>", t)
    pat[t] += 1

for p, c in pat.most_common(40):
    print(c, "|", p)

print("---- raw samples ----")
for l in lines[:12]:
    print(repr(l))
