import sys
import torch

mode = sys.argv[1]
chunks = []
try:
    while True:
        chunks.append(torch.zeros(512 * 1024 * 1024, dtype=torch.float16, device="cuda"))
except Exception as exc:
    print("MODE=%s" % mode)
    print("HINT_PRESENT=%s" % ("expandable_segments:True to avoid" in str(exc)))
    print(str(exc)[:900])
print("allocated_gib=%d" % len(chunks))
