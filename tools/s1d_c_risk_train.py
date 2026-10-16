#!/usr/bin/env python3
"""Parallel C+RiskProtect entry, using the same frozen C runtime."""
from s1d_c_runtime import main
import sys

if __name__ == '__main__':
    if '--risk-protection' not in sys.argv or '--c-checkpoint' in sys.argv:
        raise SystemExit('training entry requires --risk-protection and cannot load evaluation checkpoint')
    main()
