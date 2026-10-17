#!/usr/bin/env python3
"""Independent load-only S1d C generation entry; no optimizer is executed."""
from s1d_c_runtime import main
import sys

if __name__ == '__main__':
    if '--c-checkpoint' not in sys.argv or '--c-reference' not in sys.argv:
        raise SystemExit('load-only entry requires --c-checkpoint and --c-reference')
    main()
