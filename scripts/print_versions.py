#!/usr/bin/env python3
"""Print the exact versions of every tool in the certified path."""
import sys, platform
print("Python      :", platform.python_version(), "(", sys.version.split()[0], ")")
try:
    import mpmath; print("mpmath      :", mpmath.__version__)
except Exception as e:
    print("mpmath      : NOT FOUND", e)
try:
    import numpy; print("numpy       :", numpy.__version__)
except Exception as e:
    print("numpy       : NOT FOUND", e)
try:
    import importlib.metadata as md
    print("highspy     :", md.version("highspy"), "(LP solve stage only; not on certify path)")
except Exception as e:
    print("highspy     : NOT FOUND", e)
try:
    import highspy
    h = highspy.Highs(); print("HiGHS core  :", h.version())
except Exception:
    pass
