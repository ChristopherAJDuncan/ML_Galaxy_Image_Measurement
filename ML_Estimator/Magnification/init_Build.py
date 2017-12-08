import sys
import os

def init_build():
    runDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not runDir in sys.path:
        sys.path.insert(1,runDir)

    return
