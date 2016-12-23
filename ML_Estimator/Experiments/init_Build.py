import sys
import os

def init_build():
    dirs = ["../", "../python/"]

    for dir in dirs:
        runDir = os.path.abspath(os.path.join(os.path.dirname(__file__), dir))
        if not runDir in sys.path:
            sys.path.insert(1,runDir)

        print "Added ", runDir, " to sys path"
        
    return
