import os, sys
try:
    import mwa_qa
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mwa_qa')))
    import mwa_qa