"""

 utils.py (author: Anson Wong / git: ankonzoid)

"""
import os

# Create directory (only if doesn't exist)
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir