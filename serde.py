"""
It's just a wrapper around pickle
"""

import pickle

def serialize(ids):
    return pickle.dumps(ids)

def deserialize(ids):
    return pickle.loads(ids)
