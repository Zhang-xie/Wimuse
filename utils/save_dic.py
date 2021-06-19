import pickle
import os
from pathlib import Path

def save_as_pickle(obj, path, name ):
    with open(os.path.join(path,name), 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    a = load_obj('../lightning_logs/version_19/test_result.pickle')
    print(a)