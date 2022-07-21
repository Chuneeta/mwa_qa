import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_metrics(metrics, filename):
    """
    Writing metrics stored as a dictionary to json file
    - metrics : Dictionary containing the metrics
    - filename : Name of the output file.
    """
    if filename.split('.')[-1] != 'json':
        filename += '.json'
    with open(filename, 'w') as outfile:
        json.dump(metrics, outfile, indent=4, cls=NpEncoder)


def load_metrics(filename, filetype='json'):
    pass


def combine_jsons():
    pass


def combine_metrics():
    pass
