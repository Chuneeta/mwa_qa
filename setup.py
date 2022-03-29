from setuptools import setup
import sys
import os
from mwa_clysis import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('mwa_clysis', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('mwa_clysis', 'data') 

setup_args = {
    'name':         'mwa_clysis',
    'author':       'Chuneeta Nunhokee',
    'url':          'https://github.com/Chuneeta/mwa_clysis',
    'license':      'BSD',
    'version':      version.version,
    'description':  'MWA Calibration Solutions Analyzer.',
    'packages':     ['mwa_clysis'],
    'package_dir':  {'mwa_clysis': 'mwa_clysis'},
    'package_data': {'mwa_clysis': data_files},
    'install_requires': ['numpy>=1.14', 'scipy', 'astropy>3.0.0', 'matplotlib>=2.2', 'python_dateutil>=2.6.0', 'pytest'],
    'include_package_data': True,
    #'scripts': ['scripts/pspec_run.py', 'scripts/pspec_red.py',
    #            'scripts/bootstrap_run.py'],
    'zip_safe':     False,
}

if __name__ == '__main__':
    setup(*(), **setup_args)   
# apply(setup, (), setup_args)
