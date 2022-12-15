from setuptools import setup
from mwa_qa import version
import os
import json

data = [version.git_origin, version.git_hash,
        version.git_description, version.git_branch]
with open(os.path.join('mwa_qa', 'GIT_INFO'), 'w') as outfile:
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


data_files = package_files('mwa_qa', 'data')

setup_args = {
    'name':         'mwa_qa',
    'author':       'Chuneeta Nunhokee',
    'url':          'https://github.com/Chuneeta/mwa_qa',
    'license':      'BSD',
    'version':      version.version,
    'description':  'MWA Data Quality Analysis.',
    'packages':     ['mwa_qa'],
    'package_dir':  {'mwa_qa': 'mwa_qa'},
    'package_data': {'mwa_qa': data_files},
    'install_requires': ['numpy>=1.16.5', 'numpy<1.23.0', 'scipy',
                         'astropy>3.0.0', 'matplotlib>=2.2',
                         'python_dateutil>=2.6.0',
                         'pytest'],
    'include_package_data': True,
    'zip_safe':     False,
    'scripts': ['scripts/run_calqa.py', 'scripts/run_imgqa.py',
                'scripts/run_visqa.py', 'scripts/run_prepvisqa.py',
                'scripts/plot_ants.py', 'scripts/plot_reds.py',
                'scripts/plot_calqa.py', 'scripts/plot_imgqa.py',
                'scripts/plot_visqa.py', 'scripts/plot_caljson.py',
                'scripts/plot_prepvisqa.py', 'scripts/calqa_to_csv.py',
                'scripts/imgqa_to_csv.py', 'scripts/merge_csvfiles.py']
}

if __name__ == '__main__':
    setup(*(), **setup_args)
