#!/usr/bin/env python
from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from pathlib import Path
import pandas as pd

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')
parser.add_argument('--out', dest='outfile', default=None,
                    help='Name of ouput csvfile')
args = parser.parse_args()

main_keys = ['OBSID', 'STATUS', 'UNUSED_BLS', 'UNUSED_ANTS',
             'UNUSED_CHS', 'NON_CONVERGED_CHS', 'CONVERGENCE_VAR']
pol_keys = ['SKEWNESS_UVCUT', 'RMS_AMPVAR_FREQ', 'RMS_AMPVAR_ANT', 'DFFT_POWER',
            'DFFT_POWER_HIGH_PKPL', 'DFFT_POWER_HIGH_NKPL', 'RECEIVER_CHISQVAR']

# creating dataframe

nmkeys = len(main_keys)
npkeys = len(pol_keys)
keys = main_keys.copy()
count = 0
for p in ['XX', 'YY']:
    for pk in pol_keys:
        keys.append('{}_{}'.format(pk, p))

df = pd.DataFrame(columns=keys)
for i, json in enumerate(args.json):
    print(i, ' Reading {}'.format(json))
    data = ut.load_json(json)
    row = {}
    for k in main_keys:
        row[k] = data[k]
    for j, k in enumerate(keys[nmkeys:nmkeys + npkeys]):
        row[k] = data['XX'][pol_keys[j]]
    for j, k in enumerate(keys[nmkeys + npkeys:nmkeys + 2 * npkeys]):
        row[k] = data['YY'][pol_keys[j]]
    df = df.append(row, ignore_index=True)

if args.outfile is None:
    outfile = 'calqa_combined.csv'
elif outfile.split('.')[-1] != 'csv':
    outfile += '.csv'

df = df.dropna(subset=['OBSID']).set_index('OBSID')
df.index = df.index.astype(int)
df.sort_index()
df.to_csv(outfile)
