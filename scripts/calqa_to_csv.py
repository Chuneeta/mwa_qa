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
parser.add_argument('--filter', dest='filter', action='store_true', default=False,
                    help='If set, a filtering will be performed to clean up the data')
parser.add_argument('--drop', dest='drop', action='store_true', default=False,
                    help='If set, will drop the columns used for filtering the data')
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
elif args.outfile.split('.')[-1] != 'csv':
    outfile = args.outfile + '.csv'
else:
    outfile = args.outfile

df = df.dropna(subset=['OBSID']).set_index('OBSID')
df.index = df.index.astype(int)
df.sort_index()
if args.filter:
    # dropping obsids which fail to pass the calibration process
    df.drop(df[df['STATUS'] == 'FAIL'].index, inplace=True)
    df.drop(df[df['UNUSED_BLS'] > 30].index, inplace=True)
    df.drop(df[df['UNUSED_ANTS'] > 30].index, inplace=True)
    df.drop(df[df['UNUSED_CHS'] > 30].index, inplace=True)
    df.drop(df[df['NON_CONVERGED_CHS'] > 30].index, inplace=True)
if args.drop:
    # dropping the above columns as well
    df.drop(subset=['STATUS', 'UNUSED_BLS',
            'UNUSED_ANTS', 'NON_CONVERGED_CHS'])
df.to_csv(outfile)
