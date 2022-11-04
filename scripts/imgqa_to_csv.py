from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from pathlib import Path
import pandas as pd


parser = ArgumentParser(
    description="Plotting image metrics")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')
parser.add_argument('--out', dest='outfile', default=None,
                    help='Name of ouput csvfile')
args = parser.parse_args()

pols = ['XX', 'YY', 'V']
main_keys = ['IMAGE_ID']
pol_keys = ['MEAN_ALL', 'RMS_ALL', 'MEAN_BOX', 'RMS_BOX']
pks_keys = ['PEAK_FLUX', 'INT_FLUX']

nmk = len(main_keys)
npk = len(pol_keys)
npsk = len(pks_keys)
nsk = nmk + npk + npsk
keys = main_keys.copy()
count = 0
for p in ['XX', 'YY', 'V']:
    for pk in pol_keys:
        keys.append('{}_{}'.format(pk, p))
    for ps in pks_keys:
        keys.append('PKS_{}_{}'.format(ps, p))

df = pd.DataFrame(columns=keys)
for i, json in enumerate(args.json):
    print(i, ' Reading {}'.format(json))
    data = ut.load_json(json)
    row = {}
    for k in main_keys[0:nmk]:
        row[k] = int((str(data['XX'][k]))[0:10])
    # XX
    for j, k in enumerate(keys[nmk:nmk + npk]):
        row[k] = data['XX'][pol_keys[j]]
    for j, k in enumerate(keys[nmk + npk: nsk]):
        row[k] = data['XX']['PKS0023_026'][pks_keys[j]]

    # YY
    for j, k in enumerate(keys[nsk:nsk + npk]):
        row[k] = data['YY'][pol_keys[j]]
    for j, k in enumerate(keys[nsk + npk: nsk + npk + npsk]):
        row[k] = data['YY']['PKS0023_026'][pks_keys[j]]

     # V
    for j, k in enumerate(keys[nsk + npk + npsk: nsk + 2 * npk + npsk]):
        row[k] = data['V'][pol_keys[j]]
    for j, k in enumerate(keys[nsk + 2 * npk + npsk: nsk + 2 * npk + 2 * npsk]):
        row[k] = data['V']['PKS0023_026'][pks_keys[j]]

    df = df.append(row, ignore_index=True)

if args.outfile is None:
    outfile = 'imgqa_combined.csv'
elif outfile.split('.')[-1] != 'csv':
    outfile += '.csv'

df = df.dropna(subset=['IMAGE_ID']).set_index('IMAGE_ID')
df.index = df.index.astype(int)
df.sort_index()
df.to_csv(outfile)
