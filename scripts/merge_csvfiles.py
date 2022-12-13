from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from mwa_qa.read_csv import CSV

parser = ArgumentParser(
    description="Combine QA from calibration solutions and \
        images into a single QA")
parser.add_argument('csvfiles', nargs='+', type=Path,
                    help='QA csv files')
parser.add_argument('--out', help='combined csv path', type=str,
                    default=None, required=False)

args = parser.parse_args()


def read_csv(csvfile):
    try:
        csv_df = CSV(csvfile, obs_key='OBSID')
    except KeyError:
        try:
            csv_df = CSV(csvfile, obs_key='IMAGE_ID')
        except KeyError:
            csv_df = CSV(csvfile, obs_key='OBS')
    return csv_df


df_csv = read_csv(args.csvfiles[0]).data
for csv in args.csvfiles:
    print(csv)
    df1_csv = read_csv(csv)
    df_csv = pd.merge(df_csv, df1_csv.data,
                      left_index=True, right_index=True)

print(df_csv)
# saving dataframe
if args.out is None:
    outfile = 'combined_qa.csv'
else:
    outfile = args.out
df_csv.to_csv(outfile)
