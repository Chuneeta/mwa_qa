#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.combine_qa import DataFrameQA

parser = ArgumentParser(
    description="Combine QA from calibration solutions and \
        images into a single QA")
parser.add_argument('--cal', nargs='+', type=Path,
                    help='Hyperdrive json files')
parser.add_argument('--img', nargs='+', type=Path,
                    help='WSClean images json files')
parser.add_argument('--out', help='numpy output path', type=str,
                    default=None, required=False
                    )

args = parser.parse_args()
dframe = DataFrameQA(args.cal, args.img)
df_xx = dframe.create_dataframe('XX')
df_yy = dframe.create_dataframe('YY')
if args.out is None:
    filename = 'calimq_qa'
else:
    filename = args.out
# saving files to csv
df_xx.to_csv(filename + '_xx.csv')
df_yy.to_csv(filename + '_yy.csv')
