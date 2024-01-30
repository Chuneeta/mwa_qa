#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.vis_metrics import VisMetrics

parser = ArgumentParser(description="QA for MWA UVFITS visibility data")
parser.add_argument('--cutoff_threshold', type=float, default=3,
                    help='Cutoff threshold value for modified z-score to identify outliers.')
parser.add_argument('uvfits', type=Path, help='UVFITS visibility file')
parser.add_argument('--out', help='json output path', type=str,
                    default=None, required=False)

args = parser.parse_args()
m = VisMetrics(str(args.uvfits), args.cutoff_threshold)
m.run_metrics()
m.write_to(args.out)
