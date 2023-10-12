#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.cal_metrics import CalMetrics

parser = ArgumentParser(
    description="QA for hyperdrive calibration solutions")
parser.add_argument('soln', type=Path, help='Hyperdrive fits file')
parser.add_argument('metafits', type=Path, help='MWA metadata fits file')
parser.add_argument('--pol', default='X',
                    help='Polarization, can be either X or Y')
parser.add_argument(
    '--out', help='json output path', type=str, default=None, required=False
)

args = parser.parse_args()
m = CalMetrics(str(args.soln), str(args.metafits), args.pol)
m.run_metrics()
m.write_to(str(args.out) if args.out else None)
