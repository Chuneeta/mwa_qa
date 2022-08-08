#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.image_metrics import ImgMetrics

parser = ArgumentParser(description="QA for MWA images")
parser.add_argument('fits', type=Path, nargs='+',
                    help='FITS image file(s)')
parser.add_argument('--out', help='json output path',
                    type=Path, default=None, required=False)

args = parser.parse_args()
m = ImgMetrics([*map(str, args.fits)])
m.run_metrics()
m.write_to(args.out)
