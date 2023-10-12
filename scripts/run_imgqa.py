#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.image_metrics import ImgMetrics

parser = ArgumentParser(description="QA for MWA images")
parser.add_argument('fits', type=Path, nargs='+',
                    help='FITS image file(s)')
parser.add_argument('--const', type=int, default=1,
                    help='Beam contant to multiply for the beam radius for region selection')
parser.add_argument('--out', help='json output path',
                    type=str, default=None, required=False)

args = parser.parse_args()
m = ImgMetrics([*map(str, args.fits)])
m.run_metrics(beam_const=args.const)
m.write_to(str(args.out) if args.out else None)
