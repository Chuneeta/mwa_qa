#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.read_metafits import Metafits
import pylab

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('metafits', type=Path, help='MWA metadata fits file')
parser.add_argument('pol', default=None,
                    help='Polarization, can be either X or Y')

args = parser.parse_args()
m = Metafits(args.metafits, args.pol)
anpos_dict = m.anpos_dict()
ankeys = list(anpos_dict.keys())
# plotting the antenna positions
fig = pylab.figure()
for k in ankeys:
    pylab.plot(anpos_dict[k][0], anpos_dict[k][1], '.', color='dodgerblue')
    pylab.annotate(k, (anpos_dict[k][0], anpos_dict[k][1]))
pylab.xlabel('East-West (m)')
pylab.ylabel('North-West (m')
pylab.show()
