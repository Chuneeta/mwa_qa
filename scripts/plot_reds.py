#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.read_metafits import Metafits
from matplotlib.pyplot import cm
import pylab
import numpy as np

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('metafits', type=Path, help='MWA metadata fits file')
parser.add_argument('pol', default=None,
                    help='Polarization, can be either X or Y')

markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8',
           's', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
colors = ['r', 'b', 'g', 'b', 'y']
args = parser.parse_args()
m = Metafits(args.metafits, args.pol)
anpos_dict = m.anpos_dict()
reds_dict = m.redundant_antpairs()
redkeys = list(reds_dict.keys())
# plotting redundancy groups
colors *= (int(len(redkeys) / len(colors)) + 1)
markers *= (int(len(redkeys) / len(markers)) + 1)
fig = pylab.figure()
for i, rkey in enumerate(redkeys):
    antpairs = reds_dict[rkey]
    for antpair in antpairs:
        ant1, ant2 = antpair
        delta_x = np.abs(anpos_dict[ant1][0] - anpos_dict[ant2][0])
        delta_y = np.abs(anpos_dict[ant1][1] - anpos_dict[ant2][1])
        pylab.scatter(delta_x, delta_y, marker=markers[i], c=colors[i], s=5)
    pylab.annotate(rkey, (rkey[0], rkey[1]))
pylab.show()
