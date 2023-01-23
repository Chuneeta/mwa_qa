#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.read_metafits import Metafits
import pylab

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('metafits', type=str, help='MWA metadata fits file')
parser.add_argument('-m', '--marker', type=str,
                    default='.', help='plotting marker')
parser.add_argument('-c', '--color', type=str, default='dodgerblue',
                    help='plotting color')
parser.add_argument('-s', '--size', type=str, default=100,
                    help='size of plotting marker')
parser.add_argument('--annot', dest='annot', action='store_true',
                    default=None, help='Boolean to allow annotation of the tile numbers')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='figname', default=None,
                    help='Name of ouput figure name. Default calmetrics')
parser.add_argument('--dpi', dest='dpi', default=100,
                    help='Number of dots per inch to use to save the figures')

args = parser.parse_args()
print(args)
m = Metafits(args.metafits)
anpos_dict = m.anpos_dict()
ankeys = list(anpos_dict.keys())
# plotting the antenna positions
fig = pylab.figure()
for k in ankeys:
    pylab.scatter(anpos_dict[k][0], anpos_dict[k][1],
                  marker=args.marker, c=args.color, s=args.size)
    if args.annot:
        pylab.annotate(k, (anpos_dict[k][0], anpos_dict[k][1]))
pylab.xlabel('East-West (m)')
pylab.ylabel('North-West (m)')
pylab.tick_params(labelsize=10, direction='in', length=4, width=2)
if args.save:
    if args.figname is None:
        figname = args.metafits.replace('.metafits', '_antennas.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            print(args.figname)
            figname = args.figname + '.png'
        else:
            figname = args.figname

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()
