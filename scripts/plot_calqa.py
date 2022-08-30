from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from mwa_qa.read_calqa import CalQA
from pathlib import Path
import matplotlib as mpl
import numpy as np
import pylab

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('json', type=Path,
                    help='MWA metrics json file')
parser.add_argument('--t', type=int, dest='timestamp', default=0,
                    help='Observation timestamp to plot. Default is 0')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='figname', default='calmetrics',
                    help='Name of ouput figure name. Default calmetrics')
parser.add_argument('--cmap', dest='cmap', default='hot',
                    help='CMAP for 2D matplotlib plot')
parser.add_argument('--vmin', dest='vmin', default=-1,
                    help='Minimum value of the colorbar for the 2D matplotlib')
parser.add_argument('--vmax', dest='vmax', default=2.5,
                    help='Minimum value of the colorbar for the 2D matplotlib')
args = parser.parse_args()

if args.figname.split('.')[-1] == 'png':
    outfile1 = args.figname.replace('.png', '_fft.png')
    outfile2 = args.figname.replace('.png', '_stats.png')
else:
    outfile1 = args.figname + '_fft.png'
    outfile2 = args.figname + '_stats.png'

calqa = CalQA(args.json)
calqa.plot_fft(args.timestamp, vmin=args.vmin, vmax=args.vmax,
               cmap=args.cmap, save=args.save, figname=outfile1)
calqa.plot_amp_variances(timestamp=args.timestamp,
                         save=args.save, figname=outfile2)
