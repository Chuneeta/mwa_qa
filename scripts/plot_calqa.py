from argparse import ArgumentParser
from mwa_qa.read_calqa import CalQA
from mwa_qa import json_utils as ut
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import pylab

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

parser = ArgumentParser(
    description="Plotting CalQA metrics")
parser.add_argument('json', type=str,
                    help='MWA cal metrics json file')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='out', default=None,
                    help='Name of ouput figure name. Default calmetrics')
parser.add_argument('--dpi', dest='dpi', default=100,
                    help='Number of dots per inch to use to save the figures')

args = parser.parse_args()
calqa = CalQA(args.json)
if args.out is None:
    figname_rms = args.json.replace('.json', '_rms.png')
    figname_fft = args.json.replace('.json', '_fft.png')
else:
    if args.out.split('.')[-1] != 'png':
        figname_rms = args.out + '_rms.png'
        figname_fft = args.out + '_fft.png'
    else:
        figname_rms = args.out.replace('.png', '_rms.png')
        figname_fft = args.out.replace('.png', '_fft.png')
calqa.plot_rms(save=args.save, figname=figname_rms, dpi=args.dpi)
calqa.plot_fft(save=args.save, figname=figname_fft, dpi=args.dpi)
calqa.plot_average_dspectra(save=args.save, figname=args.out, dpi=args.dpi)
