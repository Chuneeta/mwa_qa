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
calqa.plot_rms(save=args.save, figname=args.out, dpi=args.dpi)
calqa.plot_fft(save=args.save, figname=args.out, dpi=args.dpi)
calqa.plot_average_dspectra(save=args.save, figname=args.out, dpi=args.dpi)
