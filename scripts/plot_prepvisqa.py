from argparse import ArgumentParser
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
    description="Plotting antenna positions")
parser.add_argument('json', type=str,
                    help='MWA metrics json file')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='figname', default=None,
                    help='Name of ouput figure name. Default calmetrics')
parser.add_argument('--dpi', dest='dpi', default=100,
                    help='Number of dots per inch to use to save the figures')

args = parser.parse_args()
metrics = ut.load_json(args.json)
antennas = metrics['ANNUMBERS']
nants = metrics['NANTS']
poor_ants_xx = metrics['XX']['BAD_ANTS']
poor_ants_yy = metrics['YY']['BAD_ANTS']
obsid = metrics['OBSID']
threshold = metrics['THRESHOLD']
if len(obsid.split('_')) == 1:
    titlename = obsid
else:
    titlename = ''.join(filter(str.isdigit, args.json))

# plotting RMS
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=13)
ax = pylab.subplot(111)
ax.scatter(antennas, metrics['XX']['RMS'], marker='.',
           color='dodgerblue', s=100, alpha=0.8, label='XX')
ax.scatter(antennas, metrics['YY']['RMS'], marker='.',
           color='indianred', s=100, alpha=0.8, label='YY')
ax.scatter(poor_ants_xx, np.array(metrics['XX']['RMS'])
           [poor_ants_xx], s=100, marker='o', edgecolor='blue',
           facecolor='None')
ax.scatter(poor_ants_yy, np.array(metrics['YY']['RMS'])
           [poor_ants_yy], s=150, marker='o', edgecolor='red',
           facecolor='None')
ax.grid(ls='dotted')
ax.legend()
ax.set_xlabel('Antenna Number')
ax.set_ylabel('RMS')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_rms.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_rms.png'
        else:
            figname = args.figname.replace('.png', '_rms.png')

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()

# plotting modified zscore
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=13)
ax = pylab.subplot(111)
ax.scatter(antennas, np.abs(metrics['XX']['MODZ_SCORE']["0"]), marker='.',
           color='dodgerblue', s=100, alpha=0.8, label='XX')
ax.scatter(antennas, np.abs(metrics['YY']['MODZ_SCORE']["0"]), marker='.',
           color='indianred', s=100, alpha=0.8, label='YY')
ax.axhline(threshold, linestyle='dashed', color='green', linewidth=2)
ax.grid(ls='dotted')
ax.legend()
ax.set_xlabel('Antenna Number')
ax.set_ylabel('Modified zscore')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_modz.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_modz.png'
        else:
            figname = args.figname.replace('.png', '_modz.png')

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()
