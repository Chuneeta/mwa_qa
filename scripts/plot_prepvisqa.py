from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import pylab
import os

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'


def get_indices(array1, array2):
    inds = []
    for elm in array2:
        inds.append(np.where(np.array(array1) == elm)[0][0])
    return inds


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
parser.add_argument('--title', default=None, help='Plot title')

args = parser.parse_args()
metrics = ut.load_json(args.json)
antennas = metrics['ANNUMBERS']
nants = metrics['NANTS']
poor_ants_xx = metrics['XX']['BAD_ANTS']
poor_ants_yy = metrics['YY']['BAD_ANTS']
poor_ants_xx_inds = get_indices(antennas, poor_ants_xx)
poor_ants_yy_inds = get_indices(antennas, poor_ants_yy)
obsid = metrics['OBSID']
threshold = metrics['THRESHOLD']
if args.title:
    titlename = args.title

elif len(obsid.split('_')) == 1:
    titlename = obsid
else:
    titlename = ''.join(filter(str.isdigit, os.path.basename(args.json)))

# plotting RMS
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=13)
ax = pylab.subplot(111)
ax.scatter(antennas, metrics['XX']['RMS'], marker='.',
           color='dodgerblue', s=100, alpha=0.8, label='XX')
ax.scatter(antennas, metrics['YY']['RMS'], marker='.',
           color='yellowgreen', s=100, alpha=0.8, label='YY')
ax.scatter(poor_ants_xx, np.array(metrics['XX']['RMS'])
           [poor_ants_xx_inds], s=100, marker='o', edgecolor='blue',
           facecolor='None')
ax.scatter(poor_ants_yy, np.array(metrics['YY']['RMS'])
           [poor_ants_yy_inds], s=150, marker='o', edgecolor='green',
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
if 'FIRST' in metrics['XX']['MODZ_SCORE'].keys():
    ax.scatter(metrics['ANNUMBERS_FIRST'], metrics['XX']['MODZ_SCORE']['FIRST']['0'], marker='.',
               color='dodgerblue', s=100, alpha=0.8, label='XX')
    ax.scatter(metrics['ANNUMBERS_FIRST'], metrics['YY']['MODZ_SCORE']['FIRST']['0'], marker='.',
               color='yellowgreen', s=100, alpha=0.8, label='YY')
    ax.scatter(metrics['ANNUMBERS_SECOND'], metrics['XX']['MODZ_SCORE']['SECOND']['0'], marker='.',
               color='dodgerblue', s=100, alpha=0.8)
    ax.scatter(metrics['ANNUMBERS_SECOND'], metrics['YY']['MODZ_SCORE']['SECOND']['0'], marker='.',
               color='yellowgreen', s=100, alpha=0.8
               )
else:
    ax.scatter(antennas, np.abs(metrics['XX']['MODZ_SCORE']['0']), marker='.',
               color='dodgerblue', s=100, alpha=0.8, label='XX')
    ax.scatter(antennas, np.abs(metrics['YY']['MODZ_SCORE']['0']), marker='.',
               color='yellowgreen', s=100, alpha=0.8, label='YY')
ax.fill_between(antennas, -threshold -
                20, -threshold, interpolate=True, color='red', alpha=0.2)
ax.fill_between(antennas, threshold,
                threshold + 20, interpolate=True, color='red', alpha=0.2)
# ax.axhline(threshold, linestyle='dashed', color='green', linewidth=2)
ax.grid(ls='dotted')
ax.legend()
ax.set_xlabel('Antenna Number')
ax.set_ylabel('Modified zscore')
ax.set_ylim(-10, 10)
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
