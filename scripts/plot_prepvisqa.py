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
antennas = np.arange(metrics['NANTS'])
freqs = np.arange(metrics['NFREQS'])
poor_ants_xx = metrics['XX']['POOR_ANTENNAS']
poor_ants_yy = metrics['YY']['POOR_ANTENNAS']
obsid = metrics['OBSID']
if len(obsid.split('_')) == 1:
    titlename = obsid
else:
    titlename = ''.join(filter(str.isdigit, args.json))
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=15)
ax = pylab.subplot(211)
ax.scatter(freqs, metrics['XX']['RMS_AMP_ANT'], marker='.',
           color='indianred', alpha=0.8, label='XX')
ax.scatter(freqs, metrics['YY']['RMS_AMP_ANT'], marker='.',
           color='dodgerblue', alpha=0.8, label='YY')
ax.grid()
ax.legend()
ax.set_xlabel('Frequency Channel')
ax.set_ylabel('RMS(ant)')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax = pylab.subplot(212)
ax.scatter(antennas, metrics['XX']['RMS_AMP_FREQ'], marker='o',
           color='indianred', s=30, alpha=0.8, label='XX')
ax.scatter(antennas, metrics['YY']['RMS_AMP_FREQ'], marker='o',
           color='dodgerblue', s=15, alpha=0.8, label='YY')
ax.scatter(poor_ants_xx, np.array(metrics['XX']['RMS_AMP_FREQ'])
           [poor_ants_xx], s=100, marker='o', edgecolor='red',
           facecolor='None')
ax.scatter(poor_ants_yy, np.array(metrics['YY']['RMS_AMP_FREQ'])
           [poor_ants_yy], s=150, marker='o', edgecolor='blue',
           facecolor='None')
ax.grid(ls='dotted')
ax.set_xlabel('Antenna Number')
ax.set_ylabel('RMS(freq)')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
pylab.subplots_adjust(hspace=0.3, left=0.15)
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_rms.png')
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
