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
redundant_met = metrics['REDUNDANT']
obsid = metrics['OBSID']

# amplitude chisquare
colors = list(mpl.colors.XKCD_COLORS.values())
amp_chisq_xx = redundant_met['XX']['AMP_CHISQ']
amp_chisq_yy = redundant_met['YY']['AMP_CHISQ']
poor_bls_xx = redundant_met['XX']['POOR_BLS']
poor_bls_yy = redundant_met['YY']['POOR_BLS']
red_groups = redundant_met['RED_PAIRS']
mn_limit = min([min([np.nanmin(csq) for csq in amp_chisq_xx]),
                min([np.nanmin(csq) for csq in amp_chisq_yy])])
mx_limit = max([max([np.nanmax(csq) for csq in amp_chisq_xx]),
                max([np.nanmax(csq) for csq in amp_chisq_yy])])
red_lengths = [np.linalg.norm(rp) for rp in red_groups]
if len(obsid.split('_')) == 1:
    titlename = obsid
else:
    titlename = ''.join(filter(str.isdigit, args.json))
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=15)
ax = pylab.subplot(211)
for i in range(len(amp_chisq_xx)):
    ax.semilogy(np.ones((len(amp_chisq_xx[i])))
                * i, amp_chisq_xx[i], '.', color=colors[i], alpha=0.7)
    # inds = [np.array(poor_bls_xx[i])
    # print(inds)
    # ax.semilogy(np.ones((len(poor_bls_xx[i]))) * i,
    # amp_chisq_xx[i][np.array(poor_bls_xx[i])[:, 1]], 's', color = 'r', alpha = 0.4)
ax.set_ylabel('CHISQ (XX)')
ax.grid(ls='dotted')
ax.set_ylim(mn_limit - 50, mx_limit)
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax = pylab.subplot(212)
for i in range(len(amp_chisq_yy)):
    ax.semilogy(np.ones((len(amp_chisq_yy[i])))
                * i, amp_chisq_yy[i], '.', color=colors[i], alpha=0.6)
ax.set_ylabel('CHISQ (YY)')
ax.grid(ls='dotted')
ax.set_ylim(mn_limit, mx_limit)
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax.set_xlabel('Group Number')
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
