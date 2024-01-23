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


def search_group(red_pairs, antp):
    for i, gp in enumerate(red_pairs):
        if list(antp) in gp:
            return i


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
modz_xx = redundant_met['XX']['MODZ']
modz_yy = redundant_met['YY']['MODZ']
inds_xx = redundant_met['XX']['POOR_BLS_INDS']
inds_yy = redundant_met['YY']['POOR_BLS_INDS']
poor_bls_xx = redundant_met['XX']['POOR_BLS']
poor_bls_yy = redundant_met['YY']['POOR_BLS']
poor_bls = metrics['POOR_BLS']
red_groups = redundant_met['RED_GROUPS']
red_pairs = redundant_met['RED_PAIRS']
# mn_limit = min([min([np.nanmin(csq) for csq in amp_chisq_xx]),
#                min([np.nanmin(csq) for csq in amp_chisq_yy])])
# mx_limit = max([max([np.nanmax(csq) for csq in amp_chisq_xx]),
#                max([np.nanmax(csq) for csq in amp_chisq_yy])])
# red_lengths = [np.linalg.norm(rp) for rp in red_groups]
if len(obsid.split('_')) == 1:
    titlename = obsid
else:
    titlename = ''.join(filter(str.isdigit, args.json))


modz_gridxx = np.zeros((128, 128))
modz_gridyy = np.zeros((128, 128))
modz_gridxx[:, :] = np.nan
modz_gridyy[:, :] = np.nan
for i, antp in enumerate(red_pairs):
    for j, (a1, a2) in enumerate(antp):
        group_number = search_group(red_pairs, (a1, a2))
        modz_gridxx[a1, a2] = modz_xx[group_number][j]
        modz_gridyy[a1, a2] = modz_yy[group_number][j]

ig = pylab.figure(figsize=(10, 5))
pylab.suptitle('Modified zscore - {}'.format(titlename))
ax = pylab.subplot(121)
im = ax.imshow(modz_gridxx, aspect='auto',
               cmap='viridis', vmin=-3.5, vmax=3.5)
ax.set_xlabel('Antenna 1')
ax.set_ylabel('Antenna 2')
ax.set_title('East West (XX)')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax.set_xlim(57, 128)
ax.set_ylim(57, 128)
pylab.colorbar(im)

ax = pylab.subplot(122)
im = ax.imshow(modz_gridyy, aspect='auto',
               cmap='viridis', vmin=-3.5, vmax=3.5, origin='lower')
ax.set_xlabel('Antenna 1')
ax.set_ylabel('Antenna 2')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax.set_title('North South (YY)')
ax.set_xlim(57, 128)
ax.set_ylim(57, 128)
pylab.colorbar(im)
pylab.subplots_adjust(hspace=0.2, left=0.15)

if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_modzgrid.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_modzgrid.png'
        else:
            figname = args.figname.split('.')[-2] + '_modzgrid.png'

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()


# plotting chisq
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=15)
ax = pylab.subplot(211)
for i in range(len(amp_chisq_xx)):
    ax.semilogy(np.ones((len(amp_chisq_xx[i])))
                * i, amp_chisq_xx[i], '.', color=colors[i], alpha=0.7)
    if len(inds_xx[i]) > 0:
        ax.semilogy(np.ones(len(inds_xx[i])) * i,
                    np.array(amp_chisq_xx[i])[inds_xx[i]], 'rx')
    # inds = [np.array(poor_bls_xx[i])
    # print(inds)
    # ax.semilogy(np.ones((len(poor_bls_xx[i]))) * i,
    # amp_chisq_xx[i][np.array(poor_bls_xx[i])[:, 1]], 's', color = 'r', alpha = 0.4)
ax.set_ylabel('CHISQ (XX)')
ax.grid(ls='dotted')
ax.set_ylim(10**1.5, 10**5)
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax = pylab.subplot(212)
for i in range(len(amp_chisq_yy)):
    ax.semilogy(np.ones((len(amp_chisq_yy[i])))
                * i, amp_chisq_yy[i], '.', color=colors[i], alpha=0.6)
    if len(inds_yy[i]) > 0:
        ax.semilogy(np.ones(len(inds_yy[i])) * i,
                    np.array(amp_chisq_yy[i])[inds_yy[i]], 'rx')
ax.set_ylabel('CHISQ (YY)')
ax.grid(ls='dotted')
ax.set_ylim(10**1.5, 10**5)
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax.set_xlabel('Group Number')
pylab.subplots_adjust(hspace=0, left=0.15)
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_chisq.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_chisq.png'
        else:
            figname = args.figname.split('.')[-2] + '_chisq.png'

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()

# plotting modz
fig = pylab.figure(figsize=(7, 5))
fig.suptitle(titlename, size=13)
ax = pylab.subplot(211)
for i in range(len(amp_chisq_xx)):
    ax.plot(np.ones((len(amp_chisq_xx[i])))
            * i, modz_xx[i], '.', color=colors[i], alpha=0.7)
ax.axhline(y=-3.5, ls='dashed', color='red')
ax.axhline(y=3.5, ls='dashed', color='red')
ax.set_ylabel('MOD ZSCORE (XX)')
ax.grid(ls='dotted')
ax.set_ylim(-8, 8)
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax = pylab.subplot(212)
for i in range(len(amp_chisq_yy)):
    ax.plot(np.ones((len(amp_chisq_yy[i])))
            * i, modz_yy[i], '.', color=colors[i], alpha=0.6)
ax.set_ylabel('MOD ZSCORE (YY)')
ax.grid(ls='dotted')
ax.set_ylim(-7, 7)
ax.axhline(y=-3.5, ls='dashed', color='red')
ax.axhline(y=3.5, ls='dashed', color='red')
ax.tick_params(labelsize=10, direction='in', length=4, width=2)
ax.set_xlabel('Group Number')
pylab.subplots_adjust(hspace=0, left=0.15)
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_modz.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_modz.png'
        else:
            figname = args.figname.split('.')[-2] + '_modz.png'

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()

# plotting antenna grids
fig = pylab.figure(figsize=(7, 5))
ant_grid = np.zeros((128, 128))
for (a1, a2) in poor_bls:
    ant_grid[a1, a2] = 1
pylab.imshow(ant_grid, aspect='auto', cmap='cubehelix')
pylab.grid(ls='dotted')
pylab.xlabel('Antenna 1')
pylab.ylabel('Antenna 2')
if args.save:
    if args.figname is None:
        figname = args.json.replace('.json', '_antgrid.png')
    else:
        if args.figname.split('.')[-1] != 'png':
            figname = args.figname + '_antgrid.png'
        else:
            figname = args.figname.split('.')[-2] + '_antgrid.png'

    pylab.savefig(figname, dpi=args.dpi)
    pylab.close()
else:
    pylab.show()
