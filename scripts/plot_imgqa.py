from argparse import ArgumentParser
from mwa_qa import json_utils as ut
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
    description="Plotting image metrics")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='figname', default=None,
                    help='Name of ouput figure name')

args = parser.parse_args()
# Stokes XX
rms_all_sxx = []
rms_box_sxx = []
pks_pflux_sxx = []
pks_tflux_sxx = []

# Stokes YY
rms_all_syy = []
rms_box_syy = []
pks_pflux_syy = []
pks_tflux_syy = []

# Stokes XX
rms_all_sv = []
rms_box_sv = []
pks_pflux_sv = []
pks_tflux_sv = []

# imsize = args.json[0]['IMSIZE']
imsize = [4096, 4096]
bxsize = ut.load_json(args.json[0])['PIX_BOX']
count = 0
obsids = []
for json in args.json:
    print('Reading {}'.format(json))
    data = ut.load_json(json)
    obsids.append(data['XX']['IMAGE_ID'])
    # XX POL
    rms_all_sxx.append(data['XX']['RMS_ALL'])
    rms_box_sxx.append(data['XX']['RMS_BOX'])
    pks_pflux_sxx.append(data['XX']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_sxx.append(data['XX']['PKS0023_026']['INT_FLUX'])
    # YY POL
    rms_all_syy.append(data['YY']['RMS_ALL'])
    rms_box_syy.append(data['YY']['RMS_BOX'])
    pks_pflux_syy.append(data['YY']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_syy.append(data['YY']['PKS0023_026']['INT_FLUX'])

    # V POL
    rms_all_sv.append(data['V']['RMS_ALL'])
    rms_box_sv.append(data['V']['RMS_BOX'])
    pks_pflux_sv.append(data['V']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_sv.append(data['V']['PKS0023_026']['INT_FLUX'])
    count += 1
print('Total number of observations: {}'.format(count))

ratio_rms_all_sxx = np.array(rms_all_sv) / np.array(rms_all_sxx)
ratio_rms_all_syy = np.array(rms_all_sv) / np.array(rms_all_syy)
ratio_rms_box_sxx = np.array(rms_box_sv) / np.array(rms_all_sxx)
ratio_rms_box_syy = np.array(rms_box_sv) / np.array(rms_all_syy)
ratio_pks_pflux_sxx = np.array(pks_pflux_sv) / np.array(pks_pflux_sxx)
ratio_pks_pflux_syy = np.array(pks_pflux_sv) / np.array(pks_pflux_syy)
ratio_pks_tflux_sxx = np.array(pks_tflux_sv) / np.array(pks_tflux_sxx)
ratio_pks_tflux_syy = np.array(pks_tflux_sv) / np.array(pks_tflux_syy)

# getting maximum values
max_rms_all = np.nanmax(np.array([rms_all_sxx, rms_all_syy]))
max_rms_box = np.nanmax(np.array([rms_box_sxx, rms_box_syy]))
max_rms = np.nanmax([max_rms_all, max_rms_box])
max_rms_sv = np.nanmax(np.array([rms_all_sv, rms_box_sv]))
max_pks_pflux = np.nanmax(np.array([pks_pflux_sxx, pks_pflux_syy]))
max_pks_tflux = np.nanmax(np.array([pks_tflux_sxx, pks_tflux_syy]))
max_pks = np.nanmax([max_pks_pflux, max_pks_tflux])
max_pks_sv = np.nanmax(np.array([pks_pflux_sv, pks_tflux_sv]))
max_ratio_rms_all = np.nanmax(np.array([ratio_rms_all_sxx, ratio_rms_all_syy]))
max_ratio_rms_box = np.nanmax(np.array([ratio_rms_box_sxx, ratio_rms_box_syy]))
max_ratio_rms = np.nanmax([max_ratio_rms_all, max_ratio_rms_box])
max_ratio_pks_pflux = np.nanmax(
    np.array([ratio_pks_pflux_sxx, ratio_pks_pflux_syy]))
max_ratio_pks_tflux = np.nanmax(
    np.array([ratio_pks_tflux_sxx, ratio_pks_tflux_syy]))
max_ratio_pks = np.nanmax([max_ratio_pks_pflux, max_ratio_pks_tflux])

# getting the outliers
# saving the file
if args.save:
    figname = 'img_qa' if args.figname is None else args.figname
# plotting rms
fig = pylab.figure(figsize=(12, 7))
fig.suptitle('RMS', size=16)
ax = fig.add_subplot(3, 2, 1)
ax.plot(np.arange(len(obsids)), rms_all_sxx, '.-',
        color='indianred', linewidth=2, label='XX')
ax.plot(np.arange(len(obsids)), rms_all_syy, '.-',
        color='dodgerblue', linewidth=2, label='YY')
ax.set_ylim(-0.5, max_rms + 1)
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('({} x {})'.format(imsize[0], imsize[1]), size=13)
ax = fig.add_subplot(3, 2, 2)
ax.plot(np.arange(len(obsids)), rms_box_sxx, '.-',
        color='indianred', linewidth=2, label='XX')
ax.plot(np.arange(len(obsids)), rms_box_syy, '.-',
        color='dodgerblue', linewidth=2, label='YY')
ax.set_ylim(-0.5, max_rms + 1)
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('({} x {})'.format(bxsize[0], bxsize[1]), size=13)
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 3)
ax.plot(np.arange(len(obsids)), rms_all_sv, '.-',
        color='olivedrab', linewidth=2, label='V')
ax.set_ylim(-0.5, max_rms_sv + 1)
ax.legend(loc='upper right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 4)
ax.plot(np.arange(len(obsids)), rms_box_sv, '.-',
        color='olivedrab', linewidth=2, label='V')
ax.set_ylim(-0.5, max_rms_sv + 1)
ax.legend(loc='upper right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 5)
ax.plot(np.arange(len(obsids)), ratio_rms_box_sxx, '.-', color='indianred',
        linewidth=2, label='V/XX')
ax.plot(np.arange(len(obsids)), ratio_rms_box_syy, '.-', color='dodgerblue',
        linewidth=2, label='V/YY')
ax.fill_between(np.arange(len(obsids)), 0.5,
                max_ratio_rms + 0.2, color='red', alpha=0.2)
ax.fill_between(np.arange(len(obsids)), 0.3,
                0.5, color='orange', alpha=0.2)
ax.set_ylim(0, max_ratio_rms + 0.2)
ax.set_xlim(0, len(obsids))
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
ax = fig.add_subplot(3, 2, 6)
ax.plot(np.arange(len(obsids)), ratio_rms_box_sxx, '.-', color='indianred',
        linewidth=2, label='V/XX')
ax.plot(ratio_rms_box_syy, '.-', color='dodgerblue',
        linewidth=2, label='V/YY')
ax.fill_between(np.arange(len(obsids)), 0.5,
                max_ratio_rms + 0.2, color='red', alpha=0.2)
ax.fill_between(np.arange(len(obsids)), 0.3,
                0.5, color='orange', alpha=0.2)
ax.set_ylim(0, max_ratio_rms + 0.2)
ax.set_xlim(0, len(obsids))
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
if args.save:
    pylab.savefig(figname + '_rms.png', dpi=200)

# plotting PKS
fig = pylab.figure(figsize=(12, 7))
fig.suptitle('FLUX DENSITY', size=16)
ax = fig.add_subplot(3, 2, 1)
ax.plot(pks_pflux_sxx, '*-', color='indianred', linewidth=2, label='XX')
ax.plot(pks_pflux_syy, '*-', color='dodgerblue', linewidth=2, label='YY')
ax.set_ylim(-1, max_pks + 3)
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('PEAK', size=13)
ax = fig.add_subplot(3, 2, 2)
ax.plot(pks_tflux_sxx, '*-', color='indianred', linewidth=2, label='XX')
ax.plot(pks_tflux_syy, '*-', color='dodgerblue', linewidth=2, label='YY')
ax.set_ylim(-1, max_pks + 3)
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('INTEGRATED', size=13)
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 3)
ax.plot(pks_pflux_sv, '*-', color='olivedrab', linewidth=2, label='V')
ax.set_ylim(-1, max_pks_sv + 3)
ax.legend(loc='upper right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 4)
ax.plot(pks_tflux_sv, '*-', color='olivedrab', linewidth=2, label='V')
ax.set_ylim(-1, max_pks_sv + 3)
ax.legend(loc='upper right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 5)
ax.plot(ratio_pks_pflux_sxx, '*-', color='indianred',
        linewidth=2, label='V/XX')
ax.plot(ratio_pks_pflux_syy, '*-', color='dodgerblue',
        linewidth=2, label='V/YY')
#ax.fill_between(np.arange(len(obsids)), 0.5,
#                max_ratio_pks + 0.2, color='red', alpha=0.2)
#ax.fill_between(np.arange(len(obsids)), 0.3,
#                0.5, color='orange', alpha=0.2)
ax.set_ylim(-0.1, max_ratio_pks + 0.2)
ax.set_xlim(0, len(obsids))
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
ax = fig.add_subplot(3, 2, 6)
ax.plot(ratio_pks_tflux_sxx, '*-', color='indianred',
        linewidth=2, label='V/XX')
ax.plot(ratio_pks_tflux_syy, '*-', color='dodgerblue',
        linewidth=2, label='V/YY')
#ax.fill_between(np.arange(len(obsids)), 0.5,
#                max_ratio_pks + 0.2, color='red', alpha=0.2)
#ax.fill_between(np.arange(len(obsids)), 0.3,
#                0.5, color='orange', alpha=0.2)
ax.set_ylim(-0.1, max_ratio_pks + 0.2)
ax.set_xlim(0, len(obsids))
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
if args.save:
    pylab.savefig(figname + '_pks.png', dpi=200)
else:
    pylab.show()
