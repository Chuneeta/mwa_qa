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

#imsize = args.json[0]['IMSIZE']
imsize = [4096, 4096]
bxsize = ut.load_json(args.json[0])['PIX_BOX']
count = 0
for json in args.json:
    print('Reading {}'.format(json))
    data = ut.load_json(json)
    # XX POL
    rms_all_sxx.append(data['XX']['RMS_ALL'])
    rms_box_sxx.append(data['XX']['RMS_BOX'])
    pks_pflux_sxx.append(data['XX']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_sxx.append(data['XX']['PKS0023_026']['INT_FLUX'])
    # YY POL
    rms_all_syy.append(data['YY']['RMS_ALL'])
    rms_box_syy.append(data['YY']['RMS_BOX'])
    pks_pflux_sxx.append(data['XX']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_sxx.append(data['XX']['PKS0023_026']['INT_FLUX'])

    # V POL
    rms_all_sv.append(data['V']['RMS_ALL'])
    rms_box_sv.append(data['V']['RMS_BOX'])
    pks_pflux_sv.append(data['V']['PKS0023_026']['PEAK_FLUX'])
    pks_tflux_sv.append(data['V']['PKS0023_026']['INT_FLUX'])
    count += 1

ratio_rms_all_sxx = np.array(rms_all_sv) / np.array(rms_all_sxx)
ratio_rms_all_syy = np.array(rms_all_sv) / np.array(rms_all_syy)
ratio_rms_box_sxx = np.array(rms_box_sv) / np.array(rms_all_sxx)
ratio_rms_box_syy = np.array(rms_box_sv) / np.array(rms_all_syy)
ratio_rms_all_sxx = np.array(rms_all_sv) / np.array(rms_all_sxx)
print('Total number of observations: {}'.format(count))

fig = pylab.figure(figsize=(12, 7))
fig.suptitle('RMS', size=16)
ax = fig.add_subplot(3, 2, 1)
ax.plot(rms_all_sxx, '.-', color='indianred', linewidth=2, label='XX')
ax.plot(rms_all_syy, '.-', color='dodgerblue', linewidth=2, label='YY')
ax.legend(loc='lower right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('({} x {})'.format(imsize[0], imsize[1]), size=13)
ax = fig.add_subplot(3, 2, 2)
ax.plot(rms_box_sxx, '.-', color='indianred', linewidth=2, label='XX')
ax.plot(rms_box_syy, '.-', color='dodgerblue', linewidth=2, label='YY')
ax.legend(loc='lower right', ncol=2)
ax.grid(ls='dotted')
ax.set_title('({} x {})'.format(bxsize[0], bxsize[1]), size=13)
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 3)
ax.plot(rms_all_sv, '.-', color='olivedrab', linewidth=2, label='V')
ax.legend(loc='lower right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 4)
ax.plot(rms_box_sv, '.-', color='olivedrab', linewidth=2, label='V')
ax.legend(loc='lower right')
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 2, 5)
ax.plot(ratio_rms_box_sxx, '.-', color='indianred', linewidth=2, label='V/XX')
ax.plot(ratio_rms_box_syy, '.-', color='dodgerblue', linewidth=2, label='V/YY')
ax.legend(loc='lower right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
ax = fig.add_subplot(3, 2, 6)
ax.plot(ratio_rms_box_sxx, '.-', color='indianred', linewidth=2, label='V/XX')
ax.plot(ratio_rms_box_syy, '.-', color='dodgerblue', linewidth=2, label='V/YY')
ax.legend(loc='lower right', ncol=2)
ax.grid(ls='dotted')
ax.set_xlabel('Observation Number', fontsize=12)
"""
pylab.subplot(211)
pylab.plot(rms_all_sxx, '.-', label='Stokes XX')
pylab.plot(rms_all_syy, '.-', label='Stokes YY')
pylab.legend(loc='upper left')
pylab.grid()
pylab.subplot(212)
pylab.plot(rms_all_sv, '.-', label='Stokes V')
pylab.legend(loc='upper left')
pylab.grid()

pylab.figure()
pylab.suptitle('RMS (SELECTED)')
pylab.subplot(211)
pylab.plot(rms_box_sxx, '.-', label='Stokes XX')
pylab.plot(rms_box_syy, '.-', label='Stokes YY')
pylab.grid()
pylab.legend(loc='upper left')
pylab.subplot(212)
pylab.plot(rms_box_sv, '.-', label='Stokes V')
pylab.legend(loc='upper left')
pylab.grid()

pylab.figure()
pylab.suptitle('PKS0023-026')
pylab.subplot(211)
pylab.plot(pks_pflux_sxx, '*-', label='Stokes XX')
pylab.plot(pks_pflux_syy, '*-', label='Stokes YY')
pylab.legend(loc='upper left')
pylab.grid()
pylab.subplot(212)
pylab.plot(pks_pflux_sv, '*-', label='Stokes V')
pylab.legend(loc='upper left')
pylab.grid()
"""
pylab.show()
