#!/usr/bin/env python
from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from pathlib import Path
import matplotlib as mpl
import numpy as np
import pylab
import pandas as pd

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')
parser.add_argument('--save', dest='save', action='store_true',
                    default=None, help='Boolean to allow to save the image')
parser.add_argument('--out', dest='figname', default=None,
                    help='Name of ouput figure name')
args = parser.parse_args()

obsids = []
unused_bls = []
unused_ants = []
unused_chs = []
non_converged_chs = []
convergence = []
convergence_var = []
# XX POL
rms_ampvar_freq_xx = []
rms_ampvar_ant_xx = []
receiver_chisqvar_xx = []
dfft_power_xx = []
dfft_power_xx_pkpl = []
dfft_power_xx_nkpl = []

# YY POL
rms_ampvar_freq_yy = []
rms_ampvar_ant_yy = []
receiver_chisqvar_yy = []
dfft_power_yy = []
dfft_power_yy_pkpl = []
dfft_power_yy_nkpl = []

count = 0
keys = ['OBSID', 'UNUSED_BLS', 'UNUSED_ANTS', 'UNUSED_CHS', 'NON_CONVERGED_CHS', 'CONVERGENCE_VAR',
        'RMS_AMPVAR_FREQ', 'RMS_AMPVAR_ANT', 'DFFT_POWER', 'DFFT_POWER_HIGH_PKPL',
        'DFFT_POWER_HIGH_NKPL', 'RECEIVER_CHISQVAR']
for json in args.json:
    print(count, ' Reading {}'.format(json))
    data = ut.load_json(json)
    obsids.append(data.get('OBSID', np.nan))
    # unused_bls.append(data.get('PERCENT_UNUSED_BLS', np.nan))
    # unused_ants.append(data.get('PERCENT_BAD_ANTS', np.nan))
    # unused_chs.append(data.get('PERCENT_NONCONVERGED_CHS', np.nan))
    # non_converged_chs.append(data.get('PERCENT_NONCONVERGED_CHS', np.nan))
    conv_array = np.array(data.get('CONVERGENCE', [[np.nan]]))
    conv_array[0, np.where(
        conv_array[0] < 0)[0]] = np.nan
    convergence.append(conv_array)
    convergence_var.append(data.get('CONVERGENCE_VAR', np.nan))
    # rms_ampvar_freq_xx.append(data['XX']['RMS_AMPVAR_FREQ'])
    # rms_ampvar_ant_xx.append(data['XX']['RMS_AMPVAR_ANT'])
    # receiver_chisqvar_xx.append(data['XX']['RECEIVER_CHISQVAR'])
    dfft_power_xx.append(data.get('XX', {}).get('DFFT_POWER', np.nan))
    # dfft_power_xx_pkpl.append(data['XX']['DFFT_POWER_HIGH_PKPL'])
    # dfft_power_xx_nkpl.append(data['XX']['DFFT_POWER_HIGH_NKPL'])
    # rms_ampvar_freq_yy.append(data['YY']['RMS_AMPVAR_FREQ'])
    # rms_ampvar_ant_yy.append(data['YY']['RMS_AMPVAR_ANT'])
    # receiver_chisqvar_yy.append(data['YY']['RECEIVER_CHISQVAR'])
    dfft_power_yy.append(data.get('YY', {}).get('DFFT_POWER', np.nan))
    # dfft_power_yy_pkpl.append(data['YY']['DFFT_POWER_HIGH_PKPL'])
    # dfft_power_yy_nkpl.append(data['YY']['DFFT_POWER_HIGH_NKPL'])
    count += 1

inds = np.where(np.sqrt(np.array(convergence_var)) > 1e-4)
corrupted_obsids = np.array(obsids)[inds[0]]
# extracting bad obsids
# plotting
# saving the file
if args.save:
    figname = 'cal_qa' if args.figname is None else args.figname

fig = pylab.figure(figsize=(8, 6))
pylab.title('CONVERGENCE', size=13)
pylab.imshow(np.log10(np.array(convergence)[:, 0, :]), aspect='auto',
             cmap='YlGn', vmin=-8, vmax=-5, extent=(0, 768, len(obsids), 0),
             interpolation='nearest')
pylab.xlabel('Frequency channels')
pylab.ylabel('Observation Number')
pylab.colorbar()
if args.save:
    pylab.savefig(figname + '_convergence.png', dpi=200)

exit(0)

fig = pylab.figure(figsize=(8, 6))
fig.suptitle('UNUSED DATA', size=16)
ax = fig.add_subplot(2, 1, 1)
ax.plot(np.arange(len(obsids)), unused_ants, '.-',
        color='sienna', linewidth=2, label='ANTS')
ax.plot(np.arange(len(obsids)), unused_bls, '.-',
        color='darkolivegreen', linewidth=2, label='BLS')
ax.fill_between(np.arange(len(obsids)), 60,
                100, color='red', alpha=0.2)
# ax.set_ylim(-0.5, max_rms + 1)
ax.legend(loc='upper right', ncol=2)
ax.grid(ls='dotted')
ax = fig.add_subplot(2, 1, 2)
ax.plot(np.arange(len(obsids)), unused_chs, '.-',
        color='sienna', linewidth=2, label='CHS')
ax.plot(np.arange(len(obsids)), non_converged_chs, '.-',
        color='darkolivegreen', linewidth=2, label='NON-CONVERGED CHS')
ax.fill_between(np.arange(len(obsids)), 60,
                100, color='red', alpha=0.2)
ax.legend(loc='upper right', ncol=2)
ax.set_xlabel('Observation Number', fontsize=12)
ax.grid(ls='dotted')
if args.save:
    pylab.savefig(figname + '_unused.png', dpi=200)

fig = pylab.figure(figsize=(8, 6))
ax = fig.add_subplot(3, 1, 1)
ax.semilogy(np.arange(len(obsids)), rms_ampvar_freq_xx, '.-',
            color='indianred', linewidth=2, label='XX')
ax.semilogy(np.arange(len(obsids)), rms_ampvar_freq_yy, '.-',
            color='dodgerblue', linewidth=2, label='YY')
# ax.set_ylim(-0.5, max_rms + 1)
ax.legend(loc='upper left', ncol=2)
ax.grid(ls='dotted')
ax.set_ylabel('RMS (freq)', fontsize=9)
ax = fig.add_subplot(3, 1, 2)
ax.semilogy(np.arange(len(obsids)), rms_ampvar_ant_xx, '.-',
            color='indianred', linewidth=2, label='XX')
ax.semilogy(np.arange(len(obsids)), rms_ampvar_ant_yy, '.-',
            color='dodgerblue', linewidth=2, label='YY')
ax.legend(loc='upper left', ncol=2)
ax.set_ylabel('RMS (ant)', fontsize=9)
ax.grid(ls='dotted')
ax = fig.add_subplot(3, 1, 3)
ax.semilogy(np.arange(len(obsids)), convergence_var, '.-',
            color='darkolivegreen', linewidth=2)
ax.fill_between(np.arange(len(obsids)), 10**2,
                10 ** -8, color='red', alpha=0.2)
ax.set_ylabel('CONV VAR', fontsize=9)
ax.set_xlabel('Observation Number', fontsize=12)
ax.grid(ls='dotted')
fig.subplots_adjust(hspace=0.3)
if args.save:
    pylab.savefig(figname + '_rms.png', dpi=200)


fig = pylab.figure(figsize=(8, 6))
ax = fig.add_subplot(2, 1, 1)
ax.semilogy(np.arange(len(obsids)), dfft_power_xx, '.-',
            color='coral', linewidth=2, label='Overall')
ax.semilogy(np.arange(len(obsids)), dfft_power_xx_pkpl, '.-',
            color='violet', linewidth=2, label='> 2000 ns')
ax.semilogy(np.arange(len(obsids)), dfft_power_xx_nkpl, '.-',
            color='greenyellow', linewidth=2, label='< -2000 ns')
ax.legend(loc='upper right', ncol=2)
ax.set_ylabel('XX', fontsize=12)
ax.grid(ls='dotted')
ax.legend(loc='upper left', ncol=1)
ax = fig.add_subplot(2, 1, 2)
ax.semilogy(np.arange(len(obsids)), dfft_power_yy, '.-',
            color='coral', linewidth=2, label='Overall')
ax.semilogy(np.arange(len(obsids)), dfft_power_yy_pkpl, '.-',
            color='violet', linewidth=2, label='> 2000 ns')
ax.semilogy(np.arange(len(obsids)), dfft_power_yy_nkpl, '.-',
            color='greenyellow', linewidth=2, label='< -2000 ns')
ax.legend(loc='upper left', ncol=2)
ax.set_xlabel('Observation Number', fontsize=12)
ax.set_ylabel('YY', fontsize=12)
ax.legend(loc='upper left', ncol=1)
ax.grid(ls='dotted')
if args.save:
    pylab.savefig(figname + '_fft.png', dpi=200)
else:
    pylab.show()
