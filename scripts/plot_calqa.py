#!/usr/bin/env python
from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from pathlib import Path
import pylab

parser = ArgumentParser(
    description="Plotting antenna positions")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')

args = parser.parse_args()
data0 = ut.load_json(args.json[0])
start_freq = data0['FREQ_START']
freq_width = data0['FREQ_WIDTH']
nchan = data0['NFREQ']
obsids = []
flagged_bls = []
flagged_ants = []
flagged_chs = []
non_converged_chs = []
convergence_var = []

# XX POL
rms_ampvar_freq_xx = []
rms_ampvar_ant_xx = []
receiver_chisqvar_xx = []
dfft_power_xx = []

# YY POL
rms_ampvar_freq_yy = []
rms_ampvar_ant_yy = []
receiver_chisqvar_yy = []
dfft_power_yy = []

for json in args.json:
    data = ut.load_json(json)
    obsids.append(data['OBSID'])
    flagged_bls.append(data['FLAGGED_BLS'])
    flagged_ants.append(data['FLAGGED_ANTS'])
    flagged_chs.append(data['FLAGGED_CHS'])
    non_converged_chs.append(data['NON_CONVERGED_CHS'])
    convergence_var.append(data['CONVERGENCE_VAR'])
    rms_ampvar_freq_xx.append(data['XX']['RMS_AMPVAR_FREQ'])
    rms_ampvar_ant_xx.append(data['XX']['RMS_AMPVAR_ANT'])
    receiver_chisqvar_xx.append(data['XX']['RECEIVER_CHISQVAR'])
    dfft_power_xx.append(data['XX']['DFFT_POWER'])
    rms_ampvar_freq_yy.append(data['YY']['RMS_AMPVAR_FREQ'])
    rms_ampvar_ant_yy.append(data['YY']['RMS_AMPVAR_ANT'])
    receiver_chisqvar_yy.append(data['YY']['RECEIVER_CHISQVAR'])
    dfft_power_yy.append(data['YY']['DFFT_POWER'])

# plotting
pylab.figure(figsize=(8, 6))
pylab.plot(flagged_ants, 'o',  label='ANTS')
pylab.plot(flagged_bls, 'o', label='BLS')
pylab.xlabel('Obsid')
pylab.ylabel(r'$\%$ flagged')
pylab.title('FLAG METRICS')
pylab.grid()
#pylab.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
pylab.legend()

pylab.figure(figsize=(8, 6))
pylab.plot(flagged_chs, 'o', label='Flagged chs')
pylab.plot(non_converged_chs, 'o', label='Non-converged chs')
pylab.xlabel('Obsid')
pylab.ylabel(r'$\%$')
pylab.title('FLAG METRICS')
#pylab.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
pylab.grid()
pylab.legend()

pylab.figure(figsize=(8, 6))
pylab.plot(convergence_var, '*-')
pylab.xlabel('Obsid')
pylab.ylabel('Variance')
pylab.title('CONVERGENCE')
pylab.grid()
#pylab.tick_params(axis='x', rotation=45, labelsize=6, direction='in')

pylab.figure(figsize=(8, 6))
pylab.suptitle('AMPLITUDE VARIANCE')
ax1 = pylab.subplot(211)
ax1.set_title('Across frequency')
ax1.plot(rms_ampvar_freq_xx, '.-', label='XX')
ax1.plot(rms_ampvar_freq_yy, '.-', label='YY')
ax1.set_ylabel('Rms')
ax1.set_xticks([])
ax1.legend()
ax1.grid()
ax2 = pylab.subplot(212)
ax2.set_title('Across antenna')
ax2.plot(rms_ampvar_ant_xx, '.-', label='XX')
ax2.plot(rms_ampvar_ant_yy, '.-', label='YY')
ax2.set_xlabel('Obsid')
ax2.set_ylabel('Rms')
ax2.grid()
#ax2.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
pylab.subplots_adjust(wspace=0.1)

pylab.figure(figsize=(8, 6))
pylab.plot(receiver_chisqvar_xx, '.-', label='XX')
pylab.plot(receiver_chisqvar_yy, '.-', label='YY')
pylab.xlabel('Obsid')
pylab.ylabel('Var')
pylab.title('RECEIVER CHISQ')
pylab.grid()
#pylab.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
pylab.legend()

pylab.figure(figsize=(8, 6))
pylab.plot(dfft_power_xx, '.-', label='XX')
pylab.plot(dfft_power_yy, '.-', label='YY')
pylab.xlabel('Obsid')
pylab.ylabel('Power')
pylab.title('FFT SEPECTRUM')
pylab.grid()
#pylab.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
pylab.legend()

pylab.show()
