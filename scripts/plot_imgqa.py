from argparse import ArgumentParser
from mwa_qa import json_utils as ut
from pathlib import Path
import pylab

parser = ArgumentParser(
    description="Plotting image metrics")
parser.add_argument('json', type=Path, nargs='+',
                    help='MWA metrics json files')

args = parser.parse_args()
# Stokes XX
rms_all_sxx = []
rms_box_sxx = []
pks_sxx = []

# Stokes XX
rms_all_syy = []
rms_box_syy = []
pks_syy = []

# Stokes XX
rms_all_sv = []
rms_box_sv = []
pks_sv = []

count = 0
for json in args.json:
    print(count, json)
    data = ut.load_json(json)
    # XX POL
    rms_all_sxx.append(data['XX']['RMS_ALL'])
    rms_box_sxx.append(data['XX']['RMS_BOX'])
    pks_sxx.append(data['XX']['PKS0023_026'])

    # YY POL
    rms_all_syy.append(data['YY']['RMS_ALL'])
    rms_box_syy.append(data['YY']['RMS_BOX'])
    pks_syy.append(data['YY']['PKS0023_026'])

    # V POL
    rms_all_sv.append(data['V']['RMS_ALL'])
    rms_box_sv.append(data['V']['RMS_BOX'])
    pks_sv.append(data['V']['PKS0023_026'])
    count += 1

pylab.figure()
pylab.suptitle('RMS (ALL PIXELS)')
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
pylab.plot(pks_sxx, '*-', label='Stokes XX')
pylab.plot(pks_syy, '*-', label='Stokes yy')
pylab.legend(loc='upper left')
pylab.grid()
pylab.subplot(212)
pylab.plot(pks_sv, '*-', label='Stokes V')
pylab.legend(loc='upper left')
pylab.grid()

pylab.show()
