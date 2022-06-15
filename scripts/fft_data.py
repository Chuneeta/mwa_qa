from mwa_clysis import get_stats as gs
import numpy as np
import os, sys
import pandas as pd
import argparse
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calfiles', nargs='+', type=str, help='Calfiles containing the calibration solutions in fits format')
parser.add_argument('-m', '--metafits', nargs='+', type=str, help='Metafits containing info on the observation')
args = parser.parse_args()

calfiles = args.calfiles
metafits = args.metafits
# checking if the length of calfiles matches the length of matafits files
if not metafits is None:
	assert len(calfiles) == len(metafits), "Number of metafits must be the same as the number of input calfiles"

for cfl in calfiles:
	dirname = os.path.dirname(cfl)
	cfl_gps = re.findall(r'\d+', cfl.split('/')[-1]) 
	if len(cfl_gps) == 1:
		mfl = '{}/{}.metafits'.format(dirname, cfl_gps[0])
	else:
		raise Exception("The string contains two integers values. Need only one 1")
	print ('Using calfile ... {}'.format(cfl))
	print ('Using metafits ... {}'.format(mfl))
	stats = gs.Stats(calfile = cfl, metafits = mfl)
	fft_data = stats.get_fft_data()
	dirname = os.path.dirname(cfl)
	pickle_file = '{}/fft_dict_{}.pkl'.format(dirname, cfl_gps[0])
	print ('Saving fft data to ... {}'.format(pickle_file))
	pickle.dump(fft_data, open(pickle_file, 'wb'))
