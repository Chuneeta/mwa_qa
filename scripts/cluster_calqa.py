from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
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
for json in args.json:
    print(count, ' Reading {}'.format(json))
    data = ut.load_json(json)
    obsids.append(data['OBSID'])
    unused_bls.append(data['UNUSED_BLS'])
    unused_ants.append(data['UNUSED_ANTS'])
    unused_chs.append(data['UNUSED_CHS'])
    non_converged_chs.append(data['NON_CONVERGED_CHS'])
    conv_array = np.array(data['CONVERGENCE'])
    conv_array[0, np.where(
        conv_array[0] == -1.7976931348623157e+308)[0]] = np.nan
    convergence.append(conv_array)
    convergence_var.append(data['CONVERGENCE_VAR'])
    rms_ampvar_freq_xx.append(data['XX']['RMS_AMPVAR_FREQ'])
    rms_ampvar_ant_xx.append(data['XX']['RMS_AMPVAR_ANT'])
    receiver_chisqvar_xx.append(data['XX']['RECEIVER_CHISQVAR'])
    dfft_power_xx.append(data['XX']['DFFT_POWER'])
    dfft_power_xx_pkpl.append(data['XX']['DFFT_POWER_HIGH_PKPL'])
    dfft_power_xx_nkpl.append(data['XX']['DFFT_POWER_HIGH_NKPL'])
    rms_ampvar_freq_yy.append(data['YY']['RMS_AMPVAR_FREQ'])
    rms_ampvar_ant_yy.append(data['YY']['RMS_AMPVAR_ANT'])
    receiver_chisqvar_yy.append(data['YY']['RECEIVER_CHISQVAR'])
    dfft_power_yy.append(data['YY']['DFFT_POWER'])
    dfft_power_yy_pkpl.append(data['YY']['DFFT_POWER_HIGH_PKPL'])
    dfft_power_yy_nkpl.append(data['YY']['DFFT_POWER_HIGH_NKPL'])
    count += 1

# clustering algorithms
# rms_ampvar_freq vs rms_ampvar_ant
rms_xx = np.zeros((len(rms_ampvar_freq_xx), 3))
rms_xx[:, 0] = np.log10(rms_ampvar_freq_xx)
rms_xx[:, 1] = np.log10(rms_ampvar_ant_xx)
#rms_xx[:, 1] = np.log10(convergence_var)
# Computing epsilon
# creating an object of the NearestNeighbors class
neighb = NearestNeighbors(n_neighbors=2)
nbrs = neighb.fit(rms_xx)  # fitting the data to the object
# Sort and plot the distances results
distances, indices = nbrs.kneighbors(rms_xx)
distances = np.sort(distances, axis=0)  # sorting the distances
distances = distances[:, 1]  # taking the second column of the sorted distances
kneedle = KneeLocator(np.arange(len(distances)), distances,
                      S=1.0, curve='convex', direction='increasing')
epsilon = distances[round(kneedle.knee, 3)]
minpoints = 2 * rms_xx.shape[1]
# cluster the data into five clusters
# fitting the model
#dbscan = DBSCAN(eps=epsilon, min_samples=minpoints).fit(rms_xx)
dbscan = DBSCAN(eps=epsilon, min_samples=minpoints).fit_predict(rms_xx)
# labels = dbscan.labels_  # getting the labels
dbscan_clusters = np.unique(dbscan)
# print(np.array(obsids)[inds])
# Plot the clusters
for cluster in dbscan_clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(dbscan == cluster)
    # create scatter of these samples
    pylab.scatter(rms_xx[row_ix, 0], rms_xx[row_ix, 1],
                  label='Cluster {}'.format(cluster))
# show the plot
pylab.legend()
pylab.xlabel('RMS(ant)')
pylab.ylabel('Convergence')
pylab.show()
# pylab.scatter(rms_xx[:, 0], rms_xx[:, 1], c=labels,
#              marker='o', cmap='jet', alpha=0.6)
# pylab.xlabel('RMS(freq')
# pylab.ylabel('Convergence')
# pylab.legend()
# pylab.show()
