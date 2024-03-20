
from argparse import ArgumentParser
from mycolorpy import colorlist as mcp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab

parser = ArgumentParser(
    description="EValuating cutoff thresholds for the different metrics")
parser.add_argument('csvfile', type=str, help='QA metric csv file')
parser.add_argument('--sigma', dest='sigma', type=float,
                    default=3, help='Sigma for cutoff threshold. Default is 3.')
parser.add_argument('-o', '--outfile', dest='outfile', help='thresholding cutoffs', type=str,
                    default=None, required=False)
parser.add_argument('--plot', action='store_true',
                    help='Plots the distribution of the metrics', default=None)
parser.add_argument('--save', action='store_true',
                    help='Saves the plots', default=None)
args = parser.parse_args()


df = pd.read_csv(args.csvfile, sep=',')
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)
df.dropna(axis=0, how='any', subset=["OBSID"], inplace=True)
df.dropna(subset=["EW POINT"], inplace=True)
ew_pointings = df['EW POINT']

for sub in ['NOSUB', 'SUB', 'IONOSUB']:
    if {f"NEW P_WEDGE {sub}", f"NEW P_WINDOW {sub}"}.issubset(set(df.columns)):
        df[f'P_WIN_WEDGE RATIO {sub}'] = df[f"NEW P_WINDOW {sub}"] / \
            df[f"NEW P_WEDGE {sub}"]
    if {f"NEW XX PKS0023_026 INT {sub}", f"NEW YY PKS0023_026 INT {sub}"}.issubset(set(df.columns)):
        df[f'PKS INT DIFF XXYY {sub}'] = np.abs(
            df[f'NEW XX PKS0023_026 INT {sub}'] - df[f'NEW YY PKS0023_026 INT {sub}'])
        if {f"NEW V PKS0023_026 INT {sub}"}.issubset(set(df.columns)):
            df[f'PKS INT RATIO VXXYY {sub}'] = df[f'NEW V PKS0023_026 INT {sub}'] / (
                df[f'NEW XX PKS0023_026 INT {sub}'] + df[f'NEW XX PKS0023_026 INT {sub}'])

for sub in ['IONOSUB']:
    for pow in ['WEDGE', 'WINDOW']:
        if {f"NEW P_{pow} {sub}", f"NEW P_{pow} NOSUB"}.issubset(set(df.columns)):
            df[f'P_{pow} RATIO {sub}'] = df[f'NEW P_{pow} {sub}'] / \
                df[f'NEW P_{pow} NOSUB']
    for pol in ['XX', 'YY', 'V']:
        if {f"NEW {pol} PKS0023_026 INT {sub}", f"NEW {pol} PKS0023_026 INT NOSUB"}.issubset(set(df.columns)):
            df[f'NEW {pol} PKS0023_026 INT {sub} RATIO'] = df[f'NEW {pol} PKS0023_026 INT {sub}'] / \
                df[f'NEW {pol} PKS0023_026 INT NOSUB']

if {'NEW P_WEDGE IONOSUB', 'NEW P_WEDGE SUB'}.issubset(set(df.columns)):
    df['P_WEDGE RATIO IONOSUB SUB'] = df['NEW P_WEDGE IONOSUB'] / \
        df['NEW P_WEDGE SUB']
if {'NEW P_WINDOW IONOSUB', 'NEW P_WINDOW SUB'}.issubset(set(df.columns)):
    df['P_WINDOW RATIO IONOSUB SUB'] = df['NEW P_WINDOW IONOSUB'] / \
        df['NEW P_WINDOW SUB']

# print('\n'.join(df.columns))

# POWER SPECTRUM METRICS
# splitting by pointings
un_ew_pointings = np.unique(ew_pointings)
outdata = []

for i, fl in enumerate(['NEW P_WINDOW NOSUB',
                        "P_WIN_WEDGE RATIO NOSUB",
                        "P_WINDOW RATIO IONOSUB",
                        "P_WEDGE RATIO IONOSUB"]):
    outdict = {}
    outdict['Metric'] = fl
    print(fl)
    for pt in np.unique(ew_pointings):
        inds = np.where(ew_pointings == pt)
        data = np.array(df[fl])[inds]
        mean = np.nanmean(data)
        std = np.nanstd(data)
        outdict[pt] = tuple((mean - args.sigma *
                             std, mean + args.sigma*std))

    outdata.append(outdict)

for i, fl in enumerate(["NEW V RMS BOX NOSUB",
                        "PKS INT RATIO VXXYY NOSUB",
                        "PKS INT DIFF XXYY IONOSUB",
                        "NEW XX PKS0023_026 INT IONOSUB RATIO",
                        "NEW YY PKS0023_026 INT IONOSUB RATIO"]):
    outdict = {}
    outdict['Metric'] = fl
    for pt in np.unique(ew_pointings):
        inds = np.where(ew_pointings == pt)
        data = np.array(df[fl])[inds]
        data = data[~np.isnan(data)]
        if np.all(data):
            outdict[pt] = (np.nan, np.nan)
        else:
            Q1 = np.percentile(data, 25, interpolation='midpoint')
            Q2 = np.percentile(data, 50, interpolation='midpoint')
            Q3 = np.percentile(data, 75, interpolation='midpoint')
            IQR = Q3 - Q1
            outdict[pt] = tuple((Q1 - 1.5 * IQR, Q1 + 1.5 * IQR))
    outdata.append(outdict)

if args.outfile is None:
    outfile = args.csvfile.replace('.csv', '_thresholds.csv')
else:
    outfile = args.outfile
df_out = pd.DataFrame(data=outdata)
df_out.to_csv(outfile, index=False)


def plot_pspecqa():
    colors = mcp.gen_color(cmap='rainbow', n=len(un_ew_pointings))
    labels = [r'$P_{win} (unsub)$',
              r'$\frac{P_{win}}{P_{wed}} (sub)$',
              r'$P_{win} (\frac{sub}{unsub})$',
              r'$P_{wed}(\frac{sub}{unsub})$'
              ]

    fig = pylab.figure(figsize=(12, 8))
    pylab.subplot(2, 2, 1)
    sns.kdeplot(df, x='NEW P_WINDOW NOSUB', hue='EW POINT',
                palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[0], size=17)
    # pylab.xlim(-20, 30)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][0][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][0][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 2, 2)
    sns.kdeplot(df, x='P_WIN_WEDGE RATIO NOSUB',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.title(labels[1], size=17)
    pylab.ylabel('Density', fontsize=16)
    pylab.xlim(-0.001, 0.07)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][1][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][1][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(2, 2, 3)
    sns.kdeplot(df, x='P_WINDOW RATIO IONOSUB',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[2], size=17)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][2][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][2][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)
    pylab.xlim(0.1, 1.2)

    pylab.subplot(2, 2, 4)
    sns.kdeplot(df, x='P_WEDGE RATIO IONOSUB',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[3], size=17)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][3][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][3][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplots_adjust(hspace=0.3)

    if args.save:
        figure_name = args.csvfile.replace('.csv', '_pspecqa_dist.png')
        pylab.savefig(figure_name)
    else:
        pylab.show()


def plot_imgqa():
    colors = mcp.gen_color(cmap='rainbow', n=len(un_ew_pointings))
    labels = [r'$V_{rms}\, (unsub)$',
              r'$\frac{S_{V}}{(S_{EW} + S_{NS})}$',
              r'Diff $(S_{EW} , S_{NS})$',
              r'$S_{EW}(\frac{sub}{unsub})$',
              r'$S_{NS}(\frac{sub}{unsub})$'
              ]

    fig = pylab.figure(figsize=(12, 9))
    pylab.subplot(3, 2, 1)
    sns.kdeplot(df, x='NEW V RMS BOX NOSUB', hue='EW POINT',
                palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[0], size=17)
    # pylab.xlim(-20, 30)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][4][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][4][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(3, 2, 2)
    sns.kdeplot(df, x='PKS INT RATIO VXXYY NOSUB',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.title(labels[1], size=17)
    pylab.ylabel('Density', fontsize=16)
    pylab.xlim(-0.001, 0.07)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][5][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][5][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(3, 2, 3)
    sns.kdeplot(df, x='PKS INT DIFF XXYY IONOSUB',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[2], size=17)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][6][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][6][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)
    pylab.xlim(0.1, 1.2)

    pylab.subplot(3, 2, 4)
    sns.kdeplot(df, x='NEW XX PKS0023_026 INT IONOSUB RATIO',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[3], size=17)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][7][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][7][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplot(3, 2, 5)
    sns.kdeplot(df, x='NEW YY PKS0023_026 INT IONOSUB RATIO',
                hue='EW POINT', palette='rainbow', common_norm=True)
    pylab.xlabel('')
    pylab.ylabel('Density', fontsize=16)
    pylab.title(labels[3], size=17)
    for i, pt in enumerate(un_ew_pointings):
        pylab.axvline(x=df_out[pt][8][0], ls='dashed', color=colors[i])
        pylab.axvline(x=df_out[pt][8][1], ls='dashed', color=colors[i])
    pylab.tick_params(labelsize=14, direction='out', length=3, width=1)

    pylab.subplots_adjust(hspace=0.4, wspace=0.2)

    if args.save:
        figure_name = args.csvfile.replace('.csv', '_imgqa_dist.png')
        pylab.savefig(figure_name)
    else:
        pylab.show()


if args.plot:
    plot_pspecqa()
    plot_imgqa()
