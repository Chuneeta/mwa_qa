import numpy as np
import pandas as pd
import pylab
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats

# plottig settings
colors = [name for name in mcolors.CSS4_COLORS
          if f'xkcd:{name}' in mcolors.XKCD_COLORS]
colors = colors = [name.split(':')[-1] for name in mcolors.XKCD_COLORS.keys()]

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'


class CSV(object):
    def __init__(self, csvfile, ex_keys=[], obs_key='OBS', obs_threshold=None):
        self.data = self.read_csvfile(csvfile)
        self.data = self.data.drop(ex_keys, axis='columns')
        self.obsids = np.array(self.data[obs_key])
        self.data = self.data.set_index(obs_key)
        self.data.index = self.data.index.astype(int)
        self.variables = np.array(self.data.columns)
        if obs_threshold is None:
            obs_threshold = self.obsids[-1]
        else:
            print('Enter')
            self.data = self.data[self.data.index <= obs_threshold]
            self.obsids = self.obsids[self.obsids <= obs_threshold]
        self.Nobs = len(self.obsids)

    def read_csvfile(self, csvfile):
        return pd.read_csv(csvfile)


class Stats(CSV):
    def __init__(self, csvfile, ex_keys=[], obs_key='OBS', obs_threshold=None):
        CSV.__init__(self, csvfile, ex_keys, obs_key, obs_threshold)

    def correlation_matrix(self, method='pearson'):
        """
        Returns correlation matrix
        method : Methodoly used to determine the correlation between
                 two variables (pearson, spearman, kendall)
        """
        return self.data.corr(method=method)


class Plotting(Stats):
    def __init__(self, csvfile, ex_keys=[], obs_key='OBS', obs_threshold=None):
        Stats.__init__(self, csvfile, ex_keys, obs_key, obs_threshold)

    def plot_1D(self, variables=[], plot_type='log', titlename='',
                xlabel='Observation Number', ylabel='Value', legend=True, save=None, figname=None):
        if len(variables) == 0:
            variables = self.variables
        pylab.figure()
        pylab.title(titlename)
        for i, var in enumerate(variables):
            if plot_type == 'log':
                pylab.semilogy(np.arange(self.Nobs),
                               self.data[var], linewidth=2, label=var)
            elif plot_type == 'linear':
                pylab.plot(np.arange(self.Nobs),
                           self.data[var], linewidth=2, label=var)
            else:
                raise ValueError('plot_type is not recognised')
        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)
        pylab.grid(ls='dotted')
        pylab.tick_params(labelsize=10, direction='in', length=4, width=2)
        if legend:
            pylab.legend(ncol=2, loc='upper right', fontsize=8)
        if save:
            if figname is None:
                figname = '{}_{}_1D.png'.format(
                    self.obsids[0], self.obsids[-1])
            pylab.savefig(figname)
            pylab.close()
        else:
            pylab.show()

    def jointplot(self, variable1, variable2, plot_type='log', kind='scatter', save=None, figname=None):
        if plot_type == 'log':
            plot_data = np.log10(self.data)
        else:
            plot_data = self.data
        sns_plot = sns.JointGrid(data=plot_data, x=variable1, y=variable2)
        sns_plot.plot(sns.scatterplot, sns.distplot)
        # sns_plot.annotate(stats.pearsonr)
        if save:
            if figname is None:
                figname = '{}_{}_jointplot.png'.format(
                    self.obsids[0], self.obsids[-1])
            sns_plot.figure.savefig(figname)
        else:
            pylab.show()

    def pairplot(self, variables=[], kind='scatter', diag_kind='hist', titlename='', save=None, figname=None):
        n = len(self.data.index)
        if len(variables) == 0:
            sns_plot = sns.pairplot(self.data.head(1000), kind=kind, diag_kind=diag_kind, plot_kws={
                                    'line_kws': {'color': 'black'}})
        else:
            sns_plot = sns.pairplot(self.data.head(n), vars=variables,
                                    kind=kind, diag_kind=diag_kind, plot_kws={'line_kws': {'color': 'black'}})
        if save:
            if figname is None:
                figname = '{}_{}_pairplot.png'.format(
                    self.obsids[0], self.obsids[-1])
            sns_plot.figure.savefig(figname)
        else:
            pylab.show()

    def plot_corr_matrix(self, method='pearson', titlename=None, save=None, figname=None, dpi=200):
        corr_matrix = self.correlation_matrix(method=method)
        pylab.figure(figsize=(20, 16))
        heatmap = sns.heatmap(corr_matrix, annot=True,
                              cmap='GnBu', annot_kws={"fontsize": 8})
        if titlename is None:
            titlename = '{} CORR MATRIX'.format(method.upper())
        heatmap.set_title(titlename, fontdict={
                          'fontsize': 15}, pad=12)
        if save:
            if figname is None:
                figname = '{}_{}_{}_corr.png'.format(
                    self.obsids[0], self.obsids[-1], method)
            pylab.savefig(figname, dpi=dpi, bbox_inches='tight')
        else:
            pylab.show()
