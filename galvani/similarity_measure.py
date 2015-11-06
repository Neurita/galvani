# coding=utf-8
#-------------------------------------------------------------------------------
#Author: Alexandre Manhães Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#Use this at your own risk!
#-------------------------------------------------------------------------------

from   collections      import  OrderedDict
import numpy            as      np
import nitime.analysis  as      nta
import nitime.utils     as      tsu

#TODO: Give the option here to use full matrices, instead of just ts_sets

#See Ledoit-Wolf covariance estimation and Graph-Lasso
#http://www.sciencedirect.com/science/article/pii/S1053811910011602
#http://scikit-learn.org/stable/modules/covariance.html
#http://nilearn.github.io/developers/group_sparse_covariance.html



def concatenate_timeseries(ts1, ts2):
    """Returns a concatenation of both compatible timeseries.

    Parameters
    ----------
    ts1: nitime.TimeSeries: n_samps1 x time_size
        Time series

    ts2: nitime.TimeSeries: n_samps2 x time_size
        Time series

    Returns
    -------
    ts: nitime.TimeSeries: (n_samps1 + n_samps2) x time_size
    """
    if ts1.sampling_interval != ts2.sampling_interval:
        raise ValueError('`ts_set1` and `ts_set2` must have the same sampling_interval, '
                         'got {} and {}.'.format(ts1.sampling_interval, ts2.sampling_interval))

    if ts1.data.ndim == 1:
        ts_data = np.concatenate([[ts1.data, ts2.data]])
    else:
        ts_data = np.concatenate([ts1.data, ts2.data])

    return create_ts_with_data(ts_data, ts1)


def create_ts_with_data(data, ts):
    nuts      = ts.copy()
    nuts.data = data.copy()
    return nuts


def percent_change(ts, ax=-1):
    """Returns the % signal change of each point of the times series
    along a given axis of the array time_series

    Parameters
    ----------

    ts : ndarray
        an array of time series

    ax : int, optional (default to -1)
        the axis of time_series along which to compute means and stdevs

    Returns
    -------

    ndarray
        the renormalized time series array (in units of %)

    Examples
    --------

    >>> ts = np.arange(4*5).reshape(4,5)
    >>> ax = 0
    >>> percent_change(ts,ax)
    array([[-100.    ,  -88.2353,  -78.9474,  -71.4286,  -65.2174],
           [ -33.3333,  -29.4118,  -26.3158,  -23.8095,  -21.7391],
           [  33.3333,   29.4118,   26.3158,   23.8095,   21.7391],
           [ 100.    ,   88.2353,   78.9474,   71.4286,   65.2174]])
    >>> ax = 1
    >>> percent_change(ts,ax)
    array([[-100.    ,  -50.    ,    0.    ,   50.    ,  100.    ],
           [ -28.5714,  -14.2857,    0.    ,   14.2857,   28.5714],
           [ -16.6667,   -8.3333,    0.    ,    8.3333,   16.6667],
           [ -11.7647,   -5.8824,    0.    ,    5.8824,   11.7647]])
    """
    ts      = np.asarray(ts)
    ts_mean = np.mean(ts, ax)
    if ts_mean != 0:
        return (ts / np.expand_dims(ts_mean, ax) - 1) * 100
    else:
        return ts


class TimeSeriesGroupMeasure(object):
    """A strategy class to use any of the time series group measures methods given as a callable objet.

    Parameters
    ----------
    algorithm: callable Measure
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.measure   = None

    def fit_transform(self, ts_set1, ts_set2, **kwargs):
        """Return the group measure value between both sets of timeseries.

        Parameters
        ----------
        ts_set1: nitime.TimeSeries
            n_timeseries x ts_size

        ts_set2: nitime.TimeSeries
            n_timeseries x ts_size

        Returns
        -------
            The result of applying self.algorithm to ts_set1 and ts_set2.

        Notes
        -----
            Both timeseries sets must have the same sampling_interval
        """
        if ts_set1.sampling_interval != ts_set2.sampling_interval:
            raise ValueError('`ts_set1` and `ts_set2` must have the same sampling_interval, '
                             'got {} and {}.'.format(ts_set1.sampling_interval, ts_set2.sampling_interval))

        self.measure = self.algorithm(ts_set1, ts_set2, **kwargs)

        return self.measure()


class Measure(object):
    """ A generic `n` vs `n` vectors similarity measure. Callable class"""

    def __init__(self, ts_set1, ts_set2, **kwargs):
        self.ts1 = ts_set1
        self.ts2 = ts_set2

    def __call__(self):
        raise NotImplementedError


class NiCorrelationMeasure(Measure):
    """Return a the Pearson's Correlation value between all time series in both sets.
    Calculates the correlation using nitime.

    It's a redundant implementation, I haven't compared them yet

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    Returns
    -------
    float
        Scalar correlation value
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(NiCorrelationMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        ts   = concatenate_timeseries(self.ts1, self.ts2)
        corr = nta.CorrelationAnalyzer(ts)
        return corr.corrcoef[0, 1]


class NiCoherenceMeasure(object):
    """Return a the Spectral Coherence value between all time series in both sets.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    lb: float
        Frequency lower bound

    ub: float
        Frequency upper bound

    Returns
    -------
    float
        Scalar coherence value
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(NiCoherenceMeasure, self).__init__(ts_set1, ts_set2, **kwargs)
        self.lb = kwargs.get('lb', 0.02)
        self.ub = kwargs.get('ub', 0.15)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        ts  = concatenate_timeseries(self.ts1, self.ts2)
        coh = nta.CoherenceAnalyzer(ts)

        freq_idx_coh = np.where((coh.frequencies > self.lb) * (coh.frequencies < self.ub))[0]

        mean_coh = np.mean(coh.coherence[:, :, freq_idx_coh], axis=-1)  # Averaging on the last dimension
        return np.mean(mean_coh) #average all of it


class NiGrangerCausalityMeasure(object):
    """Returns a the Granger Causality value all time series in both sets.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    lb: float
        Frequency lower bound

    ub: float
        Frequency upper bound

    Returns
    -------
    float
        Scalar Granger causality value
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(NiGrangerCausalityMeasure, self).__init__(ts_set1, ts_set2, **kwargs)
        self.lb = kwargs.get('lb', 0.02)
        self.ub = kwargs.get('ub', 0.15)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        pts1 = create_ts_with_data(percent_change(self.ts1.data), self.ts1)
        pts2 = create_ts_with_data(percent_change(self.ts2.data), self.ts2)
        pts  = concatenate_timeseries(pts1, pts2)

        #cts  = concatenate_timeseries(ts_set1, ts_set2)
        #pts  = tsu.percent_change    (cts.data)
        #pts  = create_ts_with_data   (pts, ts_set1)
        gc  = nta.GrangerAnalyzer    (pts, order=1)

        freq_idx_gc = np.where((gc.frequencies > self.lb) * (gc.frequencies < self.ub))[0]

        mean_gc = np.mean(gc.causality_xy[:, :, freq_idx_gc], -1)
        return np.nanmean(mean_gc)


class CorrelationMeasure(object):
    """Return a matrix with pearson correlation between pairs all time series in both sets.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    Returns
    -------
    numpy.ndarray (n_samps1 x n_samps2)
        correlation values
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(CorrelationMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        from scipy.stats import pearsonr

        n1 = self.ts1.shape[0]
        n2 = self.ts2.shape[0]
        if (self.ts1.data.ndim > 1 or self.ts2.data.ndim > 1) and n1 == n2 > 1:
            mp = np.array(n1, n2)
            for i1 in list(range(n1)):
                t1 = self.ts1[i1, :]
                for i2 in list(range(n2)):
                    t2 = self.ts2[i2, :]
                    mp[i1, i2] = pearsonr(t1, t2)

            return mp

        else:
            return pearsonr(self.ts1.data.flatten(), self.ts2.data.flatten())[0]



class OrdinaryLeastSquares(object):
    """Return a matrix with pearson correlation between pairs all time series in both sets.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    ols_param: str
        Name of the parameter calculated from the OLS, e.g., 'rsquared', 'rsquared_adj'.
        See statsmodels.api.OLS for valid choices.

    Returns
    -------
    numpy.ndarray (n_samps1 x n_samps2)
        correlation values
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(OrdinaryLeastSquares, self).__init__(ts_set1, ts_set2, **kwargs)
        self.ols_param = kwargs.get('ols_param', 'rsquared_adj')

    @staticmethod
    def calc_ols_param(x, y, ols_param='rsquared_adj'):
        import statsmodels.api as sm
        x1  = sm.add_constant(x)
        est = sm.OLS(y, x1)
        est = est.fit()

        return getattr(est, ols_param)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        return self.calc_ols_param(self.ts1, self.ts2, ols_param=self.ols_param)


#class GrangerCausalityMeasure(object):
#class MutualInformationMeasure(object):


# ----------------------------------------------------------------------------
# Many Vs Many Time Series Measures
# ----------------------------------------------------------------------------
class SeedCorrelationMeasure(object):
    """Return a list of correlation values between the time series in ts_set1 and ts_set2.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    kwargs: not used

    Returns
    -------
    list
        List of correlation values
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(SeedCorrelationMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        import nitime.analysis as nta

        analyzer = nta.SeedCorrelationAnalyzer(self.ts1, self.ts2)

        n_seeds = self.ts1.data.shape[0] if self.ts1.data.ndim > 1 else 1
        if n_seeds == 1:
            cor = analyzer.corrcoef
        else:
            cor = []
            for seed in range(n_seeds):
                cor.append(analyzer.corrcoef[seed])

        return cor


class MeanSeedCorrelationMeasure(SeedCorrelationMeasure):
    """Return the mean correlation value of all seed correlations the time series in ts_set1 and ts_set2.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    Returns
    -------
    float
        Average correlation value
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(MeanSeedCorrelationMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        return np.mean(super(MeanSeedCorrelationMeasure, self).__call__())


class SeedCoherenceMeasure(object):
    """Return a list of coherence values between the time series in ts_set1 and ts_set2.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries: n_samps x time_size
        Time series

    ts_set2: nitime.TimeSeries: n_samps x time_size
        Time series

    lb: float (optional)
    ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency).
        Define a frequency band of interest.

    kwargs:
    'NFFT'

    Returns
    -------
    list
        List of coherence values.
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(SeedCoherenceMeasure, self).__init__(ts_set1, ts_set2, **kwargs)
        self.lb = kwargs.get('lb', 0.02)
        self.ub = kwargs.get('ub', 0.15)
        self.nfft = kwargs.get('NFFT', None)

    def __call__(self):
        if self.ts1 == self.ts2:
            return 1

        import nitime.analysis as nta

        if self.nfft is not None:
            analyzer = nta.SeedCoherenceAnalyzer(self.ts1, self.ts1, lb=self.lb, ub=self.ub,
                                                 method={'NFFT': self.nfft})
        else:
            analyzer = nta.SeedCoherenceAnalyzer(self.ts1, self.ts2, lb=self.lb, ub=self.ub)

        n_seeds = self.ts1.data.shape[0] if self.ts1.data.ndim > 1 else 1
        if n_seeds == 1:
            coh = np.mean(analyzer.coherence, -1)
        else:
            coh = []
            for seed in range(n_seeds):
                # Averaging on the last dimension
                coh.append(np.mean(analyzer.coherence[seed], -1))

        return coh


class MeanSeedCoherenceMeasure(SeedCoherenceMeasure):
    """Return the mean coherence value of all seed coherences the time series in ts_set1 and ts_set2.

    Parameters
    ----------
    ts_set1: nitime.TimeSeries
        Time series matrix: n_samps x time_size

    ts_set2: nitime.TimeSeries
        Time series matrix: n_samps x time_size

    lb: float (optional)
    ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency).
        Define a frequency band of interest.

    kwargs:
    'NFFT'

    Returns
    -------
    float
        Average coherence value
    """
    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(MeanSeedCoherenceMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        return np.mean(super(MeanSeedCoherenceMeasure, self).__call__())


class MeanCoherenceMeasure(NiCoherenceMeasure):

    def __init__(self, ts_set1, ts_set2, **kwargs):
        super(MeanCoherenceMeasure, self).__init__(ts_set1, ts_set2, **kwargs)

    def __call__(self):
        return np.mean(super(MeanCoherenceMeasure, self).__call__())


class SimilarityMeasureFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def create_method(method_name):
        """
        Returns a TimeSeriesGroupMeasure class given its name.

        Parameters
        ----------
        method_name: string
            Choices:
            'correlation', 'coherence', 'nicorrelation', 'grangercausality', 'mean_coherence',
            'seedcorrelation', 'seedcoherence', 'mean_seedcoherence', 'mean_seedcorrelation'

        Returns
        -------
        callable object
            Timeseries selection method function

        Notes
        -----
        See: http://nipy.org/nitime/examples/seed_analysis.html for more information
        """
        methods = OrderedDict([ ('correlation',          CorrelationMeasure),
                                ('coherence',            NiCoherenceMeasure),
                                ('grangercausality',     NiGrangerCausalityMeasure),
                                ('nicorrelation',        NiCorrelationMeasure),
                                ('ordinaryleastsq',      OrdinaryLeastSquares),
                                ('seedcorrelation',      SeedCorrelationMeasure),
                                ('seedcoherence',        SeedCoherenceMeasure),
                                ('mean_coherence',       MeanCoherenceMeasure),
                                ('mean_seedcoherence',   MeanSeedCoherenceMeasure),
                                ('mean_seedcorrelation', MeanSeedCorrelationMeasure)])

        if method_name not in methods:
            raise KeyError('Could not find a method object for the name {}. '
                           'Please give one of the following {}.'.format(method_name, list(methods.keys())))

        return TimeSeriesGroupMeasure(methods[method_name])

        #if method_name == 'mutual_information' : algorithm = MutualInformationMeasure
        #if method_name == 'granger_causality'  : algorithm = GrangerCausalityMeasure

#
# %matplotlib inline
#
# import os
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.mlab import csv2rec
#
# import nitime
# import nitime.analysis as nta
# import nitime.timeseries as ts
# import nitime.utils as tsu
# from nitime.viz import drawmatrix_channels
#
#
# TR = 1.89
# f_ub = 0.15
# f_lb = 0.02
#
#
# data_path = os.path.join(nitime.__path__[0], 'data')
#
# data_rec = csv2rec(os.path.join(data_path, 'fmri_timeseries.csv'))
#
# roi_names = np.array(data_rec.dtype.names)
# nseq = len(roi_names)
# n_samples = data_rec.shape[0]
# data = np.zeros((nseq, n_samples))
#
# for n_idx, roi in enumerate(roi_names):
#     data[n_idx] = data_rec[roi]
#
# pdata = tsu.percent_change(data)
# time_series = ts.TimeSeries(pdata, sampling_interval=TR)
#
# G = nta.GrangerAnalyzer(time_series, order=1)
#
# C1 = nta.CoherenceAnalyzer(time_series)
# C2 = nta.CorrelationAnalyzer(time_series)
#
# freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
# freq_idx_C = np.where((C1.frequencies > f_lb) * (C1.frequencies < f_ub))[0]
#
# coh = np.mean(C1.coherence[:, :, freq_idx_C], -1)  # Averaging on the last dimension
# g1 = np.mean(G.causality_xy[:, :, freq_idx_G], -1)
#
# fig01 = drawmatrix_channels(coh, roi_names, size=[10., 10.], color_anchor=0)
#
# fig02 = drawmatrix_channels(C2.corrcoef, roi_names, size=[10., 10.], color_anchor=0)
#
# fig03 = drawmatrix_channels(g1, roi_names, size=[10., 10.], color_anchor=0)
#
# g2 = np.mean(G.causality_xy[:, :, freq_idx_G] - G.causality_yx[:, :, freq_idx_G], -1)
# fig04 = drawmatrix_channels(g2, roi_names, size=[10., 10.], color_anchor=0)
#
