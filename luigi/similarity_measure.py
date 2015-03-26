# coding=utf-8
#-------------------------------------------------------------------------------
#Author: Alexandre Manhães Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np
import nitime
import nitime.analysis as nta

#TODO: Give the option here to use full matrices, instead of just ts_sets

#See Ledoit-Wolf covariance estimation and Graph-Lasso
#http://www.sciencedirect.com/science/article/pii/S1053811910011602
#http://nilearn.github.io/data_analysis/functional_connectomes.html
#http://scikit-learn.org/stable/modules/covariance.html
#http://nilearn.github.io/developers/group_sparse_covariance.html


class TimeSeriesGroupMeasure(object):
    """A strategy class to use any of the time series group measures methods given as a callable objet.

    Parameters
    ----------
    algorithm: callable object
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
        """
        self.measurer = self.algorithm(ts_set1, ts_set2, **kwargs)

        self.measure  = self.measurer.measure

        return self.measure


class NiCorrelationMeasure(object):
    """Return a the Pearson's Correlation value between all time series in both sets.
    Calculates the correlation using nitime.

    It's a redundant implementation, I haven't compared them yet

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
        Scalar correlation value
    """
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2, TR=TR)

    @staticmethod
    def _fit(ts_set1, ts_set2, TR):
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

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
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2, lb, ub, TR)

    @staticmethod
    def _fit(ts_set1, ts_set2, lb, ub, TR):
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

        coh = nta.CoherenceAnalyzer(ts)

        freq_idx_coh = np.where((coh.frequencies > lb) * (coh.frequencies < ub))[0]

        return np.mean(coh.coherence[:, :, freq_idx_coh], -1)  # Averaging on the last dimension


class NiGrangerCausalityMeasure(object):
    """
    Returns a the Granger Causality value all time series in both sets.

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
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2, lb, ub, TR)

    @staticmethod
    def _fit(ts_set1, ts_set2, lb, ub, TR):
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

        gc = nta.GrangerAnalyzer(ts, order=1)

        freq_idx_gc = np.where((gc.frequencies > lb) * (gc.frequencies < ub))[0]

        return np.mean(gc.causality_xy[:, :, freq_idx_gc], -1)


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
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2)

    @staticmethod
    def _fit(ts_set1, ts_set2):
        from scipy.stats import pearsonr

        n1 = ts_set1.shape[0]
        n2 = ts_set2.shape[0]
        if (ts_set1.ndim > 1 or ts_set2.ndim > 1) and n1 == n2 > 1:
            mp = np.array(n1, n2)
            for i1 in list(range(n1)):
                t1 = ts_set1[i1, :]
                for i2 in list(range(n2)):
                    t2 = ts_set2[i2, :]
                    mp[i1, i2] = pearsonr(t1, t2)

            return mp

        else:
            return pearsonr(ts_set1.flatten(), ts_set2.flatten())[0]




#class GrangerCausalityMeasure(OneVsOneTimeSeriesMeasure):
#class MutualInformationMeasure(OneVsOneTimeSeriesMeasure):

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
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2)

    @staticmethod
    def _fit(ts_set1, ts_set2):
        import nitime.analysis as nta

        analyzer = nta.SeedCorrelationAnalyzer(ts_set1, ts_set2)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
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
        Average correlation value
    """
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):

        super(MeanSeedCorrelationMeasure, self).__init__(ts_set1, ts_set2, lb=lb, ub=ub, TR=TR, **kwargs)
        self.measure = np.mean(self.measure)


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
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        self.measure = self._fit(ts_set1, ts_set2, lb, ub, **kwargs)

    @staticmethod
    def _fit(ts_set1, ts_set2, lb, ub, **kwargs):
        import nitime.analysis as nta

        fft_par = kwargs.pop('NFFT', None)
        if fft_par is not None:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub,
                                                 method={'NFFT': fft_par})
        else:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1:
            coh = np.mean(analyzer.coherence, -1)
        else:
            coh = []
            for seed in range(n_seeds):
                # Averaging on the last dimension
                coh.append(np.mean(analyzer.coherence[seed], -1))

        return coh


class MeanSeedCoherenceMeasure(SeedCoherenceMeasure):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
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
        super(MeanSeedCoherenceMeasure, self).__init__(ts_set1, ts_set2, lb=lb, ub=ub, TR=TR, **kwargs)
        self.measure = np.mean(self.measure)


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
            'correlation', 'coherence', 'nicorrelation', 'grangercausality'
            'seedcorrelation', 'seedcoherence', 'mean_seedcoherence', 'mean_seedcorrelation'

        Returns
        -------
        callable object
            Timeseries selection method function

        Notes
        -----
        See: http://nipy.org/nitime/examples/seed_analysis.html for more information
        """

        algorithm = CorrelationMeasure
        if method_name == 'correlation'            : algorithm = CorrelationMeasure
        if method_name == 'coherence'              : algorithm = NiCoherenceMeasure
        if method_name == 'grangercausality'       : algorithm = NiGrangerCausalityMeasure
        if method_name == 'nicorrelation'          : algorithm = NiCorrelationMeasure

        if method_name == 'seedcorrelation'        : algorithm = SeedCorrelationMeasure
        if method_name == 'seedcoherence'          : algorithm = SeedCoherenceMeasure
        if method_name == 'mean_seedcoherence'     : algorithm = MeanSeedCoherenceMeasure
        if method_name == 'mean_seedcorrelation'   : algorithm = MeanSeedCorrelationMeasure

        #if method_name == 'mutual_information' : algorithm = MutualInformationMeasure
        #if method_name == 'granger_causality'  : algorithm = GrangerCausalityMeasure

        return TimeSeriesGroupMeasure(algorithm)


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
# from nitime.plotting import drawmatrix_channels
#
# TR = 1.89
# f_ub = 0.15
# f_lb = 0.02
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
# G = nta.GrangerAnalyzer(time_series, order=1)
# C1 = nta.CoherenceAnalyzer(time_series)
# C2 = nta.CorrelationAnalyzer(time_series)
# freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
# freq_idx_C = np.where((C1.frequencies > f_lb) * (C1.frequencies < f_ub))[0]
#
# coh = np.mean(C1.coherence[:, :, freq_idx_C], -1)  # Averaging on the last dimension
# g1 = np.mean(G.causality_xy[:, :, freq_idx_G], -1)
#
# fig01 = drawmatrix_channels(coh, roi_names, size=[10., 10.], color_anchor=0)
#
# fig02 = drawmatrix_channels(C2.corrcoef, roi_names, size=[10., 10.], color_anchor=0)
# g2 = np.mean(G.causality_xy[:, :, freq_idx_G] - G.causality_yx[:, :, freq_idx_G], -1)
# fig04 = drawmatrix_channels(g2, roi_names, size=[10., 10.], color_anchor=0)
