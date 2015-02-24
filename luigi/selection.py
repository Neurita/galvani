# coding=utf-8
#-------------------------------------------------------------------------------
#Author: Alexandre Manhães Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np
import nitime.fmri.io as fio


class TimeSeriesSelector(object):
    """
    A strategy class to use any of the time series selection methods given as a callable objet.
    """

    def __init__(self, algorithm):
        """
        TimeSeriesSelector, Strategy class constructor.

        Parameters
        ----------
        algorithm: callable object
            The selection method
        """
        self.algorithm = algorithm
        self.selected_ts = None

    def fit_transform(self, ts_set, **kwargs):
        """
        Returns selected timeseries from ts_set.

        Parameters
        -----------
        ts_set: nitime.Timeseries
            n_timeseries x ts_size

        Returns
        -------
        numpy.ndarray
            N x time_size
        """
        self.ts_selector = self.algorithm(ts_set, **kwargs)
        return self.ts_selector.selected_ts


class MeanTimeseries(object):

    def __init__(self, ts_set, **kwargs):
        """
        Returns the average timeseries from an array.

        Parameters
        ----------
        ts_set: nitime.Timeseries
            n_samps x time_size. Time series matrix.

        kwargs: (not used here)

        Returns
        -------
        average timeseries: numpy.ndarray (1 x time_size)
            Will return the same type as ts_set.
        """
        if hasattr(ts_set, 'data'):
            ts_set.data = ts_set.data.mean(axis=0)
        else:
            ts_set = ts_set.mean(axis=0)

        self.selected_ts = ts_set


class EigenTimeseries(object):
    """
    Return from an array ts_set of time series the eigen time series.

    Parameters
    ----------
    ts_set: nitime.Timeseries or ndarray
        n_samps x time_size. Time series matrix.

    Kwargs:  'n_comps'   : the number of components to be selected from the set. Default 1.
             'comps_perc': the percentage of components to be selected from the set

    Returns
    -------
    eigen timeseries : numpy.ndarray (n_comps x time_size)
        Will return the same type as ts_set.
    """
    def __init__(self, ts_set, **kwargs):

        from sklearn.decomposition import PCA
        #eigen time series (PCA)

        n_comps    = kwargs.get('n_comps', 1)
        comps_perc = kwargs.get('comps_perc')
        if comps_perc is not None:
            if comps_perc > 1:
                comps_perc /= 100

            n_comps = np.floor(ts_set.shape[0] * comps_perc)

        x = ts_set.data if hasattr(ts_set, 'data') else ts_set

        pca = PCA(n_components=n_comps)

        x = pca.fit_transform(x.T).T

        if x.shape[0] > n_comps:
            x = ts_set.data[0:n_comps, :]

        if hasattr(ts_set, 'data'):
            ts_set.data = x
            self.selected_ts = ts_set
        else:
            self.selected_ts = x


class ILSIATimeseries(object):
    """Return from an array ts_set of time series, the ones selected with ILSIA algorithm

    Input:
    -----
    ts_set:  nitime.Timeseries or ndarray
        n_samps x time_size. Time series matrix.

    Kwargs:  'n_comps'   : the number of components to be selected from the set. Default 1.
             'comps_perc': the percentage of components to be selected from the set

    Returns
    -------
    ilsia timeseries : numpy.ndarray (n_comps x time_size)
    """

    def __init__(self, ts_set, **kwargs):
        from .endmember_induction import ILSIA

        n_comps    = kwargs.get('n_comps', 1)
        comps_perc = kwargs.get('comps_perc')
        if comps_perc is not None:
            if comps_perc > 1:
                comps_perc /= 100
            n_comps = np.floor(ts_set.shape[0] * comps_perc)

        if hasattr(ts_set, 'data'):
            ilsia = ILSIA(ts_set.data.T, p=n_comps)
            em, cnt, idx = ilsia.fit()
            ts_set.data = em.T
        else:
            ilsia = ILSIA(ts_set.T, p=n_comps)
            em, cnt, idx = ilsia.fit()
            ts_set = em.T

        self.selected_ts = ts_set


class CCATimeseries(object):
    """Return from an array ts_set of time series, the ones selected with
    Convex Cone Analysis (CCA) algorithm.

    Parameters
    ----------
    ts_set:  nitime.Timeseries or numpy.ndarray
        n_samps x time_size. Time series matrix.

    Kwargs:  'n_comps'   : the number of components to be selected from the set. Default 1.
             'comps_perc': the percentage of components to be selected from the set

    Returns
    -------
    cca timeseries : numpy.ndarray (n_comps x time_size)
    """
    def __init__(self, ts_set, **kwargs):
        from .endmember_induction import CCA

        n_comps = kwargs.get('n_comps', 1)
        if not 'n_comps' in kwargs and 'comps_perc' in kwargs:
            comps_perc = kwargs['comps_perc']
            if comps_perc > 1:
                comps_perc /= 100
            n_comps = np.floor(ts_set.shape[0] * comps_perc)

        if hasattr(ts_set, 'data'):
            cca = CCA(ts_set.data.T, p=n_comps)
            ts_set.data = cca.fit().T
        else:
            cca = CCA(ts_set.T, p=n_comps)
            ts_set = cca.fit().T

        self.selected_ts = ts_set


class FilteredTimeseries(object):
    """Return frequency filtered timeseries from ts_set.

    Parameters
    ----------
    ts_set:  nitime.Timeseries
        n_samps x time_size. Time series matrix.

    kwargs:
        TR: float
            The sampling interval

        pre_filter   : dict or list of dicts
            One dict or a list of dicts where each dict contains the keys:
            'lb' and 'ub' that indicate the filter lower and upper bands.

             Default: 0, Nyquist
             If you want the default along with others, append a None value in the
             'sel_filter' list.

           {'lb':float or 0, 'ub':float or None, 'method':'fourier','boxcar' 'fir'
           or 'iir' }

           each voxel's data will be filtered into the frequency range [lb,ub] with
           nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
           to 'fir')

        Filtering options:
        -----------------

        boxcar_iterations: int (optional)
           For box-car filtering, how many times to iterate over the data while
           convolving with a box-car function. Default: 2

        gpass: float (optional)
           For iir filtering, the pass-band maximal ripple loss (default: 1)

        gstop: float (optional)
           For iir filtering, the stop-band minimal attenuation (default: 60).

        filt_order: int (optional)
            For iir/fir filtering, the order of the filter. Note for fir filtering,
            this needs to be an even number. Default: 64

        iir_ftype: str (optional)
            The type of filter to be used in iir filtering (see
            scipy.signal.iirdesign for details). Default 'ellip'

        fir_win: str
            The window to be used in fir filtering (see scipy.signal.firwin for
            details). Default: 'hamming'

        See: http://nipy.org/nitime/api/generated/nitime.analysis.spectral.html#nitime.analysis.spectral.FilterAnalyzer
        for more details and named arguments.

    Returns
    -------
    filtered timeseries : numpy.ndarray ([n_filters * n_samps] x time_size)

    Notes
    -----
    Each whole set of filtered tseries will be pushed to the end of the
    output array.

    """

    def __init__(self, ts_set, **kwargs):

        sel_filter = kwargs.get('pre_filter', None)
        TR         = kwargs.get('TR', 2)

        if sel_filter is None:
            return ts_set

        n_samps = ts_set.shape[0]   if ts_set.ndim > 1 else 1
        ts_set  = ts_set.squeeze()  if n_samps == 1    else list(ts_set)

        filts = []
        filts.append(ts_set)

        for f in sel_filter:
            if f is not None:
                filt = {}
                filt['lb']     = f.get('lb', 0)
                filt['ub']     = f.get('ub', None)
                filt['method'] = f.get('method', 'fir')

            filts.append(fio._tseries_from_nifti_helper(None, ts_set, TR, filt,
                                                        kwargs.get('normalize',
                                                                   None),
                                                        kwargs.get('average',
                                                                   None)))

        if n_samps == 1:
            self.selected_ts = np.array(filts)
        else:
            np.squeeze(np.array(filts))


class MeanAndFilteredTimeseries(MeanTimeseries, FilteredTimeseries):
    """Return from an array of timeseries the average and filtered versions of it.

    Parameters
    ----------
    ts_set:  nitime.Timeseries or ndarray
        n_samps x time_size. Time series matrix.

        Kwargs
        ------
        TR: float
            The sampling interval

        sel_filter   : dict or list of dicts
            One dict or a list of dicts where each dict contains the keys:
            'lb' and 'ub' that indicate the filter lower and upper bands.

             Default: 0, Nyquist
             If you want the default along with others, append a None value in the
             'sel_filter' list.

           {'lb':float or 0, 'ub':float or None, 'method':'fourier','boxcar' 'fir'
           or 'iir' }

           each voxel's data will be filtered into the frequency range [lb,ub] with
           nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
           to 'fir')

        See Filtered_Timeseries docstring for detailed filtering options

    Returns
    -------
    numpy.ndarray (n_filters x time_size)
        mean timeseries and its filtereds
    """
    def __init__(self, ts_set, **kwargs):

        FilteredTimeseries.__init__(self, ts_set, **kwargs)
        MeanTimeseries.__init__(self, self.selected_ts, **kwargs)


class EigenAndFilteredTimeseries(EigenTimeseries, FilteredTimeseries):

    def __init__(self, ts_set, **kwargs):
        FilteredTimeseries.__init__(self, ts_set, **kwargs)
        EigenTimeseries.__init__(self, self.selected_ts, **kwargs)


class TimeseriesSelectorFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def create_method(method_name):
        """Return a Timeseries selection method fit function given a method name.

        Parameters
        ----------
        method_name: string
            Choices for name of the method: 'mean', 'eigen', 'ilsia', 'cca'
                                            'filtered', 'mean_and_filtered', 'eigen_and_filtered'

        Returns
        -------
        callable object
            Timeseries selection method function
        """
        algorithm = MeanTimeseries
        if method_name == 'mean' : algorithm =  MeanTimeseries
        if method_name == 'eigen': algorithm =   EigenTimeseries
        if method_name == 'ilsia': algorithm =  ILSIATimeseries
        if method_name == 'cca'  : algorithm =  CCATimeseries

        if method_name == 'filtered'          : algorithm = FilteredTimeseries
        if method_name == 'mean_and_filtered' : algorithm = MeanAndFilteredTimeseries
        if method_name == 'eigen_and_filtered': algorithm = EigenAndFilteredTimeseries

        return TimeSeriesSelector(algorithm)

#TODO?
##-------------------------------------------------------------------------------
#class Lagged_Timeseries:

#    def __init__(self):
#        pass

#    def __call__ (self, ts_set, **kwargs):
#        """
#        Returns from an array of timeseries the lagged versions of them.
#        #---------------------------------------------------------------
#        Input: 
#        -----
#        ts_set: ndarray
#            n_samps x time_size. Time series matrix.

#        lag_range: 

#        Kwargs
#        ------
#        TR: sampling interval of the timeseries
#            Default: 2

#        shifts: dict for lagged ts generation, optional
#            If provided with a dict of the form:
#                {'lb':int, 'ub':int} for each value in range(lb, ub+1) a lagged
#                version of each extracted ts will be included in the ts set.
#                Default: {'lb': -3*TR, 'ub': +3*TR}

#        Output:  mean timeseries: 1 x time_size
#        """
#        mean_ts = Mean_Timeseries.__call__(self, ts_set, **kwargs)

#        TR = kwargs['TR'] if kwargs.has_key('TR') else 2

#        if kwargs.has_key('shifts'):
#            lb = kwargs['shifts']['lb']
#            ub = kwargs['shifts']['ub']
#        else:
#            lb = -3*TR
#            ub =  3*TR
#        shifts = range(int(lb), int(ub+1))

#        lag_ts = [ts_set]
#        for s in shifts:
#            lag_ts.append(np.roll(ts_set, s))

#        return lag_ts

##-------------------------------------------------------------------------------
# '''
# This function has been copied (and modified) from Spectrum library.
# http://thomas-cokelaer.info/software/spectrum/html/contents.html
#
# A Python library created by Thomas Cokelaer:
# http://thomas-cokelaer.info/blog/
# '''
# def xcorr(x, y=None, maxlags=None, norm='biased'):
#     """Cross-correlation using numpy.correlate
#
#     Estimates the cross-correlation (and autocorrelation) sequence of a random
#     process of length N. By default, there is no normalisation and the output
#     sequence of the cross-correlation has a length 2*N+1.
#
#     :param array x: first data array of length N
#     :param array y: second data array of length N. If not specified, computes the
#         autocorrelation.
#     :param int maxlags: compute cross correlation between [-maxlags:maxlags]
#         when maxlags is not specified, the range of lags is [-N+1:N-1].
#     :param str option: normalisation in ['biased', 'unbiased', None, 'coeff']
#
#     The true cross-correlation sequence is
#
#     .. math:: r_{xy}[m] = E(x[n+m].y^*[n]) = E(x[n].y^*[n-m])
#
#     However, in practice, only a finite segment of one realization of the
#     infinite-length random process is available.
#
#     The correlation is estimated using numpy.correlate(x,y,'full').
#     Normalisation is handled by this function using the following cases:
#
#         * 'biased': Biased estimate of the cross-correlation function
#         * 'unbiased': Unbiased estimate of the cross-correlation function
#         * 'coeff': Normalizes the sequence so the autocorrelations at zero
#            lag is 1.0.
#
#     :return:
#         * a numpy.array containing the cross-correlation sequence (length 2*N-1)
#         * lags vector
#
#     .. note:: If x and y are not the same length, the shorter vector is
#         zero-padded to the length of the longer vector.
#
#     .. rubric:: Examples
#
#     .. doctest::
#
#         >>> from spectrum import *
#         >>> x = [1,2,3,4,5]
#         >>> c, l = xcorr(x,x, maxlags=0, norm='biased')
#         >>> c
#         array([ 11.])
#
#     .. seealso:: :func:`CORRELATION`.
#     """
#     import numpy as np
#     from pylab import rms_flat
#
#     N = len(x)
#     if y == None:
#         y = x
#     assert len(x) == len(y), 'x and y must have the same length. Add zeros if needed'
#     assert maxlags <= N, 'maxlags must be less than data length'
#
#     if maxlags == None:
#         maxlags = N-1
#         lags = np.arange(0, 2*N-1)
#     else:
#         assert maxlags < N
#         lags = np.arange(N-maxlags-1, N+maxlags)
#
#     res = np.correlate(x, y, mode='full')
#
#     if norm == 'biased':
#         Nf = float(N)
#         res = res[lags] / float(N)    # do not use /= !!
#     elif norm == 'unbiased':
#         res = res[lags] / (float(N)-abs(np.arange(-N+1, N)))[lags]
#     elif norm == 'coeff':
#         Nf = float(N)
#         rms = rms_flat(x) * rms_flat(y)
#         res = res[lags] / rms / Nf
#     else:
#         res = res[lags]
#
#     #lags = np.arange(-maxlags, maxlags+1)
#     return res#, lags
