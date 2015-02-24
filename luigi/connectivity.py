# coding=utf-8
#-------------------------------------------------------------------------------
#Author: Alexandre Manhães Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#Use this at your own risk!
#-------------------------------------------------------------------------------

import logging

import numpy                as np
import nitime.fmri.io       as tsio

from   collections          import OrderedDict
from   boyle.nifti.read     import repr_imgs
from   boyle.nifti.roi      import partition_timeseries
from   boyle.nifti.check    import check_img_compatibility

from   .selection           import TimeseriesSelectorFactory
from   .similarity_measure  import SimilarityMeasureFactory

log = logging.getLogger(__name__)


class FunctionalConnectivity(object):
    """

    """
    def __init__(self, func_vol, atlas, mask=None, TR=2, roi_list=None,
                 selection_method='eigen', similarity_measure='correlation'):
        """
        Parameters
        ----------
        func_vol: nibabel SpatialImage or boyle.nifti.NeuroImage
            Time series MRI volume.

        atlas: nibabel SpatialImage or boyle.nifti.NeuroImage
            3D Atlas volume with discrete ROI values
            It will be accessed from lower to greater ROI value, that
            will be the order in the connectivity matrix, unless you
            set roi_list with the order of appearance you want.
            It must be in the same space as func_vol.

        mask: nibabel SpatialImage or boyle.nifti.NeuroImage
            Binary 3D mask volume, e.g., GM mask to extract ROI timeseries only from GM.

        TR: int or float
            Repetition time of the acquisition protocol used for the fMRI from
            where ts_set has been extracted.

        roi_list: list of ROI values
            List of the values of the ROIs to indicate the order of access to
            the ROI data.

        selection_method: string
            Defines the timeseries selection method to be applied within each ROI.
            Choices: 'mean', 'eigen', 'ilsia', 'cca'
                     'filtered', 'mean_and_filtered', 'eigen_and_filtered'
            See .timeseries.selection more information.

        similarity_measure: string
            Defines the similarity measure method to be used between selected timeseries.
            Choices: 'crosscorrelation', 'correlation', 'coherence',
                     'mean_coherence', 'mean_correlation', 'nicorrelation'
            See .timeseries.similarity_measure for more information.

        Raises
        ------
        ValueError
        If func_vol and atlas do not have the same 3D shape.
        """
        self.func_vol           = func_vol
        self.atlas              = atlas
        self.mask               = mask
        self.sampling_interval  = TR
        self.roi_list           = roi_list
        self.selection_method   = selection_method
        self.similarity_measure = similarity_measure

        self._tseries       = None
        self._selected_ts   = None
        self._func_conn     = None
        self._use_lists     = True
        self._args          = {}

        self._set_up()

    def _self_check(self):
        try:
            check_img_compatibility(self.func_vol, self.atlas)
        except:
            log.exception('Functional and atlas volumes do not have the shame spatial shape.')
            raise

        if self.mask is not None:
            try:
                check_img_compatibility(self.func_vol, self.mask)
            except:
                log.exception('Functional and atlas volumes do not have the shame spatial shape.')
                raise

    def _set_up(self):
        if self._use_lists:
            self._tseries = []
        else:
            self._tseries = OrderedDict()

    def extract_timeseries(self, **kwargs):
        """
        Extract from the functional volume the timseries and separate them
        in an ordered dict in self._tseries.

        Parameters
        ----------
        kwargs:  dict with the following keys, some are optional

            'normalize': Whether to normalize the activity in each voxel, defaults to
                None, in which case the original fMRI signal is used. Other options are:
                'percent': the activity in each voxel is converted to percent
                change, relative to this scan.
                'zscore': the activity is converted to a zscore relative to the mean and
                std in this voxel in this scan.

            'pre_filter': dict, optional
                If provided with a dict of the form:
                :var 'lb': float or 0
                Filter lower-bound

                :var 'ub': float or None
                Filter upper-bound

                :var 'method': string
                Filtering method
                Choices: 'fourier','boxcar', 'fir' or 'iir'

                Each voxel's data will be filtered into the frequency range [lb, ub] with
                nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
                to 'fir')
        """
        try:
            self._self_check()
        except:
            raise

        if self.mask is not None:
            mask_vol = self.mask.get_data()

        func_vol  = self.func_vol.get_data()
        atlas_vol = self.atlas.get_data()

        pre_filter = kwargs.pop('pre_filter', None)
        normalize  = kwargs.pop('normalize', None)

        tseries = partition_timeseries(func_vol, atlas_vol, mask_vol, zeroe=True, roi_list=self.roi_list,
                                       outdict=(not self._use_lists))

        if isinstance(tseries, list):
            #filtering
            for ts in tseries:
                tsset = tsio._tseries_from_nifti_helper(None, ts,
                                                        self.sampling_interval,
                                                        pre_filter, normalize,
                                                        None)
                self._tseries.append(tsset)

        elif isinstance(tseries, dict):
            #filtering
            for r in self.roi_list:
                tsset = tsio._tseries_from_nifti_helper(None, tseries[r],
                                                        self.sampling_interval,
                                                        pre_filter, normalize,
                                                        None)
                self._tseries[r] = tsset
        else:
            raise ValueError('Error extracting timeseries data from {}.'
                             ''.format(repr_imgs(func_vol)))


    def _select_timeseries(self, **kwargs):
        """
        Selects significant timeseries from the dict of sets of timeseries.
        Each item in ts_set will be transformed to one or fewer timeseries.

        Parameters
        ----------
        kwargs: dict with the following keys, all are optional

            'n_comps'   : int
                The number of components to be selected from the set. Default 1.

            'comps_perc': float from [0, 100]
                The percentage of components to be selected from the set, will
                ignore 'n_comps' if this is set.

            #TODO
            'shifts': int or dict
                For lagged ts generation.
                If provided with an int b, will use the range [-b, b]
                If provided with a dict of the form:
                :var 'lb': int
                :var 'ub': int
                For each value in range(lb, ub+1) a lagged version of each
                extracted ts will be included in the ts set. Default: {'lb': -3, 'ub': +3}

        Returns
        -------
        dict
            Dictionary with the same keys as ts_set, where each item in ts_set is
            a transformed/reduced set of timeseries. self._selected_ts
        """
        if self._tseries is None:
            self.extract_timeseries(**kwargs)
            kwargs.pop('normalize')
            kwargs.pop('average')

        ts_selector = TimeseriesSelectorFactory.create_method(self.selection_method)

        if isinstance(self._tseries, dict):
            self._selected_ts = OrderedDict()

            for r in self.roi_list:
                self._selected_ts[r] = ts_selector.fit_transform(self._tseries[r],
                                                                 **kwargs)

        elif isinstance(self._tseries, list):
            self._selected_ts = []
            for ts in self._tseries:
                self._selected_ts.append(ts_selector.fit_transform(ts,
                                                                   **kwargs))

    def _calculate_similarities(self, **kwargs):
        """
        Calculate a matrix of correlations/similarities between all
        timeseries in tseries.

        Parameters
        ----------
        kwargs: dict with the following keys

            'TR': int
                Data sampling interval.
                Default: 2

            'lb': int
                Lower bound frequency limit.
                Default: 0

            'ub': int
                Upper bound frequency limit.
                Default: None
        """
        if self._selected_ts is None:
            self._select_timeseries(**kwargs)

        simil_measure = SimilarityMeasureFactory.create_method(self.similarity_measure)

        n_rois = len(self._selected_ts)
        cmat = np.zeros((n_rois, n_rois))

        lb = kwargs.pop('lb', 0)
        ub = kwargs.pop('ub', None)

        # this will benefit from the ordering of the time series and
        # calculate only half matrix, then sum its transpose
        if isinstance(self._selected_ts, list):

            for tsi1, ts1 in enumerate(self._selected_ts):
                for tsi2, ts2 in enumerate(self._selected_ts):
                    cmat[tsi1, tsi2] = simil_measure.fit_transform(ts1, ts2,
                                                                   lb=lb, ub=ub,
                                                                   TR=self.sampling_interval, **kwargs)


        # this will calculate the cmat fully without the "symmetrization"
        elif isinstance(self._selected_ts, dict):

            c1 = 0
            for tsi1 in self.roi_list:
                c2 = 0
                for tsi2 in self.roi_list:
                    cmat[c1, c2] = simil_measure.fit_transform(self._selected_ts[tsi1],
                                                               self._selected_ts[tsi2],
                                                               lb=lb, ub=ub,
                                                               TR=self.sampling_interval, **kwargs)
                    c2 += 1
                c1 += 1

        #cmat = cmat + cmat.T
        #cmat[np.diag_indices_from(cmat)] /= 2

        self._func_conn = cmat

    def fit_transform(self, **kwargs):
        """
        Calculate a matrix of correlations/similarities between all timeseries in
        the tseries extracted from the functional data.

        :param kwargs: dict with the following keys, some are optional

        * TIMESERIES EXTRACTION AND PRE-PROCESSING
        :var 'normalize': Whether to normalize the activity in each voxel, defaults to
            None, in which case the original fMRI signal is used. Other options
            are: 'percent': the activity in each voxel is converted to percent
            change, relative to this scan. 'zscore': the activity is converted to a
            zscore relative to the mean and std in this voxel in this scan.

        :var 'pre_filter': dict, optional
            If provided with a dict of the form:
            :var 'lb': float or 0
            Filter lower-bound

            :var 'ub': float or None
            Filter upper-bound

            :var 'method': string
            Filtering method
            Choices: 'fourier','boxcar', 'fir' or 'iir'

        See .macuto.selection FilteredTimeseries doc strings.

        * ROI TIMESERIES SELECTION
        :var 'n_comps'   : int
        The number of components to be selected from the set, if applies.
        Default 1.

        :var 'comps_perc': float from [0, 100]
        The percentage of components to be selected from the set, will
        ignore 'n_comps' if this is set.

        * SIMILARITY MEASURES AND POST-PROCESSING
        :var 'post_filter': dict, optional
        If provided with a dict of the form:

            :var 'lb': float or 0
            Filter lower-bound

            :var 'ub': float or None
            Filter upper-bound

            :var 'method': string
            Filtering method
            Choices: 'fourier','boxcar', 'fir' or 'iir'

        :return: ndarray
        A NxN ndarray with the connectivity cross-measures between all timeseries
        in ts_set
        """
        self._args = kwargs
        self._calculate_similarities(**self._args)
        return self._func_conn



    #TODO
    # if kwargs.has_key('fekete-wilf'):
    #
    #     kwargs['sel_filter'] = []
    #     kwargs['sel_filter'].append({'lb': 0.01, 'ub': 0.1 , 'method': 'boxcar'})
    #     kwargs['sel_filter'].append({'lb': 0.03, 'ub': 0.06, 'method': 'boxcar'})
    #
    #     repts = get_rois_timeseries(func_vol, atlas, TR, **kwargs)
    #
    #     debug_here()
    #     '''
    #     wd  = '/home/alexandre/Dropbox/Documents/phd/work/cobre/'
    #     tsf = 'ts_test.pyshelf'
    #
    #     #save
    #     import os
    #     au.shelve_varlist(os.path.join(wd,tsf), ['tseries'], [repts])
    #
    #     #load
    #     import os
    #     import shelve
    #     tsf = os.path.join(wd,tsf)
    #
    #     data    = shelve.open(tsf)
    #     tseries = data['tseries']
    #
    #     lag = 3
    #
    #     lag_corrs  =
    #     part_corrs =
    #     for key, val in tseries.iteritems():
    #         #xcorr(x, y=None, maxlags=None, norm='biased'):
    #         corr01 = xcorr(val.data[0,:], val.data[1,:], maxlags=lag) #feat01
    #         corr12 = xcorr(val.data[1,:], val.data[2,:], maxlags=lag) #feat02
    #         corr20 = xcorr(val.data[2,:], val.data[0,:], maxlags=lag) #feat03
    #
    #     '''
    #
    #     '''
    #     We first extracted the average time series from the 116 automated anatomical
    #     labeling ROIs [28],
    #     The resulting time series were filtered into the
    #     0.01-0.1 Hz [22] and 0.03-0.06 Hz frequency bands [30]. For
    #     each time series array - both the filtered and original time series -
    #     we computed the lagged correlations and partial correlations
    #     ranging from +-3TR and also derived the maximal correlation of
    #     the seven. Negative values were set to zero, as well as
    #     autocorrelations. The correlation matrices were thresholded to
    #     leave a fraction a of the strongest connections using
    #     alpha = [.5, .4, ..., .1] to produce 240 graphs (3x2x8x5; fre-
    #     quency bands, linear/partial correlation, seven lags and their
    #     maximum and five thresholds respectively - see Figure S1). From
    #     each resulting connectivity matrix both weighted and binary
    #      global features were harvested. For local measures, we focused on
    #      a subset of these graphs that has been reported to be
    #      discriminative, zero lagged partial correlations in the 0.01-
    #        0.1 Hz band [22], from which both binary and weighted features
    #        were derived for each ROI.
    #     Docs:
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/journal.pone.0062867.s001.pdf
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/fekete-combining_classification_fmri_neurodiagnostics-2013.pdf
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/sato-fmri_connectivity_granger-2010.pdf
    #     '''

    #     cmat = calculate_fekete_wilf_ts_connmatrix (repts, TR, **kwargs)
    # else:
    #     repts = get_rois_timeseries(func_vol, atlas, TR, **kwargs)
    #     #calculate connectivity_matrices
    #     cmat = calculate_ts_connmatrix (repts, **kwargs)
    #
    # return cmat

#TODO?
# def save_connectivity_matrices(funcs, aal_rois, outfile, TR=None, **kwargs):
#     '''
#     Save in output_basename.pyshelf a list of connectivity matrices extracted from
#     the Parameters.
#
#     Parameters
#     ----------
#     funcs : a string or a list of strings.
#            The full path(s) to the file(s) from which the time-series is (are)
#            extracted.
#
#     aal_rois: a string or a list of strings.
#            The full path(s) to the file(s) from which the ROIs is (are)
#            extracted.
#
#     outfile: a string
#            The full path to the pyshelf file containing the list of connectivity
#             matrices produced here. The '.pyshelf' extension will be added.
#
#             If it is empty, the list will be only returned as object.
#
#     TR: float, optional
#         TR, if different from the one which can be extracted from the nifti
#         file header
#
#     Kwargs:
#     ------
#     normalize: Whether to normalize the activity in each voxel, defaults to
#         None, in which case the original fMRI signal is used. Other options
#         are: 'percent': the activity in each voxel is converted to percent
#         change, relative to this scan. 'zscore': the activity is converted to a
#         zscore relative to the mean and std in this voxel in this scan.
#
#     average: bool, optional whether to average the time-series across the
#            voxels in the ROI (assumed to be the first dimension). In which
#            case, TS.data will be 1-d
#
#     filter: dict, optional
#        If provided with a dict of the form:
#
#        {'lb':float or 0, 'ub':float or None, 'method':'fourier','boxcar' 'fir'
#        or 'iir' }
#
#        each voxel's data will be filtered into the frequency range [lb,ub] with
#        nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
#        to 'fir')
#
#     See TimeseriesSelection_Factory and SimilarityMeasure_Factory methods docstrings.
#
#     tssel_method  : defines the ts selection method. Options: 'mean', 'eigen', 'ilsia', 'cca'
#     simil_measure : defines the similarity measure method.
#         Options: 'crosscorrelation', 'correlation', 'coherence',
#                  'mean_coherence', 'mean_correlation'
#
#
#     For ts selection methods:
#     n_comps   : the number of components to be selected from the set. Default 1.
#     comps_perc: the percentage of components to be selected from the set
#
#     For similarity measure methods:
#
#
#     Returns
#     -------
#
#     The list of connectivity matrices in the same order as the input funcs.
#
#     Note
#     ----
#
#     Normalization occurs before averaging on a voxel-by-voxel basis, followed
#     by the averaging.
#
#     '''
#
#     if isinstance(funcs,str):
#         funcs = [funcs]
#
#     if isinstance(aal_rois,str):
#         aal_rois = [aal_rois]
#
#     matrices = []
#
#     #processing ROIs and time series
#     for i, funcf in enumerate(funcs):
#
#         print funcf
#
#         roisf = aal_rois[i]
#
#         fnc_nii = nib.load(funcf)
#         aal_nii = nib.load(roisf)
#
#         fvol = fnc_nii.get_data()
#         avol = aal_nii.get_data()
#
#         if not TR:
#             TR = get_sampling_interval(fnc_nii)
#
#         try:
#             mat = create_conn_matrix(fvol, avol, TR, **kwargs)
#
#         except ArithmeticError as err:
#             print ('Exception on calculate_connectivity_matrix on ' + funcf + ' and ' + roisf)
#             print ("{0}".format(err))
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             raise
#
#         matrices.append([funcf, roisf, mat])
#
#     #save all connectivity_matrices
#     if not 'pyshelf' in outfile:
#         outfile += '.pyshelf'
#
#     au.shelve_varlist(outfile, ['matrices'], [matrices])
#
#     return matrices