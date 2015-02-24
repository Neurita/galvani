# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

"""
The :mod:`luigi` module includes functions to handle timeseries
from fMRI data: timeseries.selection has functions to select timeseries from
sets of fMRI timeseries, and timeseries.similarity_measure has functions to
measure similarities between sets of timeseries.
"""

from .utils.logger import setup_logging

setup_logging()


#__all__ = []


#from .selection import (TimeSeriesSelector,
#                        MeanTimeseries,
#                        EigenTimeseries,
#                        ILSIATimeseries,
#                        CCATimeseries,
#                        FilteredTimeseries,
#                        MeanAndFilteredTimeseries,
#                        EigenAndFilteredTimeseries,
#                        TimeseriesSelectorFactory,)


#from .similarity_measure import (TimeSeriesGroupMeasure,
#                                 CorrelationMeasure,
#                                 NiCorrelationMeasure,
#                                 NiCoherenceMeasure,
#                                 NiGrangerCausalityMeasure,
#                                 SeedCorrelationMeasure,
#                                 MeanSeedCorrelationMeasure,
#                                 SeedCoherenceMeasure,
#                                 MeanSeedCoherenceMeasure,
#                                 CorrelationMeasure,
#                                 SimilarityMeasureFactory)


#from .connectivity import (FunctionalConnectivity)

#__sel_all__ = ['TimeseriesSelector',
#               'MeanTimeseries',
#               'EigenTimeseries',
#               'ILSIATimeseries',
#               'CCATimeseries',
#               'FilteredTimeseries',
#               'MeanAndFilteredTimeseries',
#               'EigenAndFilteredTimeseries', ]


#__sm_all__ = ['SimilarityMeasure',
#              'CorrelationMeasure',
#              'MeanCorrelationMeasure',
#              'CoherenceMeasure',
#              'MeanCoherenceMeasure',
#              'CrossCorrelationMeasure', ]

#__conn_all__ = ['FunctionalConnectivity']

#__all__.extend(__conn_all__)
#__all__.extend(__sel_all__)
#__all__.extend(__sm_all__)

