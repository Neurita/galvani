# // ====================================================================
# // This file is part of the Endmember Induction Algorithms Toolbox for MATLAB 
# // Copyright (C) Grupo de Inteligencia Computacional, Universidad del 
# // Pais Vasco (UPV/EHU), Spain, released under the terms of the GNU 
# // General Public License.
# //
# // Endmember Induction Algorithms Toolbox is free software: you can redistribute 
# // it and/or modify it under the terms of the GNU General Public License 
# // as published by the Free Software Foundation, either version 3 of the 
# // License, or (at your option) any later version.
# //
# // Endmember Induction Algorithms Toolbox is distributed in the hope that it will
# // be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# // of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# // General Public License for more details.
# //
# // You should have received a copy of the GNU General Public License
# // along with Endmember Induction Algorithms Toolbox. 
# // If not, see <http://www.gnu.org/licenses/>.
# // ====================================================================
#

# Manuel Grana <manuel.grana[AT]ehu.es>
# Miguel Angel Veganzones <miguelangel.veganzones[AT]ehu.es>
# Alexandre Manhaes Savio <alexandre.manhaes[AT]ehu.es>
# Grupo de Inteligencia Computacional (GIC), Universidad del Pais Vasco /
# Euskal Herriko Unibertsitatea (UPV/EHU)
# http://www.ehu.es/computationalintelligence
# 
# Copyright (2013) Grupo de Inteligencia Computacional @ Universidad del Pais Vasco, Spain.
#
# Python translation made by Alexandre Savio.
#
#
# Bibliographical references:
# [1] M. Grana, D. Chyzhyk, M. Garcia-Sebastian, y C. Hernandez,
# "Lattice independent component analysis for functional magnetic resonance imaging",
# Information Sciences, vol. 181, n. 10, p. 1910-1928, May. 2011.

# Now, a description of the algorithms is offered together to the options
# that can be passed as parameters.
#
# * ILSIA: Incremental lattice Source Induction Algorithm (ILSIA) endmembers induction algorithm.
#   - Bibliographical references:
#       [1] M. Grana, D. Chyzhyk, M. Garcia-Sebastian, y C. Hernandez,
# "Lattice independent component analysis for functional magnetic resonance imaging",
# Information Sciences, vol. 181, n. 10, pags. 1910-1928, May. 2011.
#   - Options: {'alpha'}
#       'alpha': Chebyshev-best approximation tolerance threshold (>= 0). Default = 0.
#
# * EIHA: Endmember induction heuristic algorithm (EIHA) endmembers induction algorithm.
#   - Bibliographical references:
#       [1] M. Grana, I. Villaverde, J. O. Maldonado, y C. Hernandez,
# "Two lattice computing approaches for the unsupervised segmentation of hyperspectral images",
# Neurocomput., vol. 72, n. 10-12, pags. 2111-2120, 2009.
#   - Options: {'alpha'}
#       'alpha': perturbation tolerance. Default = 2.
#
# * WM: Prof. Ritter's WM endmembers induction algorithm.
#   - Bibliographical references:
#       [1] G. X. Ritter y G. Urcid,
# "A lattice matrix method for hyperspectral image unmixing",
# Information Sciences, vol. In Press, Corrected Proof, Oct. 2010.
#   - Options: none.
#
# * NFINDR: N-FINDR endmembers induction algorithm.
#   - Bibliographical references:
#       [1] Winter, M. E.,
# "N-FINDR: an algorithm for fast autonomous spectral end-member determination in hyperspectral data",
# presented at the Imaging Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pags. 266-275.
#   - Options: {'p'|'maxit'}
#       'p': number of endmembers to be induced. If not provided it is calculated by HFC method with tol=10^(-5).
#       'maxit': maximum number of iterations. Default = 3*p.
#
# * FIPPI: Fast Iterative Pixel Purity Index (FIPPI) endmembers induction algorithm.
#   - Bibliographical references:
#       [1] Chang, C.-I.,
# "A fast iterative algorithm for implementation of pixel purity index",
# Geoscience and Remote Sensing Letters, IEEE, vol. 3, n. 1, pags. 63-67, 2006.
#   - Options: {'p'|'maxit'}
#       'p': number of endmembers to be induced. If not provided it is calculated by HFC method with tol=10^(-5).
#       'maxit': maximum number of iterations. Default = 3*p.
#
# * ATGP: ATGP endmembers induction algorithm.
#   - Bibliographical references:
#       [1] A. Plaza y C.-I. Chang,
# "Impact of Initialization on Design of Endmember Extraction Algorithms",
# Geoscience and Remote Sensing, IEEE Transactions on, vol. 44, n. 11, pags. 3397-3407, 2006.
#   - Options: {'p'}
#       'p': number of endmembers to be induced. If not provided it is calculated by HFC method with tol=10^(-5).
#

# classes implemented here:
# LAM
# ILSIA
# EIHA
# WM
# NFINDR

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import logging
import numpy as np

log = logging.getLogger(__name__)


class LAM(object):
    """The Lattice Associative Memories is a kind of associative memory working
    over lattice operations. If X=Y then W and M are Lattice AutoAssociative
    Memories (LAAM), otherwise they are Lattice HeteroAssociative Memories
    (LHAM).

    Parameters
    ----------
    X: numpy.ndarray
        input pattern matrix [n_samples x n_variables]

    Y: numpy.ndarray
        output pattern matrix [n_samples x m_variables]

    Returns
    -------
    W: numpy.ndarray
        dilative LAM [m_variables x n_variables]

    M: numpy.ndarray
        erosive LAM [m_variables x n_variables]

    Notes
    -----
    Bibliographical references:
    [1] M. Grana, D. Chyzhyk, M. Garcia-Sebastian, y C. Hernandez, \
    "Lattice independent component analysis for functional magnetic resonance imaging", 
    Information Sciences, vol. 181, n. 10, pags. 1910-1928, May. 2011.
    """

    def __init__(self, X=None, Y=None):

        if X is None:
            X = []

        if Y is None:
            Y = []

        if len(X) > 0 and len(Y) == 0:
            Y = X

        self._X = np.array(X)
        self._Y = np.array(Y)

        #Checking data size
        x_samps, x_vars = self._X.shape
        y_samps, y_vars = self._Y.shape
        if x_samps != y_samps:
            err = 'Input and output dimensions mismatch'
            log.error(err)
            self.clear()
            raise ValueError(err)

    def clear(self):
        self.__init__()

    def fit(self):
        """Compute vector lattice external products in self.W_ and self.M_"""
        x_samps, x_vars = self._X.shape
        y_samps, y_vars = self._Y.shape

        self.W_ = np.zeros(y_vars, x_vars)
        self.M_ = np.zeros(y_vars, x_vars)

        #some lattice operations
        for b in range(y_vars):
            for c in range(x_vars):
                product      = self._Y[:, b] - self._X[:, c]
                self.W_[b, c] = product.min()
                self.M_[b, c] = product.max()

        return self.W_, self.M_


#ts = np.random.randn(1000,200)
#%timeit data = ts; X_z = _standardize_data(data)
#100 loops, best of 3: 8.88 ms per loop
def _standardize(data):
    n_samps, n_feats = data.shape()

    data_mean    = np.tile(data.mean(axis=0), (n_samps, 1))
    data_std     = np.tile(data.std (axis=0), (n_samps, 1))

    data_std[data_std == 0.0] = 1.0

    return (data - data_mean) / data_std


#%timeit data = ts; X_z = _standardize_data_with_scaler(data)
#100 loops, best of 3: 7.36 ms per loop
def _standardize_with_scaler(data):
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return _standardize(data)

    scaler = StandardScaler()

    return scaler.fit_transform(data)


class ILSIA(object):
    """Incremental lattice Source Induction Algorithm (ILSIA) endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [n_samples x n_variables]

    alpha: float
        Chebyshev-best approximation tolerance threshold (>= 0). Default = 0.

    Returns
    -------
    em: numpy.ndarray
        set of induced endmembers [p x n_variables]

    cnt: numpy.array
        induced endmembers indexes vector [n_samples] with {0,1} \
        values, where '1' indicates that the corresponding sample has \
        been identified as an endmember.

    Notes
    -----
    Bibliographical references:
    [1] M. Grana, D. Chyzhyk, M. Garcia-Sebastian, y C. Hernandez, \
    "Lattice independent component analysis for functional magnetic resonance imaging", \
    Information Sciences, vol. 181, n. 10, pags. 1910-1928, May. 2011.
    """

    def __init__(self, data, alpha=0):

        self._data  = np.array(data)
        self._alpha = alpha
        self._is_standardized = False

    def shape(self):
        return self._data.shape

    def fit(self):
        from numpy.linalg import norm
        from scipy.spatial.distance import chebyshev

        if not self._is_standardized:
            self._data_z_ = _standardize_with_scaler(self._data)
            self._is_standardized = True

        n_samps, n_feats = self.shape()

        em  = [] #endmembers
        cnt = np.zeros(n_samps)

        # Initial LIS
        lis = [] #lattice independent sources
        idx = np.random.randint(0, n_samps)
        p = 1 #number of current endmembers

        # Algorithm Initialization
        # Initialize endmembers set and index vector
        cnt[idx] = 1
        samp = self._data_z_[idx, :]
        lis.append(samp)

        is_new_lis = True

        #data signs
        signs = []
        signs.append(np.sign(samp))

        #saving endmembers
        em.append(self._data[idx, :])

        #indicates wich pixels is identified as an endmember
        idxs = []
        idxs.append(idx)

        #Run over each sample
        for i in range(n_samps):
            #check for LAAM recalculation
            if is_new_lis:
                #recalculate LAAM
                laam = LAM(lis)
                wxx, mxx = laam.fit()
                is_new_lis = False

            #sample
            samp      = self._data[i, :]
            samp_sign = np.sign(samp)

            if np.sum(np.abs(samp)) > 0:
                if self._alpha <= 0:
                    #check if pixel is lattice dependent
                    #y = np.zeros(n_feats)

                    #vector version
                    samps = np.tile(samp, (n_feats, 1))
                    y = np.max(wxx + samps, axis=1)
                    #for loop version
                    #for j in range(n_feats):
                    #    y[j] = np.max(wxx[:,j] + samp)

                    #find the most similar and check the norms
                    sum_signs   = 0
                    selected_em = 0
                    for e in range(p):
                        asigns = np.array(signs)
                        sum_signs_em = np.sum(asigns[e, :] == samp_sign)
                        if sum_signs_em > sum_signs:
                            sum_signs   = sum_signs_em
                            selected_em = e

                        alis = np.array(lis)
                        if norm(alis[selected_em, :]) < norm(samp):
                            #substitute lis
                            new_lis = True
                            cnt  [i] = 1
                            cnt  [idx[selected_em]] = 0
                            idx  [selected_em] = i
                            lis  [selected_em] = samp
                            signs[selected_em] = np.sign(samp)
                            em   [selected_em] = self._data[i, :]

                        continue

                #end if self._alpha <= 0
                else:
                    # Chebyshev-Best approximation
                    x_sharp  = np.zeros(n_feats)
                    wxx_conj = -wxx

                    for j in range(n_feats):
                        x_sharp[j] = np.min(wxx_conj[:, j] + samp)
                        mu = np.max(wxx[:, j] + x_sharp)
                        mu = np.max(mu + samp)/2

                    c1 = np.zeros(n_feats)
                    for j in range(n_feats):
                        c1[j] = np.max(wxx[:, j] + mu + x_sharp)

                    if chebyshev(c1, samp) < self._alpha:

                        #find the most similar and check the norms
                        sum_signs   = 0
                        selected_em = 0
                        for e in range(p):
                            asigns = np.array(signs)
                            sum_signs_em = np.sum(asigns[e, :] == samp_sign)
                            if sum_signs_em > sum_signs:
                                sum_signs   = sum_signs_em
                                selected_em = e

                        alis = np.array(lis)
                        if norm(alis[selected_em, :] < norm(samp)):
                            # substitute LIS
                            is_new_lis = True
                            cnt  [i] = 1
                            cnt  [idx[selected_em]] = 0
                            idx  [selected_em] = i
                            lis  [selected_em] = samp
                            signs[selected_em] = np.sign(samp)
                            em   [selected_em] = self._data[i, :]

                        continue

                #Max-Min dominance
                mu1 = 0
                mu2 = 0
                for j in range(1, p+2):
                    s1 = np.zeros(n_feats)
                    s2 = np.zeros(n_feats)
                    for k in range(1, p+2):
                        if j != k:
                            if j == p+1: vi = samp
                            else:        vi = lis[j]

                            if k == p+1: vk = samp
                            else:        vk = lis[k]

                            d = vi - vk
                            m1 = np.max(d)
                            m2 = np.min(d)
                            s1 = s1 + (d == m1)
                            s2 = s2 + (d == m2)

                    mu1 = mu1 + (np.max(s1) == p)
                    mu2 = mu2 + (np.max(s2) == p)

                if mu1 == (p+1) or mu2 == (p+1):
                    #new lis
                    p += 1
                    cnt[i] = 1
                    idxs.append(1)
                    lis.append(samp)
                    signs.append(samp_sign)
                    em.append(self._data[i, :])

        self.em_  = np.array(em)
        self.cnt_ = cnt

        return self.em_, self.cnt_


class EIHA(object):
    """Endmember induction heuristic algorithm (EIHA) endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables]

    alpha: float
        perturbation tolerance. Default = 2

    Returns
    -------
    em: numpy.ndarray
        set of induced endmembers [p x nvariables]

    cnt: numpy.ndarray
        induced endmembers indexes vector [nsamples] with {0,1} values, where '1' indicates that
        the corresponding sample has been identified as an endmember.

    Notes
    -----
    Bibliographical references:
    [1] M. Grana, I. Villaverde, J. O. Maldonado, y C. Hernandez, 
    "Two lattice computing approaches for the unsupervised segmentation of hyperspectral images", 
    Neurocomputing, vol. 72, n. 10-12, pags. 2111-2120, 2009.
    """

    def __init__(self, data, alpha=2):

        self._data  = np.array(data)
        self._alpha = alpha
        self._is_standardized = False

    def shape(self):
        return self._data.shape

    def fit(self):

        from numpy.linalg import norm

        if not self._is_standardized:
            self._data_z_ = _standardize_with_scaler(self._data)
            self._is_standardized = True

        n_samps, n_feats = self.shape()
        mu_data  = self._data.mean(axis=0)
        std_data = self._data.std (axis=0)

        #Initialization
        p       = 1
        extrems = []
        signs   = []
        idxs    = []
        samp    = self._data[0, :]
        idx     = np.random.randint(0, n_samps)

        idxs.append   (idx)
        extrems.append(samp)
        signs.append  (np.sign(samp - mu_data))

        #Algorithm
        for i in range(n_samps):
            samp = self._data[i, :]
            if np.sum(np.abs(samp)) > 0:
                #perturbations
                samp_plus  = samp + self._alpha * std_data
                samp_minus = samp - self._alpha * std_data

                #flag pixel extremness
                new_extreme = 1

                #erosive and dilative independence
                erosive_indep  = np.ones(p)
                dilative_indep = np.ones(p)

                #Check if samp is in the same quadrant than any of the 
                #already selected endmembers
                for e in range(p):
                    if signs[e] == np.sign(samp - mu_data):
                        new_extreme = 0
                        if norm(samp - mu_data) > norm(extrems[e] - mu_data):
                            extrems[e] = samp
                            idxs   [e] = i
                        break

                #If samp is in the same quadrant than any of the already
                # selected endmember (new_extreme == 1) then check perturbations
                for k in range(p):
                    if np.power(samp - extrems[k], 2) < self._alpha * std_data:
                        new_extreme = 0
                        break
                    if extrems[k] > samp_minus:
                        dilative_indep[k] = 0
                    elif extrems[k] < samp_plus:
                        erosive_indep[k] = 0

                #check if there is erosive or dilative independence
                independence = False
                if   dilative_indep.any(): independence = True
                elif erosive_indep.any() : independence = True

                #Check if there is new extreme
                if new_extreme and independence:
                    p += 1
                    extrems.append(samp)
                    signs.append(np.sign(samp - mu_data))
                    idxs.append(i)

        self.em_  = np.zeros(p, n_feats)
        self.cnt_ = np.zeros(n_samps)
        for i in range(p):
            self.em_[i, :] = self._data[idxs[i],:]
            self.cnt_[idxs[i]] = 1

        return self.em_, self.cnt_


class WM(object):
    """Prof. Ritter's WM endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables]

    Returns
    -------
    em: numpy.ndarray
        set of induced endmembers [p x nchannels]

    Notes
    -----
    Bibliographical references:
    [1] G. X. Ritter y G. Urcid, 
    "A lattice matrix method for hyperspectral image unmixing", 
    Information Sciences, vol. In Press, Corrected Proof, Oct. 2010.
    """

    def __init__(self, data):
        self._data = np.array(data)

    def shape(self):
        return self._data.shape

    def fit(self):
        #hyperbox corners
        u = np.max(self._data, axis=0)
        v = np.min(self._data, axis=0)

        #LAMs
        lam = LAM(self._data)
        wxx, mxx = lam.fit()

        #polytope
        r_w, c_w = wxx.shape
        for i in range(c_w):
            wxx[:,i] += u[i]
            mxx[:,i] += v[i]

        #return the set of induced endmembers
        em = np.zeros(2*c_w + 2, r_w)
        em[0:c_w-1    , :] = wxx
        em[c_w:2*c_w-1, :] = mxx
        em[2*c_w      , :] = u
        em[2*c_w+1    , :] = v

        self.em_  = em
        self.cnt_ = []

        return self.em_


class HFC(object):
    """
    Virtual dimensionality by HFC method
    Computes the vitual dimensionality (VD) measure for an HSI
    image for specified false alarm rates.  When no false alarm rate(s) is
    specificied, the following vector is used: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5.
    This metric is used to estimate the number of materials in an HSI scene.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables]

    far: list or numpy.array
        list of false alarm probabilities [1 x p].
        Default: [10**-3, 10**-4, 10**-5]

    Returns
    -------
    vd: numpy.ndarray
        vector of virtual dimensionality values [1 x p]

    Notes
    -----
    Bibliographical references:
    [1] Chang, C.-I. and Du, Q., "Estimation of number of spectrally distinct signal sources in hyperspectral imagery"
        Geoscience and Remote Sensing, IEEE Transactions on,  vol. 42, 2004, pp. 608-619.
    [2] Wang, J. and Chang, C.-I., "Applications of Independent Component Analysis in Endmember Extraction and Abundance Quantification for Hyperspectral Imagery"
        Geoscience and Remote Sensing, IEEE Transactions on,  vol. 44, 2006, pp. 2601-2616.
    [3] J. Wang and Chein-I Chang, "Independent component analysis-based dimensionality reduction with applications in hyperspectral image analysis"
        Geoscience and Remote Sensing, IEEE Transactions on,  vol. 44, 2006, pp. 1586-1600.
    """

    def __init__(self, data, far=[10**-3, 10**-4, 10**-5]):
        self._data = np.array(data)
        if isinstance(far, list):
            self._far = far
        else:
            self._far = [far]

    def shape(self):
        return self._data.shape

    def fit(self):
        from numpy.linalg import eig
        import scipy as sp
        import scipy.stats as stats

        n_samps, n_feats = self.shape()

        #Cross-Correlation and covariance matrix
        #Eigenvalues
        lcorr = eig (np.corrcoef(self._data.T))[0][::-1]
        lcov  = eig (np.cov     (self._data.T))[0][::-1]

        ems = []
        for i in range(len(self._far)):
            n_ems = 0
            pf    = self._far[i]
            for j in range(n_feats):
                sigma_sqr = (2*lcov[j]/n_samps) + (2*lcorr[j]/n_samps) + (2/n_samps) * lcov[j] * lcorr[j]
                sigma = sp.sqrt(sigma_sqr)

                print(sigma)
                # stats.norm.ppf not valid with sigma
                # using the module of the complex number : abs(sigma)
                tau = -stats.norm.ppf(pf, 0, abs(sigma))
                if (lcorr[j]-lcov[j]) > tau: 
                    n_ems += 1

            ems.append(n_ems)

        self.vd_  = ems

        return self.vd_


def _PCA_transform(data, n_components):
    from sklearn.decomposition import PCA
    # data.shape: (nsamples x nvariables)
    pca = PCA(n_components=n_components)
    data_trans = pca.fit_transform(data)
    return data_trans


class NFINDR(object):
    """N-FINDR endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables]

    p: int
        number of endmembers to be induced.
        If not provided it is calculated by HFC method with tol=10**-5

    maxit: int
        maximum number of iterations. Default = 3*p

    Returns
    -------
    em        : set of induced endmembers [p x nvariables]

    cnt       : induced endmembers indexes vector [nsamples] with {0,1}
                         values, where '1' indicates that the corresponding sample 
                         has been identified as an endmember.

    idxs      : array of indices into the array data corresponding to the
                         induced endmembers

    Notes
    -----
    Bibliographical references:
    [1] Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral end-member determination in hyperspectral data", 
    presented at the Imaging Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pags. 266-275.
    """

    def __init__(self, data, p=-1, maxit=-1):
        if p <= 0:
            hfc = HFC(data, [10**-5])
            p = hfc.fit()[0]

        if maxit <= 0:
            maxit = 3*p

        self._data  = np.array(data)
        self._p     = p
        self._maxit = maxit

    def shape(self):
        return self._data.shape

    def fit(self):
        n_samps, n_feats = self.shape()

        p = self._p

        #Dimensionality reduction by PCA
        data_pca = _PCA_transform(self._data, p - 1)

        #Initialization
        cnt  = np.zeros(n_samps)
        idxs = np.zeros(p, dtype=np.int)

        test_matrix = np.zeros((p, p), dtype=np.float32)
        test_matrix[0,:] = 1
        for k in range(p):
            idx = np.random.randint(0, n_samps)
            test_matrix[1:p, k] = data_pca[idx, :]
            idxs[k] = idx

        actual_volume = np.abs(np.linalg.det(test_matrix))
        it =  1
        v1 = -1
        v2 = actual_volume

        #Algorithm
        while it <= self._maxit and v2 > v1:
            for k in range(p):
                for i in range(n_samps):
                    actual_sample       = test_matrix[1:p, k]
                    test_matrix[1:p, k] = data_pca[i,:]
                    volume = np.abs(np.linalg.det(test_matrix))

                    if volume > actual_volume:
                        actual_volume = volume
                        idxs[k] = i
                    else:
                        test_matrix[1:p,k] = actual_sample
            it += 1
            v1  = v2
            v2  = actual_volume

        em = np.zeros((len(idxs), n_feats), dtype=np.float32)
        for i in range(p):
            em [i,:]     = self._data[idxs[i],:]
            cnt[idxs[i]] = 1

        self.em_   = em
        self.cnt_  = cnt
        self.idxs_ = idxs

        return self.em_, self.cnt_, self.idxs_


class CCA(object):
    """Convex Cone Analysis (CCA) endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables].

    p: int
        number of endmembers to be induced.
        If not provided it is calculated by HFC method with tol=10^(-5).

    t: float
        tolerance for numerical errors. By default 10^(-6).

    Returns
    -------
    em: numpy.ndarray
        set of induced endmembers [nchannels x p]

    Notes
    -----
    Bibliographical references:
    [1] Ifarraguerri, A., "Multispectral and hyperspectral image analysis with convex cones",
    Geoscience and Remote Sensing, IEEE Transactions on, vol. 37, n. 2, pags. 756-770, 1999.
    """

    def __init__(self, data, p=-1, t=10**-6):
        if p <= 0:
            hfc = HFC(data, [10**-5])
            p = hfc.fit()

        self._data  = np.array(data)
        self._p     = p
        self._t     = t
        self._is_standardized = False

    def shape(self):
        return self._data.shape

    def fit(self):
        from numpy.linalg import eig

        if not self._is_standardized:
            self._data_z_ = _standardize_with_scaler(self._data)
            self._is_standardized = True

        n_samps, n_feats = self.shape()
        p = self._p

        #Correlation matrix and eigen decomposition
        # calculate eigenvalues of covariance and correlation between bands
        v, d = eig(np.corrcoef(self._data_z_))[0]
        v = v[n_feats-p:n_feats, :]

        #Algorithm
        #num of selected endmembers
        num_em = 0
        #init endmembers set
        em  = np.zeros((p, n_samps))

        while num_em < p:
            #boolean vector indicating which bands are being selected
            selector = np.zeros(n_samps)

            #select p-1 bands randomly
            n_selected = 0
            while n_selected < p-1:
                i = np.random.randint(0, n_samps)
                if not selector[i]:
                    selector[i]  = 1
                    n_selected  += 1

            selector = np.where(selector)

            #equation system
            ps  = np.zeros((p-1, p-1))
            p1s = np.zeros(p-1)
            for i in range(p-1):
                ps [i, :] = v[1:p-1, selector[i]]
                p1s[i]    = v[    p, selector[i]]

            sol = np.linalg.solve(ps, p1s)
            xs  = np.dot(v, np.append(sol, 1))
            if xs.min() > -self._t:
                em[num_em, :] = xs
                num_em += 1

        self.em_ = em

        return self.em_


class ATGP(object):
    """ATGP endmembers induction algorithm.

    Parameters
    ----------
    data: numpy.ndarray
        column data matrix [nsamples x nvariables]

    p: numpy.ndarray
        number of endmembers to be induced.
        If not provided it is calculated by HFC method with tol=10^(-5).

    Returns
    -------
    E: numpy.ndarray
        set of induced endmembers [p x nvariables]

    C: numpy.array
        induced endmembers indexes vector [nsamples] with {0,1} values, where '1' indicates that the
        corresponding sample has been identified as an endmember.

    Notes
    -----
     Bibliographical references:
     [1] A. Plaza y C.-I. Chang, 
    "Impact of Initialization on Design of Endmember Extraction Algorithms", 
    Geoscience and Remote Sensing, IEEE Transactions on, vol. 44, n. 11, p. 3397-3407, 2006.
    """

    def __init__(self, data, p=-1):
        if p <= 0:
            hfc = HFC(data, [10**-5])
            p = hfc.fit()

        self._data = np.array(data)
        self._p    = p

    def shape(self):
        return self._data.shape

    def fit(self):

        n_samps, n_feats = self.shape()
        p = self._p

        #Algorithm initialization
        #the sample with max energy is selected as the initial endmember
        mymax = -1
        idx   = 0
        for i in range(n_samps):
            r = self._data[i, :]
            val = np.dot(r, r)
            if val > mymax:
                mymax = val
                idx   = i

        #Initialization of the set of endmembers and the endmembers index vector
        em       = np.zeros((p, n_feats))
        em[0, :] = self._data[idx, :]
        cnt      = np.zeros(n_feats)
        cnt[idx] = 1

        #generate the identity matrix
        eye  = np.eye(n_feats)
        idxs = [idx]

        for i in range(p-1):
            uc = em[0:i+1, :]
            # Calculate the orthogonal projection with respect to the pixels
            # at present chosen.
            # This part can be replaced with any other distance
            pu = eye - np.dot(uc.T, np.dot(np.linalg.pinv(np.dot(uc, uc.T)),
                                           uc))
            mymax = -1
            idx   = 0
            # Calculate the most different pixel from the already selected
            # ones according to the orthogonal projection (or any other
            # distance selected)
            for j in range(n_samps):
                r = self._data[j, :]
                result = np.dot(pu.T, r)
                val    = np.dot(result, result.T)
                if val > mymax:
                    mymax = val
                    idx   = j
            # The next chosen pixel is the most different from the
            #  already chosen ones
            em[:, i+1] = self._data[idx, :]
            cnt[idx] = 1
            idxs.append(idx)

        self.em_= em
        self.cnt_ = cnt
        self.idxs_ = idxs

        return self.em_, self.cnt_
