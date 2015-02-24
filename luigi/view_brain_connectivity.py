#brain on drops

import os
import numpy as np
import nibabel as nib
from collections import OrderedDict

from mayavi import mlab

from ..nifti.roi import get_rois_centers_of_mass

def test_quiver3d():
    x, y, z = np.mgrid[-2:3, -2:3, -2:3]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    obj = mlab.quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
    return obj


def get_random_color():
    return tuple(np.random.rand(3))


@mlab.show
def show_brain_connectivity_on_atlas(vol, atlas_vol, colors={}, colormap=None,
                                     weights=None, sizes={}, connections=None,
                                     contour_intensity_threshold=0.4):

    rois_centers = get_rois_centers_of_mass(atlas_vol)

    show_brain_connectivity(vol, rois_centers, colors, colormap, weights, sizes,
                            connections, contour_intensity_threshold)


@mlab.show
def show_brain_connectivity(vol, rois_centers, colors={}, colormap=None,
                            weights=None, sizes={}, connections=None,
                            contour_intensity_threshold=0.4):
    """

    :param vol: ndarray
    3D brain image

    :param rois_centers: ndarray
    of shape Nx3 where N is the number of ROIs, and
    for each ROI there is a 3D position vector.

    :param colors: list of tuples
    one tuple for each ROI indicating their colors.
    (0. ,0., 0.) for black and (1., 1., 1.) for white.
    If None, will see if colormap has been given,
    or will plot yellow spheres.

    :params colormap: matplotlib colormap LUT
    If color param is None, this can be used to select ROI colors.
    For a sensible result, weights vector must be given.

    :params weights: vector
    Vector of floats with size N.

    :params sizes: vector of float
    Vector with size N
    Indicating the size of each drop

    :param connections: ndarray
    This array can have two different shapes:
    - Connectivity matrix: shape NxN, where each component
    is a connection weight.

    - Connection pairs: 2xN, where each pair is the
    indication what two ROIs is connected.


    :param contour_intensity_threshold: float
    This will indicate the contour process of the
    vol surface the voxel intensity threshold
    to use to select where to perform the contour.

    """
    n_rois = len(rois_centers)

    if colors is not None:
        assert(n_rois == len(colors))

    if connections is not None:
        if connections.shape[0] != 2:
            assert(n_rois == connections.shape[0] == connections.shape[1])

    #plot brain contour
    src = mlab.pipeline.scalar_field(vol)
    mlab.pipeline.iso_surface(src, contours=[vol.min()+contour_intensity_threshold*vol.ptp(), ],
                              opacity=0.1)
    #mlab.pipeline.volume(src)

    #plot drops
    all_rois = np.array(rois_centers.keys())

    #mlab.pipeline.iso_surface(src, contours=[vol.max()-0.1*vol.ptp(), ],)
    ridx = 0
    for rval in rois_centers:
        c = rois_centers[rval]

        params = {}
        params['color'] = colors.get(rval, (0.5, 0.5, 0))
         #[rval] if colors is not None else (0.5, 0.5, 0)
        params['scale_factor'] = sizes.get(rval, 1)

        points = mlab.points3d(c[0], c[1], c[2],
                               resolution=20,
                               scale_mode='none',
                               **params)
        ridx += 1

    if connections is not None:
        #if connectivity matrix, will transform it in array of pairs of indices
        if isinstance(connections, np.ndarray):
            if connections.shape == (n_rois, n_rois):
                conns = np.array(np.where(connections > 0))
                weights = connections[connections > 0]
        #else, an array of pairs of ROI values has been given
        else:
            n_links = connections.shape[1]
            #transform it in pairs of indices
            conns = np.zeros((2, n_links))
            for rpidx in list(range(n_links)):
                conns[0, rpidx] = np.where(all_rois == connections[0, rpidx])[0][0]
                conns[1, rpidx] = np.where(all_rois == connections[1, rpidx])[0][0]

        for pidx in list(range(conns.shape[1])):
            pair = conns[:, pidx]
            if pair[0] != pair[1]:
                #which rois?
                r1val = all_rois[pair[0]]
                r2val = all_rois[pair[1]]

                #tube color
                #this average color works if they are of different shade,
                #otherwise, you should use Lab color space.
                r1color = colors.get(r1val, (1, 1, 1))
                r2color = colors.get(r2val, (1, 1, 1))
                tube_color = tuple(np.mean([r1color, r2color], axis=0))

                #extreme points coordinates
                roi1coord = rois_centers[r1val]
                roi2coord = rois_centers[r2val]

                coords = np.concatenate([roi1coord, roi2coord])
                x = coords[0:4:3]
                y = coords[1:5:3]
                z = coords[2:6:3]
                w = weights[pidx]
                #mlab.flow

                mlab.plot3d(x, y, z, tube_radius=0.5, tube_sides=6,
                            color=tube_color)

    mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=10,
                            colormap='gray')

    return 0

if __name__ == '__main__':
    wd = '/home/alexandre/Dropbox/Documents/work/cobre/'

    atlas = os.path.join(wd, 'aal_2mm.nii.gz')
    anat = os.path.join(wd, 'MNI152_T1_2mm_brain.nii.gz')

    atlas_vol = nib.load(atlas).get_data()
    anat_vol = nib.load(anat).get_data()

    roisvals = np.unique(atlas_vol)
    roisvals = roisvals[roisvals != 0]
    n_rois = len(roisvals)

    rois_linspace = np.linspace(0.2, 1, n_rois)

    #rois_centers = get_rois_centers_of_mass(atlas_vol)

    idx = 0
    sizes = OrderedDict()
    colors = OrderedDict()
    for r in roisvals:
        colors[r] = (0, rois_linspace[idx], 0)
        sizes[r] = 10*rois_linspace[idx]
        idx += 1

    #connections = np.random.randint(0, 2, (n_rois, n_rois))
    connections = np.random.choice([0, 1], size=(n_rois, n_rois),
                                   p=[99.7/100, 0.3/100])

    show_brain_connectivity_on_atlas(anat_vol, atlas_vol, colors,
                                     connections=connections)

