from collections import OrderedDict, defaultdict
from numpy.testing import assert_array_equal
import numpy as np
import datashader as ds
from datashader.utils import export_image
import pandas as pd
import datashader.utils as du, datashader.transfer_functions as tf
from scipy.spatial import Delaunay
from colorcet import rainbow as c
import dask.dataframe as dd


def rasterize(mesh_nodes, triangles, srf):
    # print("mesh nodes ", mesh_nodes)
    # print("triangles ", triangles)
    # print("srf ", srf)

    assert_array_equal(np.array(list(triangles.keys())), np.array(list(srf.keys())))

    triangles = OrderedDict(triangles)
    srf = OrderedDict(srf)

    cvs = ds.Canvas(plot_height=512, plot_width=512)


    print("triangles.values()", triangles.values())

    verts = pd.DataFrame(np.array(list(mesh_nodes.values()))[:, :2], columns=['x', 'y'])
    tris = pd.DataFrame(triangles.values(), columns=['v0', 'v1', 'v2'])

    # Add point at the begining of the dataframe to deal with different indexing
    verts.loc[-1] = [0, 0]
    verts.index = verts.index + 1
    verts.sort_index(inplace=True)

    tris['rf_value'] = np.squeeze(np.array(list(srf.values())))

    print("tris ", tris)

    trimesh = cvs.trimesh(verts, tris, interpolate='nearest')

    return trimesh
