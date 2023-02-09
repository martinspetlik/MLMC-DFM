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
from datashader.bundling import connect_edges, hammer_bundle
from datashader.layout import random_layout, circular_layout, forceatlas2_layout


def rasterize(mesh_nodes, triangles, cond_tn_elements_triangles,
              lines=None, cond_tn_elements_lines=None, cs_lines=None, n_pixels_x=256,
              save_image=False):

    assert_array_equal(np.array(list(triangles.keys())), np.array(list(cond_tn_elements_triangles.keys())))
    assert_array_equal(np.array(list(lines.keys())), np.array(list(cond_tn_elements_lines.keys())))

    triangles = OrderedDict(triangles)
    srf = OrderedDict(cond_tn_elements_triangles)

    if len(lines) > 0:
        lines = OrderedDict(lines)

    verts = pd.DataFrame(np.array(list(mesh_nodes.values()))[:, :2], columns=['x', 'y'])
    tris = pd.DataFrame(triangles.values(), columns=['v0', 'v1', 'v2'])

    # Set the shape of the canvas
    xr = verts.x.min(), verts.x.max()
    yr = verts.y.min(), verts.y.max()
    canvas = ds.Canvas(x_range=xr, y_range=yr, plot_height=n_pixels_x, plot_width=n_pixels_x)

    # Add point at the begining of the dataframe to deal with different indexing
    verts.loc[-1] = [0, 0]
    verts.index = verts.index + 1
    verts.sort_index(inplace=True)

    tris['rf_value'] = np.squeeze(np.array(list(srf.values())))
    trimesh = canvas.trimesh(verts, tris, interpolate='nearest')

    cvs_lines = np.zeros((n_pixels_x, n_pixels_x))
    if len(lines) > 0:
        cvs_lines = rasterize_lines(canvas, verts, lines, cond_tn_elements_lines,  cs_lines)

    # nearest_img = tf.shade(canvas.trimesh(verts, tris, interpolate='nearest'), name='10 Vertices')
    # export_image(img=nearest_img, filename='mesh_nearest_img', fmt=".png", export_path=".")
    #export_image(img=linear_img, filename='mesh_linear_img', fmt=".png", export_path=".")
    #lines_img = tf.shade(cvs_lines, name='lines', cmap=c)

    # tf.stack(nearest_img, linear_img, how="over")
    #export_image(img=tf.stack(nearest_img, img_points, how="over"), filename='image_trimesh_points', fmt=".png", export_path=".")
    #export_image(img=tf.stack(nearest_img, lines_img, how="over"), filename='dfm_image', fmt=".png", export_path=".")
    return trimesh, cvs_lines


def rasterize_lines(cvs, verts, lines, cond_tn_elements_lines, cs_lines):

    for cs, l_ids in cs_lines.items():
        lines_to_rasterize = {}
        for l_id in l_ids:
            lines_to_rasterize[l_id] = [*lines[l_id], cond_tn_elements_lines[l_id]]

        lines_df = pd.DataFrame(lines_to_rasterize.values(), columns=['source', 'target', 'rf_value'])
        forcedirected = forceatlas2_layout(verts, lines_df)

        cvs_lines = cvs.line(connect_edges(forcedirected, lines_df, weight='rf_value'), 'x', 'y',
                             agg=ds.mean('rf_value'), line_width=cs)

    return cvs_lines
