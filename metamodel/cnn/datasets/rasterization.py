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

    ##
    # Dealing with NaN values - caused sometimes by trimesh (observed for large number of pixels e.g. 256x256 and small number of mesh elements e.g. 20)
    ##
    coord_x, coord_y = np.where(np.isnan(trimesh) == True)
    if np.any(np.isnan(trimesh)):
        neighbour_pixels_x = [-1, 0, 1]
        neighbour_pixels_y = [-1, 0, 1]
        grid_x, grid_y = np.meshgrid(neighbour_pixels_x, neighbour_pixels_y)
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
        for x,y in zip(coord_x, coord_y):
            neigh_value_sum = 0
            i = 0
            for x_n, y_n in zip(grid_x, grid_y):
                if x_n == 0 and y_n == 0:
                    continue
                if 0 <= x-x_n < trimesh.shape[0] and 0 <= y-y_n < trimesh.shape[1]:
                    if not np.isnan(trimesh[x-x_n, y-y_n].values):
                        neigh_value_sum += trimesh[x-x_n, y-y_n].values
                        i += 1
            neigh_value = neigh_value_sum / i
            trimesh[x,y] = neigh_value

    cvs_lines = np.full((n_pixels_x, n_pixels_x), np.nan)
    if len(lines) > 0:
        cvs_lines = rasterize_lines(canvas, verts, lines, cond_tn_elements_lines,  cs_lines)

    nearest_img = tf.shade(canvas.trimesh(verts, tris, interpolate='nearest'), name='10 Vertices')
    export_image(img=nearest_img, filename='mesh_nearest_img', fmt=".png", export_path=".")

    #linear_img = tf.shade(canvas.trimesh(verts, tris, interpolate='linear'), name='10 Vertices')
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
