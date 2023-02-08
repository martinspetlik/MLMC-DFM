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
              lines=None, cond_tn_elements_lines=None, cs_lines=None,
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
    canvas = ds.Canvas(x_range=xr, y_range=yr, plot_height=256, plot_width=256)

    # Add point at the begining of the dataframe to deal with different indexing
    verts.loc[-1] = [0, 0]
    verts.index = verts.index + 1
    verts.sort_index(inplace=True)

    tris['rf_value'] = np.squeeze(np.array(list(srf.values())))
    trimesh = canvas.trimesh(verts, tris, interpolate='nearest')

    if len(lines) > 0:
        cvs_lines = rasterize_lines(canvas, verts, lines, cond_tn_elements_lines,  cs_lines, cmap=c)

    # nearest_img = tf.shade(canvas.trimesh(verts, tris, interpolate='nearest'), name='10 Vertices')
    # export_image(img=nearest_img, filename='mesh_nearest_img', fmt=".png", export_path=".")
    #export_image(img=linear_img, filename='mesh_linear_img', fmt=".png", export_path=".")
    #lines_img = tf.shade(cvs_lines, name='lines', cmap=c)

    # tf.stack(nearest_img, linear_img, how="over")
    #export_image(img=tf.stack(nearest_img, img_points, how="over"), filename='image_trimesh_points', fmt=".png", export_path=".")
    #export_image(img=tf.stack(nearest_img, lines_img, how="over"), filename='dfm_image', fmt=".png", export_path=".")
    return trimesh, cvs_lines


# def custom_stack(trimesh, cvs_lines):
#
#     print("cvs_lines ", cvs_lines)
#     print("trimesh ", trimesh)
#
#     print("cvs lines values ", cvs_lines.values)
#     #print("cvs_lines.coords", np.ndarray(cvs_lines.coords))
#
#     coords_dataset = cvs_lines.coords.to_dataset()
#
#     print("coords_dataset ", type(coords_dataset))
#
#     coords = cvs_lines.coords
#
#     print("coords._data ", coords._data)
#     print("coords._data ", coords._data._coords)
#     print("coords._indexes ", coords._data._indexes)
#
#
#     flatten_values = cvs_lines.values.flatten()
#
#     for index, value in enumerate(flatten_values):
#         x_coord = coords['x'][index]
#         y_coord = coords['y'][index]
#         print("x coord: {}, y coord: {}".format(x_coord, y_coord))
#
#         exit()
#
#
#     print("flatten_values", flatten_values)
#
# #     exit()
#
#     print("coords ", coords)
#     print("coords.x ", coords['x'][10])
#
#     for i in range(len(coords['x'])):
#         x_coord = coords['x'][i]
#         y_coord = coords['y'][i]
#
#         print("x coord: {}, y coord: {}".format(x_coord, y_coord))
#         exit()
#
#     exit()
#
#     for x, y in zip(cvs_lines.coords['x'], cvs_lines.coords['y']):
#         print("x: {}, y: {}".format(x[0], y[0]))
#         exit()
#
#
#     print("cvs_lines.coords[0, 5] ", cvs_lines.coords[0, 5])
#
#     exit()
#
#
#
#
#
#
#
#
#     for x, y in zip(cvs_lines.coords['x'], cvs_lines.coords['y']):
#         print("x: {}, y: {}".format(x, y))
#         print("value ", cvs_lines[x, y])
#         exit()
#     print("cvs_lines")
#     exit()


def rasterize_lines(cvs, verts, lines, cond_tn_elements_lines, cs_lines, cmap):
    # print("lines", lines)
    # print("verts", verts)
    #print("cs_lines ", cs_lines)

    for cs, l_ids in cs_lines.items():
        lines_to_rasterize = {}
        for l_id in l_ids:
            lines_to_rasterize[l_id] = [*lines[l_id], cond_tn_elements_lines[l_id]]

        lines_df = pd.DataFrame(lines_to_rasterize.values(), columns=['source', 'target', 'rf_value'])
        #lines_df['rf_value'] = np.squeeze(np.array(list(cond_tn_elements_lines.values())))

    # # lines_df['rf_value'] = 10
    #
    #
    #
    # # print("verts ", verts.tail())
    # # print("lines ", lines_df.tail())
    # # exit()
    # #lines_raster = cvs.line(lines_df, 'x', 'y')
    #
    # #print("lines_raster ", lines_raster)
    #
        forcedirected = forceatlas2_layout(verts, lines_df)
    # # print("forcedirected ", forcedirected)
    # # exit()
    #
    # # print("verts ", verts)
    # # print("lines_df ", lines_df)
    #
    # #ce = connect_edges(verts, lines_df, weight='rf_value')
    #
    # #print("lines_df ", lines_df)
    # from datashader.reductions import Reduction, _sum_zero, count
    # import xarray as xr
    # class custom_aggregator(Reduction):
    #     """Mean of all elements in ``column``.
    #
    #     Parameters
    #     ----------
    #     column : str
    #         Name of the column to aggregate over. Column data type must be numeric.
    #         ``NaN`` values in the column are skipped.
    #     """
    #
    #     def _build_bases(self, cuda=False):
    #         return (_sum_zero(self.column), count(self.column))
    #
    #     @staticmethod
    #     def _finalize(bases, cuda=False, **kwargs):
    #         sums, counts = bases
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             x = np.where(counts > 0, sums / counts, np.nan)
    #         return xr.DataArray(x, **kwargs)

        connected_edges = connect_edges(forcedirected, lines_df, weight='rf_value')
        #print("connected edges ", connected_edges)

        cvs_lines = cvs.line(connect_edges(verts, lines_df, weight='rf_value'),
                            'x', 'y', agg=ds.mean('rf_value'), line_width=cs)

        #print("cvs line ", cvs_lines.values)

        # for i in range(cvs_lines.values.shape[0]):
        #     for j in range(cvs_lines.values.shape[1]):
        #         if not np.isnan(cvs_lines.values[i][j]):
        #             print(cvs_lines.values[i][j])
        #
        # exit()

    #lines_img = tf.shade(cvs_line, name='lines', cmap=cmap)

    #points_img = tf.shade(cvs.points(pd.DataFrame(lines.values(), columns=['x', 'y']), 'x', 'y'))

    #print("lines_img ")

    # print("type(nearest_img)", type(nearest_img))
    # print("nearest_img.to_numpy().shape ", nearest_img.to_numpy().shape)
    #np.save("numpy_image", nearest_img.to_numpy())

    # rgb_image = np.array(nearest_img.to_pil())[..., :3]
    # print("rgb_image ", rgb_image)
    #
    # np.save("rgb_image", rgb_image)
    # print("rgb_image.shape ", rgb_image.shape)
    #
    # linear_img = tf.shade(cvs.trimesh(verts, tris, interpolate='linear'), cmap=c, name='10 Vertices Interpolated')

    # export_image(img=lines_img, filename='lines_img', fmt=".png", export_path=".")
    #
    # print("ce ", ce)
    #
    # print("lines_img ", lines_img)
    # print("lines_img.coord ", lines_img.coords)

    #export_image(img=points_img, filename='points_img', fmt=".png", export_path=".")

    #export_image(img=linear_img, filename='mesh_linear_img', fmt=".png", export_path=".")

    return cvs_lines#, lines_img
