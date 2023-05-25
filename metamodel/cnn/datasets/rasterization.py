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


class Rasterization():

    def __init__(self):
        self._lines_df = {}
        self._verts = None
        self._tris = None
        self._lines_set = False


    def _prepare_verts_tris(self, mesh_nodes, triangles, cond_tn_elements_triangles, lines=None, cond_tn_elements_lines=None):
        assert_array_equal(np.array(list(triangles.keys())), np.array(list(cond_tn_elements_triangles.keys())))
        assert_array_equal(np.array(list(lines.keys())), np.array(list(cond_tn_elements_lines.keys())))

        triangles = OrderedDict(triangles)

        if len(lines) > 0:
            lines = OrderedDict(lines)

        verts = pd.DataFrame(np.array(list(mesh_nodes.values()))[:, :2], columns=['x', 'y'])
        tris = pd.DataFrame(triangles.values(), columns=['v0', 'v1', 'v2'])

        # Set the shape of the canvas
        self._xr = verts.x.min(), verts.x.max()
        self._yr = verts.y.min(), verts.y.max()

        # Add point at the begining of the dataframe to deal with different indexing
        verts.loc[-1] = [0, 0]
        verts.index = verts.index + 1
        verts.sort_index(inplace=True)

        self._verts = verts
        self._tris = tris

    def rasterize(self, mesh_nodes, triangles, cond_tn_elements_triangles,
                  lines=None, cond_tn_elements_lines=None, cs_lines=None, n_pixels_x=256,
                  save_image=False):

        srf = OrderedDict(cond_tn_elements_triangles)

        if self._verts is None and self._tris is None:
            self._prepare_verts_tris(mesh_nodes, triangles, cond_tn_elements_triangles, lines, cond_tn_elements_lines)

        canvas = ds.Canvas(x_range=self._xr, y_range=self._yr, plot_height=n_pixels_x, plot_width=n_pixels_x)

        self._tris['rf_value'] = np.squeeze(np.array(list(srf.values())))
        trimesh = canvas.trimesh(self._verts, self._tris, interpolate='nearest')

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
            cvs_lines = self.rasterize_lines(canvas, self._verts, lines, cond_tn_elements_lines,  cs_lines)

        #nearest_img = tf.shade(canvas.trimesh(self._verts, self._tris, interpolate='nearest'), name='10 Vertices')
        #export_image(img=nearest_img, filename='mesh_nearest_img', fmt=".png", export_path=".")

        #linear_img = tf.shade(canvas.trimesh(verts, tris, interpolate='linear'), name='10 Vertices')
        #export_image(img=linear_img, filename='mesh_linear_img', fmt=".png", export_path=".")

        # if len(lines) > 0:
        #     for cvs_l in cvs_lines:
        #         lines_img = tf.shade(cvs_l, name='lines', cmap=c)
        #
        #         nearest_img = tf.stack(nearest_img, lines_img, how="over")
        #
        #         # tf.stack(nearest_img, linear_img, how="over")
        #         #export_image(img=tf.stack(nearest_img, img_points, how="over"), filename='image_trimesh_points', fmt=".png", export_path=".")
        #     export_image(img=nearest_img, filename='dfm_image' + "{}".format(np.random.random()), fmt=".png", export_path=".")
        return trimesh, cvs_lines

    def rasterize_lines(self, cvs, verts, lines, cond_tn_elements_lines, cs_lines):
        if not self._lines_set:
            self._lines_to_rasterize = {}
            self._forcedirected = {}

        cvs_lines_list = []
        for cs, l_ids in cs_lines.items():
            self._lines_to_rasterize.setdefault(cs, {})
            lines_to_rasterize_data = {}
            for l_id in l_ids:
                if not self._lines_set:
                    self._lines_to_rasterize[cs][l_id] = [*lines[l_id]]
                lines_to_rasterize_data[l_id] = cond_tn_elements_lines[l_id]

            if not self._lines_set:
                self._lines_df[cs] = pd.DataFrame(self._lines_to_rasterize[cs].values(), columns=['source', 'target'])

                self._forcedirected[cs] = forceatlas2_layout(verts, self._lines_df[cs])

            self._lines_df[cs]['rf_value'] = np.squeeze(np.array(list(lines_to_rasterize_data.values())))
            cvs_lines = cvs.line(connect_edges(verts, self._lines_df[cs], weight='rf_value'), 'x','y', agg=ds.mean('rf_value'), line_width=cs)
            cvs_lines_list.append(cvs_lines)

        if not self._lines_set:
            self._lines_set = True

        return cvs_lines_list
