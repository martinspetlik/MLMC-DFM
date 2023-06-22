from typing import *
from abc import *

import warnings
import logging


import os
import sys
import numpy as np
import threading
import subprocess
import yaml
import ruamel.yaml
import attr
import collections
import traceback
import time
#import pandas
import scipy.spatial as sc_spatial
import scipy.interpolate as sc_interpolate
import atexit
import scipy as sc
from scipy import stats
from scipy.spatial import distance
from sklearn.utils.extmath import randomized_svd

src_path = os.path.dirname(os.path.abspath(__file__))

from bgem.gmsh import gmsh_io
from bgem.polygons import polygons
from homogenization import fracture
from homogenization import sim_sample
import matplotlib.pyplot as plt
import copy
#from plots_skg import matplotlib_variogram_plot
import shutil
#import compute_effective_cond
import gstools
from mlmc.random import correlated_field as cf
from sklearn.mixture import GaussianMixture


logging.getLogger('bgem').disabled = True

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')




def in_file(base):
    return "flow_{}.yaml".format(base)


def mesh_file(base):
    return "mesh_{}.msh".format(base)


def fields_file(base):
    return "fields_{}.msh".format(base)


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


# class FlowThread(threading.Thread):
#
#     def __init__(self, basename, outer_regions, config_dict):
#         self.base = basename
#         self.outer_regions_list = outer_regions
#         self.flow_args = config_dict["flow_executable"].copy()
#         n_steps = config_dict["n_pressure_loads"]
#         t = np.pi * np.arange(0, n_steps) / n_steps
#         self.p_loads = np.array([np.cos(t), np.sin(t)]).T
#         super().__init__()
#
#     def run(self):
#         in_f = in_file(self.base)
#         out_dir = self.base
#         # n_loads = len(self.p_loads)
#         # flow_in = "flow_{}.yaml".format(self.base)
#         params = dict(
#             mesh_file=mesh_file(self.base),
#             fields_file=fields_file(self.base),
#             outer_regions=str(self.outer_regions_list),
#             n_steps=len(self.p_loads)
#             )
#         substitute_placeholders("flow_templ.yaml", in_f, params)
#         # self.flow_args = ["docker", "run", "-v", "{}:{}".format(out_dir, out_dir),
#         #                   "docker://flow123d/flow123d-gnu:3.9.0", "flow123d"]
#         self.flow_args.extend(['--output_dir', out_dir, in_f])
#
#         if os.path.exists(os.path.join(out_dir, "flow_fields.msh")):
#             return True
#         with open(self.base + "_stdout", "w") as stdout:
#             with open(self.base + "_stderr", "w") as stderr:
#                 print("flow args ", self.flow_args)
#                 completed = subprocess.run(self.flow_args, stdout=stdout, stderr=stderr)
#             print("Exit status: ", completed.returncode)
#             status = completed.returncode == 0
#         conv_check = self.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
#         print("converged: ", conv_check)
#         return status  # and conv_check
#
#     def check_conv_reasons(self, log_fname):
#         with open(log_fname, "r") as f:
#             for line in f:
#                 tokens = line.split(" ")
#                 try:
#                     i = tokens.index('convergence')
#                     if tokens[i + 1] == 'reason':
#                         value = tokens[i + 2].rstrip(",")
#                         conv_reason = int(value)
#                         if conv_reason < 0:
#                             print("Failed to converge: ", conv_reason)
#                             return False
#                 except ValueError:
#                     continue
#         return True


@attr.s(auto_attribs=True)
class Region:
    name:str = ""
    dim:int = -1
    boundary:bool = False
    mesh_step:float = 0.0

    def is_active(self, dim):
        active = self.dim >= 0
        if active:
            assert dim == self.dim, "Can not create shape of dim: {} in region '{}' of dim: {}.".format(dim, self.name, self.dim)
        return active


def gmsh_mesh_bulk_elements(mesh):
    """
    Generator of IDs of bulk elements.
    :param mesh:
    :return:
    """
    is_bc_reg_id={}
    for name, reg_id_dim in mesh.physical.items():
        is_bc_reg_id[reg_id_dim] = (name[0] == '.')
    for eid, ele in mesh.elements.items():
        (t, tags, nodes) = ele
        dim = len(nodes) - 1
        if not is_bc_reg_id[(tags[0], dim)]:
            yield eid, ele


class BulkBase(ABC):
    @abstractmethod
    def element_data(self, mesh, eid):
        """
        :return:
        """
        pass


@attr.s(auto_attribs=True)
class BulkFields(BulkBase):
    mean_log_conductivity: Tuple[float, float]
    cov_log_conductivity: Optional[List[List[float]]]
    angle_mean: float = attr.ib(converter=float)
    angle_concentration: float = attr.ib(converter=float)

    #print("cov_log_conductivity ", cov_log_conductivity)

    def element_data(self, mesh, eid):

        # Unrotated tensor (eigenvalues)
        if self.cov_log_conductivity is None:
            log_eigenvals = self.mean_log_conductivity
        else:
            log_eigenvals = np.random.multivariate_normal(
                mean=self.mean_log_conductivity,
                cov=self.cov_log_conductivity)

        unrotated_tn = np.diag(np.power(10, log_eigenvals))
        # print("unrotated_tn ", unrotated_tn)
        # exit()

        # rotation angle
        if self.angle_concentration is None or self.angle_concentration == 0:
            angle = np.random.uniform(0, 2*np.pi)
        elif self.angle_concentration == np.inf:
            angle = self.angle_mean
        else:
            angle = np.random.vonmises(self.angle_mean, self.angle_concentration)
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])
        #print("rot_mat ,", rot_mat)
        #print("unrotated ", unrotated_tn)
        cond_2d = rot_mat @ unrotated_tn @ rot_mat.T

        # print("cond_2d ", cond_2d)
        # exit()

        #
        # print("cond_2d ", cond_2d)
        # exit()
        return 1.0, cond_2d
        #print("unrotated_tn ", unrotated_tn)

        # ## gstools experiments
        # from mlmc.random import correlated_field as cf
        # model = cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField("exp", log=True,
        #                                                                        sigma=1e-3, mode_no=10000)),
        #
        #
        #
        # #####
        # exit()


@attr.s(auto_attribs=True)
class BulkFieldsGSTools(BulkBase):
    mean_log_conductivity: Tuple[float, float]
    cov_log_conductivity: Optional[List[List[float]]]
    angle_mean: float = attr.ib(converter=float)
    angle_concentration: float = attr.ib(converter=float)
    corr_lengths_x: Optional[List[float]]  # [15, 0, 0]
    corr_lengths_y: Optional[List[float]]  # = [3, 0, 0]
    anis_angles: Optional[List[float]]  # = [0, 0, 0]
    anisotropy: float = None
    rotation: float = None
    mode_no: int = 10000
    angle_var = 1
    _rf_sample = None
    _mesh_data= None
    log = False

    def _pca(self, mean_k_xx_yy, cov_matrix_k_xx_yy):
        n_samples = 10000
        samples = np.random.multivariate_normal(mean=mean_k_xx_yy, cov=cov_matrix_k_xx_yy, size=n_samples)

        # sample_means = [np.mean(samples[:, 0]), np.mean(samples[:, 1])]
        # sample_vars = [np.var(samples[:, 0]), np.var(samples[:, 1])]
        # print("sample means: {} vars: {}".format(sample_means, sample_vars))

        covariance_matrix = np.cov(samples.T)
        #print("covariance matrix ", covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        self._projection_matrix = (eigen_vectors.T[:][:]).T

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix_k_xx_yy)
        self._projection_matrix = (eigen_vectors.T[:][:]).T
        p_components = samples.dot(self._projection_matrix)

        #print("p components shape ", p_components.shape)

        pc_means = [np.mean(p_components[:, 0]), np.mean(p_components[:, 1])]
        pc_vars = [np.var(p_components[:, 0]), np.var(p_components[:, 1])]
        #print("pc means: {} vars: {}".format(pc_means, pc_vars))

        ##########
        ## K_xx ##
        ##########
        if self.corr_lengths_x[0] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[0], self.corr_lengths_y[0]]

        self._model_k_xx = gstools.Exponential(dim=2, var=pc_vars[0], len_scale=len_scale, angles=self.anis_angles[0])

        ##########
        ## K_yy ##
        ##########
        if self.corr_lengths_x[1] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[1], self.corr_lengths_y[1]]

        self._model_k_yy = gstools.Exponential(dim=2, var=pc_vars[1], len_scale=len_scale, angles=self.anis_angles[1])


        ###########
        ## Angle ##
        ###########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2]]

        #model_angle = gstools.Gaussian(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._model_angle_x = gstools.Exponential(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._model_angle_y = gstools.Exponential(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

        # print("self._model_k_xx ", self._model_k_xx)
        # print("self._model_k_yy ", self._model_k_yy)
        # print("self._model_angle ", self._model_angle)

        self._pc_means = pc_means

    def _create_field(self, mean_log_conductivity, cov_log_conductivity):
        self._pca(mean_log_conductivity, cov_log_conductivity)

        field_k_xx = cf.Field('k_xx', cf.GSToolsSpatialCorrelatedField(self._model_k_xx, log=self.log,
                                                                       #sigma=np.sqrt(self.cov_log_conductivity[0,0]),
                                                                       mode_no=self.mode_no,
                                                                       #mu=pca_means[0]
                                                                       ))

        field_k_yy = cf.Field('k_yy', cf.GSToolsSpatialCorrelatedField(self._model_k_yy, log=self.log,
                                                                       #sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       #mu=pca_means[1]
                                                                       ))

        field_angle_x = cf.Field('angle_x', cf.GSToolsSpatialCorrelatedField(self._model_angle_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no))

        field_angle_y = cf.Field('angle_y', cf.GSToolsSpatialCorrelatedField(self._model_angle_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no))

        self._fields = cf.Fields([field_k_xx, field_k_yy, field_angle_x, field_angle_y])

    def element_data(self, mesh, eid):
        if self._rf_sample is None:
            #if self.model_k_xx is None:
            self._create_field(self.mean_log_conductivity, self.cov_log_conductivity)

            self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            # print("mesh data ", list(self._mesh_data.keys()))
            # print("mesh data ele ids ", self._mesh_data["ele_ids"])
            # print("mesh_data['points'] ", self._mesh_data['points'])
            self._fields.set_points(self._mesh_data['points'], self._mesh_data['point_region_ids'], self._mesh_data['region_map'])
            self._rf_sample = self._fields.sample()

            self._rf_sample["k_xx"] += self._pc_means[0]
            self._rf_sample["k_yy"] += self._pc_means[1]

            srf_data = np.array([self._rf_sample["k_xx"], self._rf_sample["k_yy"]])

            # bins = 60
            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(srf_data[0, :], bins=bins, color="red", label="k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # plt.hist(srf_data[1, :], bins=bins, color="red", label="k_xy", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # plt.hist(srf_data[2, :], bins=bins, color="red", label="k_yy", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()


            # pca_cond.transform()
            # inv_srf_data = pca_cond.inverse_transform(srf_data.T).T
            inv_srf_data = np.matmul(srf_data.T, self._projection_matrix.T).T

            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(inv_srf_data[0, :], bins=bins, color="red", label="inv k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # plt.hist(inv_srf_data[1, 0], bins=bins, color="red", label="inv k_xy", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            # #
            # plt.hist(inv_srf_data[2, 0], bins=bins, color="red", label="inv k_yy", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()

            # for i in range(inv_srf_data.shape[1]):
            #     print("inv data ", inv_srf_data[:, i])
            #
            # print("inv srf data cov ", np.cov(inv_srf_data))
            # print("inv srf data ", inv_srf_data.shape)

            self._rf_sample["k_xx"] = inv_srf_data[0, :]
            self._rf_sample["k_yy"] = inv_srf_data[1, :]

            ##########################
            ## Variogram estimation ##
            ##########################
            # import skgstat as skg
            # random_field = np.power(10, self._rf_sample["k_xx"])
            # random_field = self._rf_sample["k_xx"]
            # #V = skg.Variogram(self._mesh_data['points'], random_field)
            # #bin_center, gamma = V.get_empirical(bin_center=True)
            # #print("V describe", V.describe())
            #
            # #print("self._mesh_data['points'] ", self._mesh_data['points'].shape)
            # #print("random field shape ", random_field.shape)
            #
            # x_coord, y_coord = self._mesh_data['points'][:, 0], self._mesh_data['points'][:, 1]
            # bin_center, gamma = gstools.vario_estimate((x_coord, y_coord), random_field)
            # fit_model = gstools.Exponential(dim=2)
            # params = fit_model.fit_variogram(bin_center, gamma, nugget=False)
            #
            # print("params ", params)

            ###########################
            ###########################



            data_srf_angle_vec = np.stack([self._rf_sample["angle_x"],  self._rf_sample["angle_y"]])

            if self.anisotropy is not None and self.rotation is not None:
                c, s = np.cos(self.rotation), np.sin(self.rotation)
                rot_mat_anis = np.array([[c, -s], [s, c]])
                data_srf_angle_vec[0, ...] *= self.anisotropy
                data_srf_angle_vec = rot_mat_anis @ data_srf_angle_vec

            cos_sin_angle = data_srf_angle_vec / np.linalg.norm(data_srf_angle_vec, axis=0)
            angle = np.arctan2(cos_sin_angle[1, :], cos_sin_angle[0, :])
            self._rf_sample["angle"] = angle

            xv = self._mesh_data['points'][:, 0]
            yv = self._mesh_data['points'][:, 1]
            #
            # plt.hist(srf_data[:, 0], bins=bins, color="red", label="final k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # plt.hist(srf_data[:, 2], bins=bins, color="red", label="k_yy", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()

            #import matplotlib
            #matplotlib.rcParams.update({'font.size': 34})

            # print("rf sample angle ", len(rf_sample["angle"]))
            # exit()
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, self._rf_sample["k_xx"], level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$log(k_{xx})$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, np.power(10, self._rf_sample["k_xx"]), level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$k_{xx}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, self._rf_sample["k_yy"], level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$log(k_{yy})$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, np.power(10, self._rf_sample["k_yy"]), level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$k_{yy}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, self._rf_sample["angle"], level=32)
            #     fig.colorbar(cont)
            #     plt.title("angle srf")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            #
            # k_xy = []
            # for eid in self._mesh_data["ele_ids"]:
            #     ele_idx = np.where(self._mesh_data["ele_ids"] == eid)
            #     k_xx = self._rf_sample["k_xx"][ele_idx][0]
            #     k_yy = self._rf_sample["k_yy"][ele_idx][0]
            #
            #     unrotated_tn = np.diag(np.power(10, np.array([k_xx, k_yy])))
            #
            #     angle = self._rf_sample["angle"][ele_idx][0]
            #     c, s = np.cos(angle), np.sin(angle)
            #
            #     rot_mat = np.array([[c, -s], [s, c]])
            #     # print("rot mat ", rot_mat)
            #
            #     cond_2d = rot_mat @ unrotated_tn @ rot_mat.T
            #
            #     k_xy.append((cond_2d[0,1] + cond_2d[1,0])/2)
            #
            # print("k_xy ", k_xy)
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, k_xy, level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$k_{xy}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # exit()
            #



            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # ax = axes  # [0]
            # #ax.set_xscale('log')
            # ax.hist(angle, bins=25, density=True)
            # fig.legend()
            # fig.show()


            # rf_sample["angle"] = (rf_sample["angle"] * 100) % (2*np.pi)  # uniform distribution
            #
            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # ax = axes  # [0]
            # # ax.set_xscale('log')
            # ax.hist(rf_sample["angle"], bins=25, density=True)
            # fig.legend()
            # fig.show()

            # xv = mesh_data['points'][:, 0]
            # yv = mesh_data['points'][:, 1]

            # print("rf sample angle ", len(rf_sample["angle"]))
            # exit()
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, rf_sample["angle"], level=32)
            #     fig.colorbar(cont)
            #     plt.title("angle srf")
            #     plt.show()
            # except Exception as e:
            #    print(e)

        ele_idx = np.where(self._mesh_data["ele_ids"] == eid)
        k_xx = self._rf_sample["k_xx"][ele_idx][0]
        k_yy = self._rf_sample["k_yy"][ele_idx][0]


        unrotated_tn = np.diag(np.power(10, np.array([k_xx, k_yy])))


        angle = self._rf_sample["angle"][ele_idx][0]
        c, s = np.cos(angle), np.sin(angle)

        rot_mat = np.array([[c, -s], [s, c]])
        #print("rot mat ", rot_mat)

        cond_2d = rot_mat @ unrotated_tn @ rot_mat.T
        # print("cond_2d ", cond_2d)
        # exit()
        return 1.0, cond_2d

        # print("mesh.elements ", mesh.elements)
        # print("mesh.points ", mesh.points)
        # exit()
        #
        # el_type, tags, node_ids = mesh.elements[eid]
        # center = np.mean([np.array(mesh.nodes[nid]) for nid in node_ids], axis=0)
        # exit()






        # # Unrotated tensor (eigenvalues)
        # if self.cov_log_conductivity is None:
        #     log_eigenvals = self.mean_log_conductivity
        # else:
        #     log_eigenvals = np.random.multivariate_normal(
        #         mean=self.mean_log_conductivity,
        #         cov=self.cov_log_conductivity,
        #         )

        # unrotated_tn = np.diag(np.power(10, log_eigenvals))
        #
        # # rotation angle
        # if self.angle_concentration is None or self.angle_concentration == 0:
        #     angle = np.random.uniform(0, 2*np.pi)
        # elif self.angle_concentration == np.inf:
        #     angle = self.angle_mean
        # else:
        #     angle = np.random.vonmises(self.angle_mean, self.angle_concentration)
        # c, s = np.cos(angle), np.sin(angle)
        # rot_mat = np.array([[c, -s], [s, c]])
        # cond_2d = rot_mat @ unrotated_tn @ rot_mat.T
        # return 1.0, cond_2d

    @staticmethod
    def extract_mesh(mesh):
        """
        Extract mesh from file
        :param mesh_file: Mesh file path
        :return: Dict
        """
        is_bc_region = {}
        region_map = {}
        for name, (id, _) in mesh.physical.items():
            unquoted_name = name.strip("\"'")
            is_bc_region[id] = (unquoted_name[0] == '.')
            region_map[unquoted_name] = id

        triangles = {}

        bulk_elements = []
        for id, el in mesh.elements.items():
            _, tags, i_nodes = el
            region_id = tags[0]
            if not is_bc_region[region_id] and len(i_nodes) == 3:
                bulk_elements.append(id)

        n_bulk = len(bulk_elements)
        centers = np.empty((n_bulk, 3))
        ele_ids = np.zeros(n_bulk, dtype=int)
        point_region_ids = np.zeros(n_bulk, dtype=int)

        for i, id_bulk in enumerate(bulk_elements):
            _, tags, i_nodes = mesh.elements[id_bulk]
            region_id = tags[0]

            # print("i_nodes ", i_nodes)
            # print("[mesh.nodes[i_node] for i_node in i_nodes] ", [mesh.nodes[i_node] for i_node in i_nodes])

            centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
            point_region_ids[i] = region_id
            ele_ids[i] = id_bulk
            triangles[ele_ids[i]] = i_nodes

        min_pt = np.min(centers, axis=0)
        max_pt = np.max(centers, axis=0)
        diff = max_pt - min_pt
        min_axis = np.argmin(diff)
        non_zero_axes = [0, 1, 2]
        # TODO: be able to use this mesh_dimension in fields
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map,
                'mesh_nodes': mesh.nodes, 'triangles': triangles}


# @attr.s(auto_attribs=True)
# class BulkFieldGSTools(BulkBase):
#     dim: int = attr.ib(converter=int)
#     sigma: float = attr.ib(converter=float)
#     corr_length: float = attr.ib(converter=float)
#     model_name: str = "exp"
#     log: bool = True
#     mode_no: int = 10000
#     model = None
#
#     def _create_field(self):
#         from mlmc.random import correlated_field as cf
#         self.model = cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(self.model_name, log=self.log,
#                                                                                sigma=self.sigma, mode_no=self.mode_no)),
#
#     def element_data(self, mesh, eid):
#         if self.model is None:
#             self._create_field()
#
#
#
#         # Unrotated tensor (eigenvalues)
#         if self.cov_log_conductivity is None:
#             log_eigenvals = self.mean_log_conductivity
#         else:
#             log_eigenvals = np.random.multivariate_normal(
#                 mean=self.mean_log_conductivity,
#                 cov=self.cov_log_conductivity,
#                 )
#
#         unrotated_tn = np.diag(np.power(10, log_eigenvals))
#
#         # rotation angle
#         if self.angle_concentration is None or self.angle_concentration == 0:
#             angle = np.random.uniform(0, 2*np.pi)
#         elif self.angle_concentration == np.inf:
#             angle = self.angle_mean
#         else:
#             angle = np.random.vonmises(self.angle_mean, self.angle_concentration)
#         c, s = np.cos(angle), np.sin(angle)
#         rot_mat = np.array([[c, -s], [s, c]])
#         cond_2d = rot_mat @ unrotated_tn @ rot_mat.T
#         return 1.0, cond_2d


class BulkMicroScale(BulkBase):
    def __init__(self, microscale):
        self.microscale = microscale
        self.microscale_tensors = None

    def element_data(self, mesh, eid):
        # print("mesh ", mesh)
        # print("eid ", eid)
        if self.microscale_tensors is None:
            self.microscale_tensors = self.microscale.effective_tensor_from_bulk()
        return 1.0, self.microscale_tensors[eid]


class BulkFromFine(BulkBase):
    def __init__(self, fine_problem):
        points, values = fine_problem.bulk_field()
        self.mean_val = np.mean(values)
        tria = sc_spatial.Delaunay(points)
        #print("Values, shape:", values.shape)
        self.interp = sc_interpolate.LinearNDInterpolator(tria, values.T, fill_value=0)
        self.interp_nearest = sc_interpolate.LinearNDInterpolator(points, values.T)

    def element_data(self, mesh, eid):
        el_type, tags, node_ids = mesh.elements[eid]
        center = np.mean([np.array(mesh.nodes[nid]) for nid in node_ids], axis=0)
        v = self.interp(center[0:2])
        if np.alltrue(v == 0):
            #v = self.interp_nearest(center[0:2])
            v = [[self.mean_val, 0, self.mean_val]]
        #print("V, shape:", v.shape)
        v00, v01, v11 = v[0]
        cond = np.array([[v00, v01], [v01, v11]])
        #print("cond, shape: ", cond.shape)
        e0, e1 = np.linalg.eigvalsh(cond)
        if e0 < 1e-20 or e1 < 1e-20:
            print(e0, e1, v, center)
            assert False

        print("cond ", cond)
        exit()
        return 1.0, cond


class BulkChoose(BulkBase):
    def __init__(self, finer_level_path):
        self.cond_tn = np.array(pandas.read_csv(finer_level_path, sep=' '))

    def element_data(self, mesh, eid):
        idx = np.random.randint(len(self.cond_tn))
        return 1.0, self.cond_tn[idx].reshape(2,2)


class SpatialCorrelatedFieldHom(cf.SpatialCorrelatedField):

    def _initialize(self, **kwargs):
        self._cond_tensors = None
        ### Attributes computed in precalculation.
        self.cov_mat = None
        # Covariance matrix (dense).
        self._n_approx_terms = None
        # Length of the sample vector, number of KL (Karhunen-Loe?ve) expansion terms.
        self._cov_l_factor = None
        # (Reduced) L factor of the SVD decomposition of the covariance matrix.
        self._sqrt_ev = None
        # (Reduced) square roots of singular values.


    def _sample(self):
        cond_tn_values = self._cond_tensors
        # Remove nans if there are any
        mask = np.isnan(cond_tn_values).any(axis=1)
        cond_tn_values = cond_tn_values[~mask]

        mult_coef = 1 / np.abs(np.min(cond_tn_values))
        values = cond_tn_values * mult_coef

        # values[:, 0] = np.log10(values[:, 0])
        # values[:, 2] = np.log10(values[:, 2])

        # Fit a Gaussian Mixture Model (GMM)
        n_components = 250
        gmm = GaussianMixture(n_components=n_components)  # Choose the number of components
        # gmm.fit(values.reshape(-1, 1))
        gmm.fit(values)

        size = self.n_points
        samples, _ = gmm.sample(n_samples=size)

        samples /= mult_coef

        return samples

    # def _sample(self):
    #     """
    #     :param uncorelated: Random samples from standard normal distribution.
    #     :return: Random field evaluated in points given by 'set_points'.
    #     """
    #     if self._cov_l_factor is None:
    #         self.svd_dcmp()
    #     #print("self._cond_tensors.shape ", self._cond_tensors.shape)
    #     #print("self._cond_tensors ", self._cond_tensors)
    #     # print("len(self._cond_tensors) ", len(self._cond_tensors))
    #     # print("type self.n_approx_terms ", type(self.n_approx_terms))
    #     # print("self.n_approx_terms ", self.n_approx_terms)
    #
    #     size = self.n_points
    #     #size = self.n_approx_terms
    #     uncorelated_indices = np.random.choice(len(self._cond_tensors), size=size) #np.random.choice(len(self._cond_tensors), size=self._n_approx_terms)
    #     #print("uncorelated indices", uncorelated_indices)
    #     uncorelated = self._cond_tensors[uncorelated_indices]
    #     #print("uncorelated ", uncorelated)
    #     print("uncorelated.shape ", uncorelated.shape)
    #     print("self._cov_l_factor ", self._cov_l_factor.shape)
    #
    #     return uncorelated #self._cov_l_factor.dot(uncorelated)


class BulkHomogenizationFineSample(BulkBase):
    def __init__(self, config_dict):
        self._cond_tns = None
        self._get_tensors(config_dict)

        self.mean_log_conductivity = None
        if "mean_log_conductivity" in config_dict:
            self.mean_log_conductivity = config_dict["mean_log_conductivity"]

        #print("config dict ", config_dict)

        avg_var_list, avg_len_scale_list = self._calc_var_len_scale(config_dict)

        #print("avg var list ", avg_var_list)
        #print("avg len scale list ", avg_len_scale_list)

        #print("self. cond tn ", self._cond_tns)

        self._cond_tns = BulkHomogenizationFineSample.symmetrize_cond_tns(self._cond_tns)

        self._rf_sample = None
        srf_model = SpatialCorrelatedFieldHom(corr_exp="exp", corr_length=np.mean(avg_len_scale_list))#, sigma=np.sqrt(np.mean(avg_var_list)))
        srf_model._cond_tensors = self._cond_tns

        print("SRF MODEL corr length: {}, sigma: {}".format(srf_model._corr_length, srf_model.sigma))

        field_cond_tn = cf.Field('cond_tn', srf_model)
        self._fields = cf.Fields([field_cond_tn])

    @staticmethod
    def symmetrize_cond_tns(cond_tns):
        sym_values = (cond_tns[:, 1] + cond_tns[:, 2])/2
        cond_tns = np.delete(cond_tns, 2, 1)
        cond_tns[:, 1] = sym_values
        return cond_tns

    def _get_tensors(self, config_dict):
        if "pred_cond_tn_pop_file" in config_dict["fine"]:
            cond_pop_file = config_dict["fine"]["pred_cond_tn_pop_file"]
        else:
            cond_pop_file = config_dict["fine"]["cond_tn_pop_file"]

        #print("cond_pop_file ", cond_pop_file)
        #cond_pop_file =
        #print("cond pop file ", cond_pop_file)
        self._cond_tns = np.load(cond_pop_file)
        #print("self._cond_tns ", self._cond_tns)
        #self._cond_tns = self._cond_tns.reshape(int(len(self._cond_tns)/4), 4)

        # cond_file = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/output_good_backup/L01_S0000000/cond_tensors.yaml"
        # with open(cond_file, "r") as f:
        #     cond_tns = ruamel.yaml.load(f)
        #
        # self._cond_tns = cond_tns

    @staticmethod
    def get_len_scales(cond_tns):
        #print("cond tns keys ()", cond_tns.keys())
        coords = np.array(list(cond_tns.keys()))
        x_coord, y_coord = coords[:, 0], coords[:, 1]
        rf_values = np.array(list(cond_tns.values()))

        # print("coords ", coords)
        # print("coords.shape ", coords.shape)
        # print("rf values .shape ", rf_values.shape)

        bin_center, gamma = gstools.vario_estimate((x_coord, y_coord), rf_values[:, 0])
        fit_model = gstools.Exponential(dim=2)
        k_xx_params = fit_model.fit_variogram(bin_center, gamma, nugget=False)
        # print("k xx params ", k_xx_params)
        # print("np.var(rf_values[:, 0]) ", np.var(rf_values[:, 0]))

        bin_center, gamma = gstools.vario_estimate((x_coord, y_coord), rf_values[:, 1])
        fit_model = gstools.Stable(dim=2)
        k_xy_params = fit_model.fit_variogram(bin_center, gamma, nugget=False)

        # ax = fit_model.plot()
        # print("bin center ", bin_center)
        # print("gamma ", gamma)
        # ax.scatter(bin_center, gamma)
        # plt.plot()
        # print(fit_model)
        # print("k xy params ", k_xy_params)
        # print("np.var(rf_values[:, 1]) ", np.var(rf_values[:, 1]))

        bin_center, gamma = gstools.vario_estimate((x_coord, y_coord), rf_values[:, 2])
        fit_model = gstools.Exponential(dim=2)
        k_yy_params = fit_model.fit_variogram(bin_center, gamma, nugget=False)
        # print("k yy params ", k_yy_params)
        # print("np.var(rf_values[:, 2]) ", np.var(rf_values[:, 2]))
        # exit()

        return k_xx_params, k_xy_params, k_yy_params

    @staticmethod
    def get_vars(cond_tns):
        rf_values = np.array(list(cond_tns.values()))
        return np.var(rf_values[:, 0]), np.var(rf_values[:, 1]), np.var(rf_values[:, 2])

    def _calc_var_len_scale(self, config_dict):
        sample_id = 0
        avg_len_scale_list = []
        avg_var_list = []
        while True:
            cond_file = os.path.join(config_dict["fine"]["sample_cond_tns"], "S{:07d}_{}".format(sample_id, sim_sample.DFMSim.COND_TN_FILE))
            pred_cond_file = os.path.join(config_dict["fine"]["sample_cond_tns"],
                                     "S{:07d}_{}".format(sample_id, sim_sample.DFMSim.PRED_COND_TN_FILE))

            if not os.path.exists(cond_file):
                break

            with open(cond_file, "r") as f:
                cond_tns = ruamel.yaml.load(f)

            with open(pred_cond_file, "r") as f:
                pred_cond_tns = ruamel.yaml.load(f)

            if len(pred_cond_tns) > 0:
                k_xx_params, k_xy_params, k_yy_params = BulkHomogenizationFineSample.get_len_scales(pred_cond_tns)
                k_xx_var, k_xy_var, k_yy_var = BulkHomogenizationFineSample.get_vars(pred_cond_tns)
            else:
                k_xx_params, k_xy_params, k_yy_params = BulkHomogenizationFineSample.get_len_scales(cond_tns)
                k_xx_var, k_xy_var, k_yy_var = BulkHomogenizationFineSample.get_vars(cond_tns)


            k_xx_len_scale = k_xx_params[0]['len_scale']
            k_xy_len_scale = k_xy_params[0]['len_scale']
            k_yy_len_scale = k_yy_params[0]['len_scale']

            #print("LEN SCALE k_xx: {}, k_xy: {}, k_yy:{}".format(k_xx_len_scale, k_xy_len_scale, k_yy_len_scale))
            #print("VAR k_xx: {}, k_xy: {}, k_yy:{}".format(k_xx_var, k_xy_var, k_yy_var))

            avg_len_scale = (k_xx_len_scale + k_xy_len_scale + k_yy_len_scale) / 3
            avg_var = (k_xx_var + k_xy_var + k_yy_var) / 3

            avg_len_scale_list.append(avg_len_scale)
            avg_var_list.append(avg_var)

            sample_id += 1

        return avg_var_list, avg_len_scale_list


    def element_data(self, mesh, eid):
        if self._rf_sample is None:
            self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            self._fields.set_points(self._mesh_data['points'],
                                    self._mesh_data['point_region_ids'],
                                    self._mesh_data['region_map'])
            self._rf_sample = self._fields.sample()

            xv = self._mesh_data['points'][:, 0]
            yv = self._mesh_data['points'][:, 1]

            #print("self._rf_sample[cond_tn].shape ", self._rf_sample["cond_tn"].shape)

            # import matplotlib
            # #matplotlib.rcParams.update({'font.size': 34})
            #
            # bins=60
            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(self._rf_sample["cond_tn"][:, 0], bins=bins, color="red", label="k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(np.abs(self._rf_sample["cond_tn"][:, 0]), bins=bins, color="red", label="abs k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(np.exp(self._rf_sample["cond_tn"][:, 0]), bins=bins, color="red", label="abs k_xx", density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, self._rf_sample["cond_tn"][:, 0], level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$k_{xx}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, np.abs(self._rf_sample["cond_tn"][:, 0]), level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$abs k_{xx}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # plt.hist(np.abs(self._rf_sample["cond_tn"][:, 1]), bins=bins, color="red", label="abs k_xy",
            #          density=True)
            # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # plt.legend()
            # plt.show()
            #
            # try:
            #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            #     cont = ax.tricontourf(xv, yv, self._rf_sample["cond_tn"][:, 1], level=32)
            #     fig.colorbar(cont)
            #     plt.title(r"$k_{xy}$")
            #     plt.show()
            # except Exception as e:
            #     print(e)
            #
            # # bins = 60
            # # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # # plt.hist(self._rf_sample["cond_tn"][:, 3], bins=bins, color="red", label="k_yy", density=True)
            # # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # # plt.legend()
            # # plt.show()
            # #
            # # # np_bins = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 50)  # Define histogram bins
            # # plt.hist(np.abs(self._rf_sample["cond_tn"][:,3]), bins=bins, color="red", label="abs k_yy", density=True)
            # # # plt.hist(samples[:, 0], bins=bins, color="blue", label="k_xy sampled", alpha=0.65, density=True)
            # # plt.legend()
            # # plt.show()
            # #
            # # try:
            # #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            # #     cont = ax.tricontourf(xv, yv, self._rf_sample["cond_tn"][:, 3], level=32)
            # #     fig.colorbar(cont)
            # #     plt.title(r"$k_{yy}$")
            # #     plt.show()
            # # except Exception as e:
            # #     print(e)
            # #
            # # try:
            # #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            # #     cont = ax.tricontourf(xv, yv, np.abs(self._rf_sample["cond_tn"][:, 3]), level=32)
            # #     fig.colorbar(cont)
            # #     plt.title(r"$abs k_{yy}$")
            # #     plt.show()
            # # except Exception as e:
            # #     print(e)
            #
            # print("self._rf_sample ", self._rf_sample)

        ele_idx = np.where(self._mesh_data["ele_ids"] == eid)
        cond_tn = self._rf_sample["cond_tn"][ele_idx]

        #print("cond tn ", cond_tn)

        cond_tn = [[cond_tn[0][0], cond_tn[0][1]], [cond_tn[0][1], cond_tn[0][2]]]

        # cond_tn = cond_tn.reshape(2, 2)
        # sym_value = (cond_tn[0,1] + cond_tn[1,0])/2
        # cond_tn[0,1] = cond_tn[1,0] = sym_value

        cond_tn[0][0] = np.abs(cond_tn[0][0])
        cond_tn[1][1] = np.abs(cond_tn[1][1])

        return 1.0, cond_tn



class BulkHomogenization(BulkBase):
    def __init__(self, config_dict):
        self._cond_tns = None
        #print("config_dict ", config_dict)

        self._get_tensors(config_dict)
        self.mean_log_conductivity = None

        if "mean_log_conductivity" in config_dict:
            self.mean_log_conductivity = config_dict["mean_log_conductivity"]

        self._center_points = np.asarray(list(self._cond_tns.keys()))

    def _get_tensors(self, config):
        with open(config["cond_tns_yaml_file"], "r") as f:
            self._cond_tns = ruamel.yaml.load(f)

        if len(self._cond_tns) == 0:
            with open(config["pred_cond_tns_yaml_file"], "r") as f:
                self._cond_tns = ruamel.yaml.load(f)

        tria = sc_spatial.Delaunay(list(self._cond_tns.keys()))
        self._mean_val = np.mean(list(self._cond_tns.values()))

        values = np.array(list(self._cond_tns.values()))[:, [0,1,3]]
        self._interp = sc_interpolate.LinearNDInterpolator(tria, values, fill_value=0)
        self._interp_nearest = sc_interpolate.NearestNDInterpolator(tria, values)
        #print("self._cond_tns ", self._cond_tns)

    def element_data(self, mesh, eid):
        el_type, tags, node_ids = mesh.elements[eid]
        center = np.mean([np.array(mesh.nodes[nid]) for nid in node_ids], axis=0)[0:2]
        dist_2 = np.sum((self._center_points - center) ** 2, axis=1)
        cond_tn = self._cond_tns[tuple(self._center_points[np.argmin(dist_2)])]

        interp_value = self._interp(center)

        # print("cond tn ", cond_tn)
        # print("cond tn ", cond_tn.shape)
        # print("interp ", interp_value)

        if np.all(interp_value == 0):
            interp_value = self._interp_nearest(center)

        interp_value = np.squeeze(interp_value)

        cond_tn = np.array([[interp_value[0], interp_value[1]], [interp_value[1], interp_value[2]]])
        return 1.0, cond_tn.reshape(2, 2)


@attr.s(auto_attribs=True)
class FractureModel:
    fractures: fracture.Fractures
    region_to_fracture: Dict[int, int]
    aperture_per_size: float = attr.ib(converter=float)
    water_viscosity: float = attr.ib(converter=float)
    gravity_accel: float = attr.ib(converter=float)
    water_density: float = attr.ib(converter=float)
    target_sigma: float = None
    max_fr: float = 1
    bulk_model: BulkBase = None

    def element_data(self, mesh, eid, elid_to_fr):
        el_type, tags, node_ids = mesh.elements[eid]

        # reg_id = tags[0] - 10000
        # line, compute conductivity from fracture size using cubic law
        # Isotropic conductivity in fractures. (Simplification.)
        # i_fr = self.region_to_fracture[reg_id]
        i_fr = elid_to_fr[eid]
        fr_size = self.fractures.fractures[i_fr].rx

        if self.target_sigma is None or self.bulk_model is None:
            # print("variable cond")
            # exit()
            cs = fr_size * self.aperture_per_size
            cond = cs ** 2 / 12 * self.water_density * self.gravity_accel / self.water_viscosity
        else:
            #print("target sigma: {}, max_fr: {}".format(self.target_sigma, self.max_fr))
            # print("self.max_fr ", self.max_fr)
            # print("self.bulk_model.mean_log_conductivity ", int(np.mean(self.bulk_model.mean_log_conductivity)))
            cond = self.target_sigma * self.max_fr * 10 ** int(np.mean(self.bulk_model.mean_log_conductivity)) # @TODO: remove abs numbers ASAP
            cs = np.sqrt(12 * cond / (self.water_density * self.gravity_accel / self.water_viscosity))

        #print("i_fr: {}, cs: {}, cond: {}".format(i_fr, cs, cond))
        # exit()
        cond_tn = cond * np.eye(2, 2)

        return cs, cond_tn, fr_size

def tensor_3d_flatten(tn_2d):
    tn3d = np.eye(3)
    tn3d[0:2, 0:2] = tn_2d
    # tn3d[0:2, 0:2] += tn_2d # ???
    return tn3d.ravel()

def find_closest(center, center_cond_field, kdtree):
    centers, cond_field = center_cond_field
    _, index = kdtree.query(center)
    # print("index ", index)
    # print("centers[index] ", centers[index])
    # print("cond_field[index] ", cond_field[index])
    # exit()
    return cond_field[index]


def write_fields(mesh, basename, bulk_model, fracture_model, elid_to_fr, elids_same_value=None,
                 positions=None, input_tensors=None, center_cond_tn_field=None, interp_data=None):
    elem_ids = []
    cond_tn_field = []
    cond_tn_field_bulk = []
    cs_field = []
    fracture_cs = []
    fracture_len = []
    el_cond = {}
    centers = []
    centers_bulk = []
    n_fr = 0

    # rotation_matrix = np.array([[0, 1, 0],
    #                             [-1, 0, 0],
    #                             [0, 0, 1]])
    #rotation_matrix = np.array([[0, 1], [-1, 0]])

    # all_cond_tn = {}
    # all_ele_ids = {}
    #
    # indices = np.reshape(np.array(list(range(9))), (3, 3))
    #
    # rot = indices.T
    #
    # rot = rot.flatten()
    # print("rot flatten ", rot)
    #
    # all_ell_ids = {17: 0, 18: 0, 19: 1, 20: 1, 21: 2, 22: 2, 23: 3, 24: 3, 25: 4, 26: 4, 27: 5, 28: 5, 29: 6, 30: 6,
    #                31: 7, 32: 7, 33: 8, 34: 8}
    #
    # all_cond_tn = {0: np.array([[8.77108150e-07, -1.08600165e-07],
    #                             [-1.08600165e-07, 6.58661058e-07]]), 1: np.array([[4.92966922e-07, -3.66120247e-10],
    #                                                                               [-3.66120247e-10, 4.91705754e-07]]),
    #                2: np.array([[4.36438326e-07, 2.24193169e-07],
    #                             [2.24193169e-07, 4.09646465e-07]]), 3: np.array([[8.89458094e-07, 2.74144935e-07],
    #                                                                              [2.74144935e-07, 8.19394721e-07]]),
    #                4: np.array([[9.90674584e-07, 4.36280744e-07],
    #                             [4.36280744e-07, 1.24064447e-06]]), 5: np.array([[2.63761706e-06, 6.14196903e-07],
    #                                                                              [6.14196903e-07, 1.03941896e-06]]),
    #                6: np.array([[4.31686662e-07, -6.25398254e-08],
    #                             [-6.25398254e-08, 5.65524507e-07]]), 7: np.array([[1.17374197e-06, 2.34448527e-08],
    #                                                                               [2.34448527e-08, 1.14371008e-06]]),
    #                8: np.array([[1.46986308e-06, -3.47998338e-07],
    #                             [-3.47998338e-07, 1.53616856e-06]])}
    # i = 0

    # for el_id, ele in gmsh_mesh_bulk_elements(mesh):
    #     elem_ids.append(el_id)
    #     el_type, tags, node_ids = ele
    #     n_nodes = len(node_ids)
    #
    #     if n_nodes == 2:
    #         cs, cond_tn, fr_size = fracture_model.element_data(mesh, el_id, elid_to_fr)
    #         # print("fr cs: {}, cond_tn: {}".format(cs, cond_tn))
    #         fracture_cs.append(cs)
    #         fracture_len.append(fr_size)
    #         # sigma = cs * (cond_tn[0][0] /10**-6)
    #         # print("sigma: {}", sigma)
    #     else:
    #         # n_nodes == 3
    #         cs, cond_tn = bulk_model.element_data(mesh, el_id)
    #         # print("bulk cs: {}, cond_tn: {}".format(cs, cond_tn))
    #
    #         if elids_same_value is not None and el_id in elids_same_value:
    #             if elids_same_value[el_id] in elem_ids:
    #                 cond_tn = el_cond[elids_same_value[el_id]]
    #             # all_cond_tn.append(cond_tn)
    #         # print("cond tn ", cond_tn.shape)
    #         # exit()
    #         #
    #         # print("np.reshape(cond_tn, (3,3)) ", np.reshape(np.array(cond_tn), (3,3)))
    #         # exit()
    #         # cond_tn = rotation_matrix.T @ cond_tn @ rotation_matrix
    #         # cond_tn = cond_tn.flatten()
    #
    #         el_cond[el_id] = cond_tn
    #
    #     cs_field.append(np.array(cs))
    #     cond_tn_field.append(tensor_3d_flatten(cond_tn))
    #

    if center_cond_tn_field is not None:
        from scipy.spatial import KDTree
        centers_cnd, values_cnd = center_cond_tn_field
        kdtree = KDTree(centers_cnd)

    for el_id, ele in gmsh_mesh_bulk_elements(mesh):
        elem_ids.append(el_id)
        el_type, tags, node_ids = ele
        n_nodes = len(node_ids)
        center = np.average(np.array([mesh.nodes[i_node] for i_node in node_ids]), axis=0)

        if center_cond_tn_field is not None:
            cond_tn_flatten = find_closest(center, center_cond_tn_field, kdtree)

        if interp_data is not None:
            eids = interp_data[0][1]
            cond_values = interp_data[1]
            el_index = eids.index(el_id)
            cond_tn = cond_values[el_index]

            cond_tn_flatten = np.zeros(9)
            cond_tn_flatten[0] = cond_tn[0]
            cond_tn_flatten[1] = cond_tn[1]
            cond_tn_flatten[3] = cond_tn[1]
            cond_tn_flatten[4] = cond_tn[2]
            cond_tn_flatten[8] = 1

        if n_nodes == 2:
            n_fr += 1
            cs, cond_tn, fr_size = fracture_model.element_data(mesh, el_id, elid_to_fr)
            #print("fr cs: {}, cond_tn: {}".format(cs, cond_tn))
            fracture_cs.append(cs)
            fracture_len.append(fr_size)
            # sigma = cs * (cond_tn[0][0] /10**-6)
            # print("sigma: {}", sigma)
        else:
            # n_nodes == 3
            #print("bulk model self. rf sample ", bulk_model._rf_sample)
            cs, cond_tn = bulk_model.element_data(mesh, el_id)
            #print("bulk model self. rf sample ", bulk_model._rf_sample)
            #exit()

            if input_tensors is not None:
                indices = positions[el_id]
                cond_tn = input_tensors[:, indices[0], indices[1]]
                cond_tn = [[cond_tn[0], cond_tn[1]], [cond_tn[1], cond_tn[2]]]
                #print("cond_tn ", cond_tn)

            #cs = 1.0
            #cond_tn = all_cond_tn[rot[all_ell_ids[el_id]]]
            # i += 1
            # print("bulk cs: {}, cond_tn: {}".format(cs, cond_tn))
            if elids_same_value is not None and el_id in elids_same_value:
                if elids_same_value[el_id] in elem_ids:
                    cond_tn = el_cond[elids_same_value[el_id]]
                    #all_ele_ids[el_id] = all_ele_ids[elids_same_value[el_id]]
            # else:
            #     all_cond_tn[i] = cond_tn
            #     all_ele_ids[el_id] = i
            #     i += 1
            # print("cond tn ", cond_tn.shape)
            # exit()
            #
            # print("np.reshape(cond_tn, (3,3)) ", np.reshape(np.array(cond_tn), (3,3)))
            # exit()
            #cond_tn = rotation_matrix.dot(cond_tn) #@ rotation_matrix
            #cond_tn = rotation_matrix.T @ cond_tn @ rotation_matrix

            #print("cond tn ", cond_tn)
            #print("cond tn 2 ", cond_tn_2)
            #cond_tn = cond_tn.flatten()

            el_cond[el_id] = cond_tn

        cs_field.append(np.array(cs))
        if n_nodes == 2:
            cond_tn_field.append(tensor_3d_flatten(cond_tn))
        else:
            if center_cond_tn_field is not None or interp_data is not None:
                cond_tn_field.append(cond_tn_flatten)
            else:
                cond_tn_field.append(tensor_3d_flatten(cond_tn))
            centers_bulk.append(center)
            cond_tn_field_bulk.append(cond_tn_field[-1])

        centers.append(center)

    if n_fr == 0:
        print("NO FRACTURES")

    # print("centers ", centers)
    # exit()
    # #print("mesh nodes ", )
    # print("el cond ", el_cond)

    # print("all_cond_tn ", all_cond_tn)
    # print("all_ell_ids ", all_ele_ids)
    # exit()
    #print("cond_tn_field ", cond_tn_field)
    # cond_tn_field_arr = np.array(cond_tn_field)
    # print("cond_tn_field_arr ", cond_tn_field_arr.shape)
    # np.reshape(cond_tn_field_arr, ())
    # exit()

    fname = fields_file(basename)
    with open(fname, "w") as fout:
        mesh.write_ascii(fout)

        # print("fout ", fout)
        # print("elem_ids ", elem_ids)
        # print("cond tn field ", cond_tn_field)
        # print("len cond tn field ", len(cond_tn_field))

        # print("cond_tn_field ", cond_tn_field)
        # print("cross section ", cs_field)

        mesh.write_element_data(fout, elem_ids, 'conductivity_tensor', np.array(cond_tn_field))
        mesh.write_element_data(fout, elem_ids, 'cross_section', np.array(cs_field).reshape(-1, 1))

    # print("fracture_cs ", fracture_cs)
    # exit()
    center_cond = (centers_bulk, cond_tn_field_bulk)
    return elem_ids, cs_field, cond_tn_field, fracture_cs, fracture_len, center_cond


@attr.s(auto_attribs=True)
class FlowProblem:
    basename: str
    # Basename for files of this flow problem.
    fr_range: Tuple[float, float]
    # Fracture range to extract from the full list of the generated fractures.
    fractures: List[Any]
    # The Fractures object with generated fractures.
    bulk_model: BulkBase
    # The bulk model (specific for fine, coarse, etc.)
    config_dict: Dict[str, Any]
    # global config dictionary.

    regions: List[Region] = attr.ib(factory=list)
    # List of regions used in the geometry and mesh preparation
    side_regions: List[Region] = attr.ib(factory=list)
    # separate regions for sides of the outer wire, with fracture subregions and normals
    # used for creating boundary region set and for boundary averaging of the conductivity tensor
    reg_to_fr: Dict[int, int] = attr.ib(factory=dict)
    # Maps region id to original fracture id
    reg_to_group: List[Tuple[int, int]] = attr.ib(factory=dict)
    # Groups of regions for which the effective conductivity tensor will be computed separately.
    # One group is specified by the tuple of bulk and fracture region ID.
    group_positions: Dict[int, np.array] = attr.ib(factory=dict)
    # Centers of macro elements.
    skip_decomposition: bool = False

    # created later
    mesh: gmsh_io.GmshIO = None

    # safe conductivities produced by `make_fields`
    _elem_ids: Any = None
    _cond_tn_field: Any = None
    _fracture_cs: Any = None
    _fracture_len: Any = None

    pressure_loads: Any = None

    @classmethod
    def make_fine(cls, fr_range, fractures, config_dict):
        bulk_conductivity = config_dict["sim_config"]['bulk_conductivity']

        if "cond_tn_pop_file" in config_dict["fine"] or "pred_cond_tn_pop_file" in config_dict["fine"]:
            #@TODO: sample from saved population of conductivity tensors

            bulk_conductivity = config_dict["sim_config"]['bulk_conductivity']
            config_dict["mean_log_conductivity"] = bulk_conductivity["mean_log_conductivity"]
            bulk_model = BulkHomogenizationFineSample(config_dict)



            # bulk_cond_tn_pop_file = config_dict["fine"]["cond_tn_pop_file"]
            #bulk_model = BulkChoose(finer_level_path)
        else:
            #print("**bulk_conductivity ", bulk_conductivity)
            if "marginal_distr" in bulk_conductivity and bulk_conductivity["marginal_distr"] is not False:
                means, cov = FlowProblem.calculate_cov(bulk_conductivity["marginal_distr"])

                bulk_conductivity["mean_log_conductivity"] = means
                bulk_conductivity["cov_log_conductivity"] = cov
                del bulk_conductivity["marginal_distr"]

            #print("BULKFIelds bulk_conductivity ", bulk_conductivity)
            if "gstools" in config_dict["sim_config"] and config_dict["sim_config"]["gstools"]:
                #print("**bulk_conductivity ", bulk_conductivity)
                bulk_model = BulkFieldsGSTools(**bulk_conductivity)
            else:
                bulk_model = BulkFields(**bulk_conductivity)
        return FlowProblem("fine", fr_range, fractures, bulk_model, config_dict)


    @staticmethod
    def calculate_cov(marginal_distrs):
        #@TODO: 3D case not supported yet
        corr_coeff = marginal_distrs["corr_coeff"]

        cov = np.zeros((marginal_distrs["n_marginals"], marginal_distrs["n_marginals"]))
        means = np.zeros((marginal_distrs["n_marginals"]))
        for i in range(marginal_distrs["n_marginals"]):
            if "marginal_{}".format(i) in marginal_distrs:
                cov[i,i] = marginal_distrs["marginal_{}".format(i)]["std_log_conductivity"] ** 2
                means[i] = marginal_distrs["marginal_{}".format(i)]["mean_log_conductivity"]

        cov[1, 0] = cov[0, 1] = corr_coeff * np.sqrt(cov[0,0] * cov[1,1])

        return means, cov


    # @classmethod
    # def make_coarse(cls, fr_range, fractures, micro_scale_problem, config_dict):
    #     bulk_model = BulkMicroScale(micro_scale_problem)
    #     return FlowProblem("coarse",
    #                        fr_range, fractures, bulk_model, config_dict)

    @classmethod
    def make_coarse(cls, fr_range, fractures, config_dict):
        #bulk_model = BulkMicroScale(micro_scale_problem)
        #print("coarse fr range ", fr_range)
        bulk_conductivity = config_dict["sim_config"]['bulk_conductivity']
        config_dict["mean_log_conductivity"] = bulk_conductivity["mean_log_conductivity"]
        bulk_model = BulkHomogenization(config_dict)

        return FlowProblem("coarse",
                           fr_range, fractures, bulk_model, config_dict)

    @classmethod
    def make_microscale(cls, fr_range, fractures, fine_flow, config_dict):
        # use bulk fields from the fine level
        bulk_model = BulkFromFine(fine_flow)

        return FlowProblem("coarse_ref", fr_range, fractures, bulk_model, config_dict)

    @property
    def pressure_loads(self):
        return self.thread.p_loads

    def add_region(self, name, dim, mesh_step=0.0, boundary=False):
        #print("mesh step ", mesh_step)
        reg = Region(name, dim, boundary, mesh_step)
        reg.id = len(self.regions)
        self.regions.append(reg)
        return reg

    def init_decomposition(self, outer_polygon, bulk_reg, tol):
        """
        Create polygon decomposition and add the outer polygon.
        :param outer_polygon: [np.array[2], ..] Vertices of the outer polygon.
        :return: (PolygonDecomposition, side_regions)
        """
        pd = polygons.PolygonDecomposition(tol)
        last_pt = outer_polygon[-1]
        side_regions = []
        for i_side, pt in enumerate(outer_polygon):
            reg = self.add_region(".side_{}".format(i_side), dim=1, mesh_step=self.mesh_step, boundary=True)

            reg.sub_reg = self.add_region(".side_fr_{}".format(i_side), dim=0, mesh_step=self.mesh_step, boundary=True)
            diff = np.array(pt) - np.array(last_pt)
            normal = np.array([diff[1], -diff[0]])
            reg.normal = normal / np.linalg.norm(normal)
            side_regions.append(reg)

            sub_segments = pd.add_line(last_pt, pt, deformability=0)

            if not (type(sub_segments) == list and len(sub_segments) == 1):
                from bgem.polygons.plot_polygons import plot_decomp_segments
                plot_decomp_segments(pd)

            for seg in sub_segments:
                seg.attr = reg
            last_pt = pt
            #assert type(sub_segments) == list and len(sub_segments) == 1, sub_segments
        assert len(pd.polygons) == 2
        pd.polygons[1].attr = bulk_reg


        return pd, side_regions

    def add_fractures(self, pd, fracture_lines, eid):
        outer_wire = pd.outer_polygon.outer_wire.childs
        #print("outer wire ", outer_wire)

        #print("len fracture lines ", len(fracture_lines))

        assert len(outer_wire) == 1
        outer_wire = next(iter(outer_wire))
        fracture_regions = []
        for i_fr, (p0, p1) in fracture_lines.items():
            reg = self.add_region("fr_{}".format(i_fr), dim=1, mesh_step=self.mesh_step)
            #print("i_fr: {}, reg.id: {}".format(i_fr, reg.id))
            self.reg_to_fr[reg.id] = i_fr
            fracture_regions.append(reg)
            if self.skip_decomposition:
                continue
            # print("    ", i_fr, "fr size:", np.linalg.norm(p1 - p0))
            try:
                # pd.decomp.check_consistency()
                # if eid == 0 and i_fr == 5:
                #      print("stop")
                #      # plot_decomp_segments(pd, [p0, p1])
                # TODO: make short fractures more deformable
                sub_segments = pd.add_line(p0, p1, deformability=1)
            except Exception as e:
                # new_points = [pt for seg in segments for pt in seg.vtxs]
                print('Decomp Error, dir: {} base: {} eid: {}  i_fr: {}'.format(os.getcwd(), self.basename, eid, i_fr))
                traceback.print_exc()
                # plot_decomp_segments(pd, [p0, p1])
                # raise e
                # print(e)
                # pass
            # pd.decomp.check_consistency()

            # remove segments out of the outer polygon
            if type(sub_segments) == list:
                # print("sub segments ", sub_segments)
                # exit()
                for seg in sub_segments:
                    if seg.attr is None:
                        seg.attr = reg
                    # remove segments poking out of outer polygon
                    if seg.wire[0] == seg.wire[1] and (seg.wire[0].parent == pd.outer_polygon.outer_wire):
                        points = seg.vtxs
                        # print("pd segments ", pd.segments.keys())
                        # print("seg ", seg)
                        if seg.id in pd.segments.keys():
                            pd.delete_segment(seg)
                            for pt in points:
                                if pt.is_free():  # and pt.id in pd.points.keys():
                                    pd._rm_point(pt)
                                # else:
                                #     print("not rm pt.id ", pt.id)

        # plot_decomp_segments(pd, [p0, p1])
        # assign boundary region to outer polygon points
        for seg, side in outer_wire.segments():
            side_reg = seg.attr
            if side_reg.boundary:
                assert hasattr(side_reg, "sub_reg")
                seg.vtxs[side].attr = side_reg.sub_reg
        # none region to remaining
        for shape_list in pd.decomp.shapes:
            for shape in shape_list.values():
                if shape.attr is None:
                    shape.attr = self.none_reg

        # # plot_decomp_segments(pd)
        return pd, fracture_regions

    def make_fracture_network(self):
        self.mesh_step = self.fr_range[0]

        # Init regions
        self.none_reg = self.add_region('none', dim=-1)
        bulk_reg = self.add_region('bulk_2d', dim=2, mesh_step=self.mesh_step)
        self.reg_to_group[bulk_reg.id] = 0

        # make outer polygon
        geom = self.config_dict["sim_config"]["geometry"]
        lx, ly = geom["domain_box"]

        if "outer_polygon" in geom:
            self.outer_polygon = geom["outer_polygon"]
        else:
            self.outer_polygon = [[-lx / 2, -ly / 2], [+lx / 2, -ly / 2], [+lx / 2, +ly / 2], [-lx / 2, +ly / 2]]

        #self.outer_polygon = [[-lx / 2, 0], [0, 0], [0, +ly / 2], [-lx / 2, +ly / 2]]

        #self.outer_polygon = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
        #self.outer_polygon = [[-500, 0], [0, 0], [0, 500], [-500, 500]]

        #print("outer polygon ", self.outer_polygon)

        pd, self.side_regions = self.init_decomposition(self.outer_polygon, bulk_reg, tol=self.mesh_step)
        self.group_positions[0] = np.mean(self.outer_polygon, axis=0)

        square_fr_range = [self.fr_range[0], np.min([self.fr_range[1], self.outer_polygon[1][0] - self.outer_polygon[0][0]])]
        #square_fr_range = [10, square_fr_range[1]]
        #print("square fr range ", square_fr_range)

        # if self.config_dict.get("homogenization", False):
        #     coarse_step = self.config_dict["coarse"]["step"] * 2
        #     square_fr_range[0] = coarse_step

        #print("square_fr_range ", square_fr_range)
        #exit()


        #self.fracture_lines = self.fractures.get_lines(self.fr_range)
        self.fracture_lines = self.fractures.get_lines(square_fr_range)
        #print("fracture lines ", self.fracture_lines)
        #print("len fracture lines ", len(self.fracture_lines))
        # print("list fractures ", list(self.fracture_lines.values()))

        pd, fr_regions = self.add_fractures(pd, self.fracture_lines, eid=0)
        #self.reg_to_group[fr_region.id] = 0

        for reg in fr_regions:
            self.reg_to_group[reg.id] = 0
        self.decomp = pd

    # def _make_mesh_window(self):
    #     import geometry_2d as geom
    #
    #     self.skip_decomposition = False
    #
    #     self.make_fracture_network()
    #
    #     gmsh_executable = self.config_dict["gmsh_executable"]
    #
    #     g2d = geom.Geometry2d("mesh_window_" + self.basename, self.regions)
    #     g2d.add_compoud(self.decomp)
    #     g2d.make_brep_geometry()
    #     step_range = (self.mesh_step * 0.9, self.mesh_step * 1.1)
    #     g2d.call_gmsh(gmsh_executable, step_range)
    #     self.mesh = g2d.modify_mesh()


    def get_subdomain(self, outer_polygon):
        self.mesh

    def make_mesh(self, mesh_file=None):
        import homogenization.geometry_2d as geom
        if mesh_file is None:
            mesh_file = "mesh_{}.msh".format(self.basename)

        #print("os.getcwd() ", os.getcwd())
        #print("mesh file ", mesh_file)
        self.skip_decomposition = os.path.exists(mesh_file)

        #print("self. skip decomposition ", self.skip_decomposition)

        #print("self regions ", self.regions)
        if not self.skip_decomposition:
            self.make_fracture_network()
            gmsh_executable = self.config_dict["sim_config"]["gmsh_executable"]
            g2d = geom.Geometry2d("mesh_" + self.basename, self.regions)
            g2d.add_compoud(self.decomp)
            g2d.make_brep_geometry()
            step_range = (self.mesh_step * 0.9, self.mesh_step * 1.1)
            g2d.call_gmsh(gmsh_executable, step_range)
            #print("self.reg_to_fr ", self.reg_to_fr)
            self.mesh, self.elid_to_fr = g2d.modify_mesh(self.reg_to_fr)
            #print("self.elid_to_fr ", self.elid_to_fr)
        elif self.skip_decomposition and self.config_dict["sim_config"]["mesh_window"]:
            self._make_mesh_window()
        else:
            self.elid_to_fr = {}
            self.make_fracture_network()
            self.mesh = gmsh_io.GmshIO()
            with open(mesh_file, "r") as f:
                self.mesh.read(f)

    def get_centers(self, mesh):
        centers = []
        elem_ids = []
        for el_id, ele in gmsh_mesh_bulk_elements(mesh):
            elem_ids.append(el_id)
            el_type, tags, node_ids = ele
            #n_nodes = len(node_ids)
            center = np.average(np.array([mesh.nodes[i_node] for i_node in node_ids]), axis=0)
            centers.append(center)
        return [centers, elem_ids]

    def interpolate_fields(self, center_cond_tn_field, mode="linear"):
        fracture_model = FractureModel(self.fractures, self.reg_to_fr,
                                       **self.config_dict["sim_config"]['fracture_model'],
                                       bulk_model=self.bulk_model)

        mesh_centers_eids = self.get_centers(self.mesh)

        # print("center_cond_tn_field[0] ", center_cond_tn_field[0])
        # print("center_cond_tn_field[1] ", center_cond_tn_field[1])
        # print("mesh_centers_eids[0] ", mesh_centers_eids[0])

        #print("center cond tn field ", center_cond_tn_field)

        grid_fine = np.array(center_cond_tn_field[0])[:, :2]
        values = np.array(center_cond_tn_field[1])[:, [0,1,4]]
        grid_hom = np.array(mesh_centers_eids[0])[:, :2]

        # print("grid fine ", grid_fine)
        # # print("values ", values)
        # print("grid hom ", grid_hom)
        # print("grid fine ", grid_fine.shape)
        # print("values ", values.shape)
        # print("grid hom ", grid_hom.shape)
        # x_fine = grid_fine[:, 0]
        # y_fine = grid_fine[:, 1]
        #
        # xy = np.zeros((2, np.size(x_fine)))
        # xy[0] = x_fine
        # xy[1] = y_fine
        # xy = xy.T

        tria = sc_spatial.Delaunay(grid_fine)
        mean_val = np.mean(values)

        interp = sc_interpolate.LinearNDInterpolator(tria, values, fill_value=0)
        value_hom = interp(grid_hom)

        for index, row in enumerate(value_hom):
            if np.alltrue(row == 0):
                v = [mean_val, 0, mean_val]
                value_hom[index] = v

        interp_data = [mesh_centers_eids, value_hom]

        elem_ids, cs_field, cond_tn_field, fracture_cs, fracture_len, center_cond = write_fields(self.mesh,
                                                                                                 self.basename,
                                                                                                 self.bulk_model,
                                                                                                 fracture_model,
                                                                                                 self.elid_to_fr,
                                                                                                 center_cond_tn_field=center_cond_tn_field,
                                                                                                 interp_data=interp_data,
                                                                                                 )

        self._elem_ids = elem_ids
        self._cond_tn_field = cond_tn_field
        self._fracture_cs = fracture_cs
        self._fracture_len = fracture_len
        self._center_cond = center_cond

    def make_fields(self, elids_same_value=None, positions=None, input_tensors=None):
        """
        Calculate the conductivity and the cross-section fields, write into a GMSH file.

        :param cond_tensors_2d: Dictionary of the conductivities determined from a subscale
        calculation.
        :param cond_2d_samples: Array Nx2x2 of 2d tensor samples from own and other subsample problems.
        :return:
        """
        fracture_model = FractureModel(self.fractures, self.reg_to_fr,
                                       **self.config_dict["sim_config"]['fracture_model'],
                                       bulk_model=self.bulk_model)
        elem_ids, cs_field, cond_tn_field, fracture_cs, fracture_len, center_cond = write_fields(self.mesh, self.basename,
                                                                                    self.bulk_model, fracture_model,
                                                                                    self.elid_to_fr,
                                                                                    elids_same_value,
                                                                                    positions,
                                                                                    input_tensors)

        self._elem_ids = elem_ids
        self._cond_tn_field = cond_tn_field
        self._fracture_cs = fracture_cs
        self._fracture_len = fracture_len
        self._center_cond = center_cond


    def bulk_field(self):
        assert self._elem_ids is not None

        points = []
        values00 = []
        values11 = []
        values01 = []
        for eid, tensor in zip(self._elem_ids, self._cond_tn_field):
            el_type, tags, node_ids = self.mesh.elements[eid]
            if len(node_ids) > 2:
                center = np.mean([np.array(self.mesh.nodes[nid]) for nid in node_ids], axis=0)
                points.append(center[0:2])
                # tensor is flatten 3x3
                values00.append(tensor[0])
                values01.append(tensor[1])
                values11.append(tensor[4])

        return np.array(points), np.array([values00, values01, values11])

    # def elementwise_mesh(self, coarse_mesh, mesh_step, bounding_polygon):
    #     import geometry_2d as geom
    #     from bgem.polygons.plot_polygons import plot_decomp_segments
    #
    #     mesh_file = "mesh_{}.msh".format(self.basename)
    #     if os.path.exists(mesh_file):
    #         # just initialize reg_to_group map
    #         self.skip_decomposition = True
    #
    #     self.mesh_step = mesh_step
    #     self.none_reg = self.add_region('none', dim=-1)
    #     # centers of macro elements
    #
    #     self.reg_to_group = {}  # bulk and fracture region id to coarse element id
    #     g2d = geom.Geometry2d("mesh_" + self.basename, self.regions, bounding_polygon)
    #     for eid, (tele, tags, nodes) in coarse_mesh.elements.items():
    #         # eid = 319
    #         # (tele, tags, nodes) = coarse_mesh.elements[eid]
    #         #print("Geometry for eid: ", eid)
    #         if tele != 2:
    #             continue
    #         prefix = "el_{:03d}_".format(eid)
    #         outer_polygon = np.array([coarse_mesh.nodes[nid][:2] for nid in nodes])
    #         # set mesh step to maximal height of the triangle
    #         area = np.linalg.norm(np.cross(outer_polygon[1] - outer_polygon[0], outer_polygon[2] - outer_polygon[0]))
    #
    #         self.mesh_step = min(self.mesh_step, area / np.linalg.norm(outer_polygon[1] - outer_polygon[0]))
    #         self.mesh_step = min(self.mesh_step, area / np.linalg.norm(outer_polygon[2] - outer_polygon[1]))
    #         self.mesh_step = min(self.mesh_step, area / np.linalg.norm(outer_polygon[0] - outer_polygon[2]))
    #
    #         self.group_positions[eid] = np.mean(outer_polygon, axis=0)
    #         #edge_sizes = np.linalg.norm(outer_polygon[:, :] - np.roll(outer_polygon, -1, axis=0), axis=1)
    #         #diam = np.max(edge_sizes)
    #
    #         bulk_reg = self.add_region(prefix + "bulk_2d_", dim=2, mesh_step=self.mesh_step)
    #         self.reg_to_group[bulk_reg.id] = eid
    #         # create regions
    #         # outer polygon
    #         normals = []
    #         shifts = []
    #         if eid==1353:
    #             print("break")
    #         pd, side_regions = self.init_decomposition(outer_polygon, bulk_reg, tol = self.mesh_step*0.8)
    #         for i_side, side_reg in enumerate(side_regions):
    #             side_reg.name = "." + prefix + side_reg.name[1:]
    #             side_reg.sub_reg.name = "." + prefix + side_reg.sub_reg.name[1:]
    #             self.reg_to_group[side_reg.id] = eid
    #             normals.append(side_reg.normal)
    #             shifts.append(side_reg.normal @ outer_polygon[i_side])
    #         self.side_regions.extend(side_regions)
    #
    #         # extract fracture lines larger then the mesh step
    #         fracture_lines = self.fractures.get_lines(self.fr_range)
    #         line_candidates = {}
    #         for i_fr, (p0, p1) in fracture_lines.items():
    #             for n, d in zip(normals, shifts):
    #                 sgn0 = n @ p0 - d > 0
    #                 sgn1 = n @ p1 - d > 0
    #                 #print(sgn0, sgn1)
    #                 if sgn0 and sgn1:
    #                     break
    #             else:
    #                 line_candidates[i_fr] = (p0, p1)
    #                 #print("Add ", i_fr)
    #             #plot_decomp_segments(pd, [p0, p1])
    #
    #
    #         pd, fr_regions = self.add_fractures(pd, line_candidates, eid)
    #         for reg in fr_regions:
    #             reg.name = prefix + reg.name
    #             self.reg_to_group[reg.id] = eid
    #         if not self.skip_decomposition:
    #             g2d.add_compoud(pd)
    #
    #     if self.skip_decomposition:
    #         self.mesh = gmsh_io.GmshIO()
    #         with open(mesh_file, "r") as f:
    #             self.mesh.read(f)
    #         return
    #
    #     g2d.make_brep_geometry()
    #     step_range = (self.mesh_step * 0.9, self.mesh_step *1.1)
    #     gmsh_executable = self.config_dict["gmsh_executable"]
    #     g2d.call_gmsh(gmsh_executable, step_range)
    #     self.mesh = g2d.modify_mesh()

    def run(self):
        outer_reg_names = []
        for reg in self.side_regions:
            outer_reg_names.append(reg.name)
            outer_reg_names.append(reg.sub_reg.name)
        self.thread = FlowThread(self.basename, outer_reg_names, self.config_dict)

        self.thread.start()
        return self.thread

    # def effective_tensor_from_balance(self, side_regions):
    #     """
    #     :param mesh: GmshIO mesh object.
    #     :param side_regions: List of side regions with the "normal" attribute.
    #     :return:
    #     """
    #     loads = self.pressure_loads
    #     with open(os.path.join(self.basename, "water_balance.yaml")) as f:
    #         balance = yaml.load(f, Loader=yaml.FullLoader)['data']
    #     flux_response = np.zeros_like(loads)
    #     reg_map = {}
    #     for reg in side_regions:
    #         reg_map[reg.name] = reg
    #         reg_map[reg.sub_reg.name] = reg
    #     for entry in balance:
    #         reg = reg_map.get(entry['region'], None)
    #         bc_influx = entry['data'][0]
    #         if reg is not None:
    #             flux_response[entry['time']] += reg.normal * bc_influx
    #     flux_response /= len(side_regions)
    #     #flux_response *= np.array([100, 1])[None, :]
    #
    #
    #     # least square fit for the symmetric conductivity tensor
    #     rhs = flux_response.flatten()
    #     # columns for the tensor values: C00, C01, C11
    #     pressure_matrix = np.zeros((len(rhs), 3))
    #     for i_load, (p0, p1) in enumerate(loads):
    #         i0 = 2 * i_load
    #         i1 = i0 + 1
    #         pressure_matrix[i0] = [p0, p1, 0]
    #         pressure_matrix[i1] = [0, p0, p1]
    #     C = np.linalg.lstsq(pressure_matrix, rhs, rcond=None)[0]
    #     cond_tn = np.array([[C[0], C[1]], [C[1], C[2]]])
    #     self.plot_effective_tensor(flux_response, cond_tn, label)
    #     #print(cond_tn)
    #     return cond_tn

    def element_volume(self, mesh, nodes):
        nodes = np.array([mesh.nodes[nid] for nid in  nodes])
        if len(nodes) == 1:
            return 0
        elif len(nodes) == 2:
            return np.linalg.norm(nodes[1] - nodes[0])
        elif len(nodes) == 3:
            return np.linalg.norm(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0]))
        else:
            assert False

    @staticmethod
    def line_fit(fluxes):
        fit_line = np.poly1d(np.polyfit(fluxes[:, 0], fluxes[:, 1], 1))

        x = range(-10, 10)
        fit_line_values = fit_line(x)
        # print("x ", x)
        # print("fit line values ", fit_line_values)
        angle = np.arctan2(fit_line_values[-1] - fit_line_values[0], x[-1] - x[0])

        theta = np.radians(np.rad2deg(angle))
        rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))

        #line_data = np.hstack((np.vstack(x), np.vstack(fit_line_values)))
        # fluxes = np.hstack((np.vstack(x), np.vstack(fit_line_values)))

        #rotated_line = line_data.dot(rotation_matrix)
        rotated_fluxes = fluxes.dot(rotation_matrix)
        c_x = np.max(np.abs(rotated_fluxes[:, 0]))
        c_y = np.max(np.abs(rotated_fluxes[:, 1]))



        e_val = [c_x, c_y]
        #print("eval c_x, c_y", e_val)

        return e_val
    @staticmethod
    def get_node_features(fields_mesh, feature_names):
        """
        Extract mesh from file
        :param fields_mesh: Mesh file
        :param feature_names: [[], []] - fields in each sublist are joint to one feature, each sublist corresponds to one vertex feature
        :return: list
        """
        mesh = gmsh_io.GmshIO(fields_mesh)
        features = []
        all_fields = []
        for f_names in feature_names:
            all_fields.append(mesh._fields)
            joint_features = join_fields(mesh._fields, f_names)
            features.append(list(joint_features.values()))

        return np.array(features).T, all_fields

    def effective_tensor_from_bulk(self, pressure_loads=None, reg_to_group_1=None, basename=None, elids_same_value=None):
        """
        :param bulk_regions: mapping reg_id -> tensor_group_id, groups of regions for which the tensor will be computed.
        :return: {group_id: conductivity_tensor} List of effective tensors.
        """
        sym_condition = False
        bulk_regions = self.reg_to_group

        if pressure_loads is None:
            pressure_loads = self.pressure_loads

        # print("bulk regions ", bulk_regions)
        # print("self.regions ", self.regions)
        # print("self.reg_to_group ", self.reg_to_group)
        #print("self.pressure_loads ", pressure_loads)

        #compute_effective_cond.effective_tensor_from_bulk(self.regions, bulk_regions, self.pressure_loads, os.path.join(self.basename, "flow_fields.msh"))
        #print("bulk regions ", bulk_regions)

        out_mesh = gmsh_io.GmshIO()
        with open("flow_fields.msh", "r") as f:
            out_mesh.read(f)
        time_idx = 0
        time, field_cs = out_mesh.element_data['cross_section'][time_idx]
        ele_reg_vol = {eid: (tags[0] - 10000, self.element_volume(out_mesh, nodes))
                       for eid, (tele, tags, nodes) in out_mesh.elements.items()}

        ###
        # Get conductivity tensor
        ###
        # feature_names = [['conductivity_tensor'], ['cross_section']]
        # FlowProblem.get_node_features("fields_fine.msh", feature_names)


        assert len(field_cs) == len(ele_reg_vol)
        velocity_field = out_mesh.element_data['velocity_p0']
        #print("velocity field ", velocity_field)

        loads = pressure_loads
        #print("loads ", loads)
        group_idx = {group_id: i_group for i_group, group_id in enumerate(set(bulk_regions.values()))}
        n_groups = len(group_idx)
        #print("n groups ", n_groups)
        group_labels = n_groups * ['_']
        for reg_id, group_id in bulk_regions.items():
            i_group = group_idx[group_id]
            old_label = group_labels[i_group]
            new_label = self.regions[reg_id].name
            group_labels[i_group] = old_label if len(old_label) > len(new_label) else new_label

        n_directions = len(loads)
        flux_response = np.zeros((n_groups, n_directions, 2))
        area = np.zeros((n_groups, n_directions))
        print("Averaging velocities ...")
        for i_time, (time, velocity) in velocity_field.items():
            # for eid, ele_vel in velocity.items():
            #     print("eid : {}, ele_vel: {}".format(eid, ele_vel))
            #     reg_id, vol = ele_reg_vol[eid]
            #     cs = field_cs[eid][0]
            #     #print("vol: {}, cs: {}".format(vol, cs))
            #     volume = cs * vol
            #     i_group = group_idx[bulk_regions[reg_id]]
            #     flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:2]))
            #     area[i_group, i_time] += volume

            # print("elids ", elids_same_value)
            # if elids_same_value is not None:
            #     new_velocities = {}
            #     for e_id, e_id_2 in elids_same_value.items():
            #         vel_1 = velocity[e_id]
            #         vel_2 = velocity[e_id_2]
            #
            #         print("vel1 : {}, vel2: {}, mean vel: {}".format(vel_1, vel_2, np.mean([vel_1, vel_2], axis=0)))
            #
            #         new_velocities[e_id] = new_velocities[e_id_2] = np.mean([vel_1, vel_2], axis=0)
            #
            #     velocity = new_velocities

            #print("velocity ", velocity)

            for eid, ele_vel in velocity.items():
                #print("eid : {}, ele_vel: {}".format(eid, ele_vel))

                reg_id, vol = ele_reg_vol[eid]
                cs = field_cs[eid][0]
                #print("vol: {}, cs: {}".format(vol, cs))
                volume = cs * vol
                i_group = group_idx[bulk_regions[reg_id]]
                flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:2]))

                #neg_pressure = np.matmul(np.linalg.inv(cond_tn), velocities[e_id])

                area[i_group, i_time] += volume

        #print("flux response ", flux_response)
        #print("area ", area)

        flux_response /= area[:, :, None]
        cond_tensors = {}
        print("Fitting tensors ...")
        #print("flux response ", flux_response)

        #print("groud_idx ", group_idx.items())
        for group_id, i_group in group_idx.items():
            flux = flux_response[i_group]
            #print("flux ", flux)
            # least square fit for the symmetric conductivity tensor
            rhs = flux.flatten()
            # columns for the tensor values: C00, C01, C11
            pressure_matrix = np.zeros((len(rhs), 4))
            if sym_condition:
                pressure_matrix = np.zeros((len(rhs)+1, 4))

            for i_load, (p0, p1) in enumerate(loads):
                i0 = 2 * i_load
                i1 = i0 + 1
                #@TODO: skalarni soucin rychlosti s (p0, p1)
                # pressure_matrix[i0] = [p0, p1, 0]
                # pressure_matrix[i1] = [0, p0, p1]

                pressure_matrix[i0] = [p0, p1, 0, 0]
                pressure_matrix[i1] = [0, 0, p0, p1]

            if sym_condition:
                pressure_matrix[-1] = [0, 1, -1, 0]
                rhs = np.append(rhs, [0])

            #print("pressure matrix ", pressure_matrix)
            #print("rhs ", rhs)
            #print("rhs type ", type(rhs))
            #print("rhs ", rhs)

            #print("pressure matrix ", pressure_matrix)
            #print("rhs ", rhs)
            # exit()

            C, residuals, rank, sing_values = np.linalg.lstsq(pressure_matrix, rhs)

            # print("residuals ", residuals)
            # print("rank ", rank)
            # print("sing values ", sing_values)
            #exit()

            #C = np.linalg.lstsq(pressure_matrix, rhs)[0]
            #print("C ", C)
            #print("(C[1]-C[2])/(C[1]+C[2]/2) ", np.abs((C[1]-C[2])/((C[1]+C[2]/2))))
            #C[1] = C[1]
            #cond_tn = np.array([[C[0], C[1]], [C[1], C[2]]])
            cond_tn = np.array([[C[0], C[1]], [C[2], C[3]]])
            diff = np.abs((C[1]-C[2])/((C[1]+C[2]/2)))

            #print("cond tn ", cond_tn)
            #exit()

            #exit()

            e_val = None
            if residuals > 10**(-8):
                exit()
                e_val = FlowProblem.line_fit(flux)

            # if i_group < 10:
            #      # if flux.shape[0] < 5:
            #      print("Plot tensor for eid: ", group_id)
            #      print("Fluxes: \n", flux)
            #      print("pressures: \n", loads)
            #      print("cond: \n", cond_tn)
            #      #if e_val is None:
            #      e_val, e_vec = np.linalg.eigh(cond_tn)
            #      print("e val ", e_val)
            #
            #      #print("log mean eval cond", np.log10(np.mean(e_val)))
            #      self.plot_effective_tensor(flux, cond_tn, self.basename + "_" + group_labels[i_group])
            #      #exit()
            #      #print(cond_tn)
            cond_tensors[group_id] = cond_tn
        self.cond_tensors = cond_tensors
        self.flux = flux
        self.pressure_matrix = pressure_matrix
        self._bulk_regions = bulk_regions
        self._regions = self.regions
        self._pressure_loads = pressure_loads
        return cond_tensors, diff

    def labeled_arrow(self, ax, start, end, label):
        """
        Labeled and properly scaled arrow.
        :param start: origin point, [x,y]
        :param end: tip point [x,y]
        :param label: string label, placed near the tip
        :return:
        """
        scale = np.linalg.norm(end - start)
        ax.arrow(*start, *end, width=0.003 * scale, head_length=0.1 * scale, head_width =0.05 * scale)
        if (end - start)[1] > 0:
            vert_align = 'bottom'
        else:
            vert_align = 'top'
        ax.annotate(label, end + 0.1*(end - start), va=vert_align)

    def plot_effective_tensor(self, fluxes, cond_tn, label):
        """
        Plot response fluxes for pressure gradients with angles [0, pi) measured from the X-azis counterclockwise.
        :param fluxes: np.array of 2d fluex vectors.
        :return:
        """
        import matplotlib.pyplot as plt

        e_val, e_vec = np.linalg.eigh(cond_tn)
        #e_val = np.abs(e_val)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax = axes
        # setting the axis limits in [left, bottom, width, height]
        #rect = [0.1, 0.1, 0.8, 0.8]
        ax.set_aspect('equal')
        #ax_polar = fig.add_axes(rect, polar=True, frameon=False)
        continuous_loads = np.array([(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi, 1000)])
        X, Y = cond_tn @ continuous_loads.T
        # print("Fluxes: ", fluxes)
        ax.scatter(X, Y, c='green', s=0.1)
        ax.scatter(-X, -Y, c='green', s=0.1)
        ax.scatter(fluxes[:, 0], fluxes[:, 1], c='red', s=30, marker='+')
        ax.scatter(-fluxes[:, 0], -fluxes[:, 1], c='red', s=30, marker='+')
        lim = max(max(X), max(fluxes[:, 0]), max(Y), max(fluxes[:, 1])) * 1.2
        #ax.set_xlim(-lim, lim)
        #ax.set_ylim(-lim, lim)
        self.labeled_arrow(ax, [0,0], 0.9 * e_val[0] * e_vec[:, 0], "{:5.2g}".format(e_val[0]))
        self.labeled_arrow(ax, [0,0], 0.9 * e_val[1] * e_vec[:, 1], "{:5.2g}".format(e_val[1]))
        fig.suptitle("Conductivity tensor: {}".format(label))

        #ax_polar.grid(True)
        #fig.savefig("cond_tn_{}.pdf".format(label))
        plt.close(fig)
        plt.show()

    def summary(self):
        return dict(
            pos=[self.group_positions[eid].tolist() for eid in self.cond_tensors.keys()],
            cond_tn=[self.cond_tensors[eid].tolist() for eid in self.cond_tensors.keys()],
            #fracture_cs=self._fracture_cs,
            #fracture_len=self._fracture_len,
            flux=self.flux.tolist(),
            pressure_matrix=self.pressure_matrix.tolist(),
            #bulk_regions=self._bulk_regions,
            #regions=self._regions,
            pressure_loads=self._pressure_loads.tolist()
        )


class BothSample:

    def __init__(self, sample_config):
        """
        Attributes:
        seed,
        h_fine_step,
        h_coarse_step,
        do_coarse
        config_path
        :param sample_config:
        """
        # sample_config attributes:
        # finer_level_path - Path to the file with microscale tensors from the previous level. Used for sampling conductivity.
        # config_path
        # do_coarse
        # h_coarse_step
        # h_fine_step
        # seedm
        # i_level
        #print("sample config ", sample_config)

        self.__dict__.update(sample_config)


        #print("self levels ", self.levels)

        self.i_level = 1
        self.finer_level_path = None  # It should be set if config param 'choose_from_finer_level' is True
        self.h_fine_step = self.levels[1]["step"]
        self.h_coarse_step = self.levels[0]["step"]

        self.do_coarse = False

        np.random.seed(self.seed)
        # with open(self.config_path, "r") as f:
        #     self.config_dict = yaml.load(f) # , Loader=yaml.FullLoader
        self.config_dict = sample_config

    @staticmethod
    def excluded_area(r_min, r_max, kappa, coef=1):
        norm_coef = kappa / (r_min**(-kappa) - r_max**(-kappa))
        return coef * (norm_coef * (r_max**(1-kappa) - r_min**(1-kappa))/ (1-kappa))**2

    @staticmethod
    def calculate_mean_excluded_volume(r_min, r_max, kappa, geom=False):
        #print("r min ", r_min)
        #print("r max ", r_max)
        #print("kappa ", kappa)
        if geom:
            #return 0.5 * (kappa / (r_min**(-kappa) - r_max**(-kappa)))**2 * 2*(((r_max**(2-kappa) - r_min**(2-kappa))) * ((r_max**(1-kappa) - r_min**(1-kappa))))/(kappa**2 - 3*kappa + 2)
            return ((r_max ** (1.5 * kappa - 0.5) - r_min ** (1.5 * kappa - 0.5)) / (
                        (-1.5 * kappa - 0.5) * (r_min ** (-1 * kappa) - r_max ** (-1 * kappa)))) ** 2
        else:
            return 0.5 * (kappa / (r_min ** (-kappa) - r_max ** (-kappa))) ** 2 * 2 * (((r_max ** (2 - kappa) - r_min ** (2 - kappa))) * ((r_max ** (1 - kappa) - r_min ** (1 - kappa)))) / (kappa ** 2 - 3 * kappa + 2)

    @staticmethod
    def calculate_mean_fracture_size(r_min, r_max, kappa, power=1):
        f0 = (r_min ** (-kappa) - r_max**(-kappa))/kappa
        return (1/f0) * (r_max**(-kappa+power) - r_min**(-kappa+power))/(-kappa+power)

    def generate_fractures(self):
        geom = self.config_dict["geometry"]
        lx, ly = geom["fractures_box"]
        fr_size_range = geom["pow_law_size_range"]
        pow_law_exp_3d = geom["pow_law_size_exp"]
        pow_law_sample_range = geom["pow_law_sample_range"]
        rho = False
        rho_2D = False
        if "rho" in geom:
            rho = geom["rho"]  # P_30 * V_ex
        if "rho_2D" in geom:
            rho_2D = geom["rho_2D"]  # P_30 * V_ex
        n_frac_limit = geom["n_frac_limit"]
        #print("n frac limit ", n_frac_limit)
        #print("fr size range ", fr_size_range)
        p_32 = geom["p_32"]

        #print("lx: {}, ly: {} ".format(lx, ly))

        # generate fracture set
        fracture_box = [lx, ly, 0]
        area = lx * ly

        #print("pow_law_sample_range ", pow_law_sample_range)

        if rho_2D is not False:
            #print("rho 2d is not false")
            #exit()
            self.config_dict["fracture_model"]["max_fr"] = pow_law_sample_range[1]
            A_ex = BothSample.excluded_area(pow_law_sample_range[0], pow_law_sample_range[1],
                                            kappa=pow_law_exp_3d-1, coef=np.pi/2)
            #print("A_ex ", A_ex)

            # rho_2D = N_f/A * A_ex, N_f/A = intensity
            #print("rho_2D ", rho_2D)

            intensity = rho_2D / A_ex
            #print("intensity ", intensity)

            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize(pow_law_exp_3d - 1, pow_law_sample_range, intensity)
                           )

            pop.set_sample_range(pow_law_sample_range)

        elif rho is not False:
            V_ex = BothSample.calculate_mean_excluded_volume(r_min=fr_size_range[0],
                                                             r_max=fr_size_range[1],
                                                             kappa=pow_law_exp_3d-1)

            v_ex = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
                                                             r_max=fr_size_range[1],
                                                             kappa=pow_law_exp_3d - 1, power=1)

            R2 = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
                                                         r_max=fr_size_range[1],
                                                         kappa=pow_law_exp_3d - 1, power=2)


            #print("V_ex ", V_ex)
            #print("rho ", rho)
            p_30 = rho / V_ex
            #print("p_30 ", p_30)

            #print("v_ex ", v_ex)
            #print("R2 ", R2)

            p_30 = rho / (v_ex*R2)
            #print("final P_30 ", p_30)

            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize(pow_law_exp_3d - 1, fr_size_range, p_30)
                           )

            pop.set_sample_range(pow_law_sample_range)

        else:
            #print("rho is False")
            exit()
            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize.from_mean_area(pow_law_exp_3d - 1, fr_size_range, p_32, pow_law_exp_3d)
                           )

            if pow_law_sample_range:
                pop.set_sample_range(pow_law_sample_range)
            elif n_frac_limit:
                # pop.set_sample_range([None, np.min([lx, ly, self.config_dict["geometry"]["fr_max_size"]])],
                #                      sample_size=n_frac_limit)
                # pop.set_sample_range([None, max(lx, ly)],
                #                      sample_size=n_frac_limit)
                pop.set_sample_range(fr_size_range,
                                     sample_size=n_frac_limit)

        #print("total mean size: ", pop.mean_size())
        #print("size range:", pop.families[0].size.sample_range)

        pos_gen = fracture.UniformBoxPosition(fracture_box)
        fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)

        # #print("fractures len ", len(fractures))
        # print("fractures ", fractures)
        # print("fr size range ", fr_size_range)

        fr_set = fracture.Fractures(fractures, fr_size_range[0] / 2)

        #print("fr set ", fr_set)
        return fr_set

    def make_summary(self, done_list):
        results = {problem.basename: problem.summary() for problem in done_list}
        #print("results ", results)

        with open("summary.yaml", "w") as f:
            yaml.dump(results, f)


    def calculate(self):
        fractures = self.generate_fractures()

        # fine problem
        fine_flow = FlowProblem.make_fine(self.i_level, (self.h_fine_step, self.config_dict["geometry"]["fr_max_size"]), fractures, self.finer_level_path,
                                          self.config_dict)
        mesh_file = "/home/"
        fine_flow.make_mesh()
        fine_flow.make_fields()
        fine_flow.run()
        done = []
        # coarse problem
        # if self.do_coarse:
        #     coarse_ref = FlowProblem.make_microscale(self.i_level, (self.h_fine_step, self.h_coarse_step), fractures, fine_flow, self.config_dict)
        #     coarse_flow = FlowProblem.make_coarse(self.i_level, (self.h_coarse_step, np.inf), fractures, coarse_ref, self.config_dict)
        #
        #     # coarse mesh
        #     coarse_flow.make_fracture_network()
        #     coarse_flow.make_mesh()
        #
        #     # microscale mesh and run
        #     #coarse_ref.make_fracture_network()
        #     coarse_ref.elementwise_mesh(coarse_flow.mesh, self.h_fine_step,  coarse_flow.outer_polygon)
        #     coarse_ref.make_fields()
        #     coarse_ref.run().join()
        #     done.append(coarse_ref)
        #
        #     # coarse fields and run
        #     coarse_flow.make_fields()
        #     coarse_flow.run()
        #     coarse_flow.thread.join()
        #     done.append(coarse_flow)
        #     coarse_flow.effective_tensor_from_bulk()
        fine_flow.thread.join()
        done.append(fine_flow)
        fine_flow.effective_tensor_from_bulk()
        self.make_summary(done)

    # def mean_tensor(self, balance_dict, normals, regions):
    #     for n, reg in zip(normals, regions):
    #         balance_dict


def finished(start_time):
    sample_time = time.time() - start_time
    time.sleep(1)
    with open("FINISHED", "w") as f:
        f.write(f"done\n{sample_time}")


def compute_semivariance(cond_field_xy, scalar_cond_log, dir):
    from scipy.spatial import distance

    semivariance = {}

    i = 0
    for xy_i, cond_i in zip(cond_field_xy, scalar_cond_log):
        #print("xy: {}, cond:{} ".format(xy_i[0], cond_i))
        i+= 1

        for xy_j, cond_j in zip(cond_field_xy[i:], scalar_cond_log[i:]):
            #if xy_i[0][0] >= xy_j[0][0] and xy_i[0][1] >= xy_j[0][1]:
            dist = distance.euclidean(xy_i[0], xy_j[0])
            if dist < 800:
                #print("dist: {}, xy_i: {} xy_j: {}, cond_i:{} cond_j:{} ".format(dist, xy_i[0], xy_j[0], cond_i, cond_j))
                semivariance.setdefault(int(dist), []).append([cond_i, cond_j])


    #exit()


    ##print("semivariance.keays() " ,semivariance.keys())

    #print(list(semivariance.keys()))
    distances = list(semivariance.keys())
    distances = [np.min(list(semivariance.keys()))]

    dst = list(semivariance.keys())
    dst.sort()
    for key in dst:
        #print("semivariance[100] ", semivariance[key])
    
        #data = semivariance[100][:int(len(semivariance[100]) / 2)]
        data = semivariance[key]

        data_array = np.array(data)
        #print("data_array.shape", data_array.shape)
        x_values = data_array[:, 0]
        y_values = data_array[:, 1]

        # x_values = np.expand_dims(x_values, axis=1)
        # y_values = np.expand_dims(y_values, axis=1)

        # #print("x_values ", x_values)
        # #print("y_values ", y_values)
        #
        # #print("len(x_values)", len(x_values))
        #exit()


        #corr_coef = np.corrcoef(semivariance[100][:int(len(semivariance[100])/2)])
        #corr_coef = np.corrcoef(data_array)
        import pandas as pd
        import seaborn as sn
        import matplotlib.pyplot as plt
        from scipy.stats.stats import pearsonr

        data = {'A': x_values,
                'B': y_values}

        df = pd.DataFrame(data)

        corr_matrix = df.corr()
        ax = sn.heatmap(corr_matrix, annot=True)
        #ax = sns.heatmap(glue, annot=True)
        ax.set(xlabel=dir + " " + str(key))
        ax.xaxis.tick_top()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(x_values, y_values)
        plt.show()

        #print("key: {}, pearsonr: {}".format(key, pearsonr(x_values, y_values)))

        # corr_matrix = df.corr("kendall")
        # sn.heatmap(corr_matrix, annot=True)
        # plt.show()
        # corr_matrix = df.corr("spearman")
        # sn.heatmap(corr_matrix, annot=True)
        # plt.show()
        # corr_coef = np.corrcoef([x_values, y_values])
        # plt.matshow(corr_coef)
        # plt.show()



def compute_variogram(cond_field_xy, scalar_cond_log, maxlag=800, dir=None):
    import matplotlib.pyplot as plt
    import itertools
    import skgstat as skg

    #print("compute variogram dir ", dir)

    compute_semivariance(cond_field_xy, scalar_cond_log, dir)

    # Select pairs to sample various point distances
    #radius = 0.5 * np.linalg.norm(points.max_pt - points.min_pt, 2)
    n_samples = len(cond_field_xy)
    assert len(scalar_cond_log) == n_samples

    bin_func = "uniform"
    #bin_func = "even"

    #all_combinations = list(itertools.combinations(cond_field_xy, 2))

    cond_field_xy_arr = np.squeeze(np.array(cond_field_xy))
    scalar_cond_log_arr = np.squeeze(np.array(scalar_cond_log))

    #print(cond_field_xy_arr.shape)
    #print("cond field xy arr ", cond_field_xy_arr)

    #print(scalar_cond_log_arr.shape)
    #print("scalar cond log arr", scalar_cond_log_arr)

    ##print("cond_field_values_arr[:,0] ", cond_field_values_arr[:, 0,0].shape)

    #V1 = skg.Variogram(cond_field_xy_arr, scalar_cond_log_arr, use_nugget = True, n_lags=100)
    #V1 = skg.Variogram(cond_field_xy_arr, scalar_cond_log_arr, n_lags=50, maxlag=0.8)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    V1 = skg.Variogram(cond_field_xy_arr, scalar_cond_log_arr, maxlag=maxlag, bin_func=bin_func)

    #print("V1.values ", V1.values)
    ##print("V1.value_matrix ", V1.value_matrix())


    V1.distance_difference_plot()

    #print("V1.distance ", V1.distance)
    #print("len(V1.distance)", len(V1.distance))
    #print("V1.values ", len(V1.values))

    #print("V1 bins ", V1._bins)
    #print("V1 cov ", V1.cov)
    #print("V1 cof ", V1.cof)

    #v1_figure = V1.plot(axes=ax, hist=True)
    # #print("v1 figure ", v1_figure)
    # #print("v1 type ", type(v1_figure))

    # v1_axes = v1_figure.get_axes()
    #
    # print("v1_axes ", v1_axes)
    #
    # line_axes = v1_axes[0]
    # print("type line ", type(line_axes))
    # print("line dict ", line_axes.__dict__)
    # print("xaxis ", line_axes.xaxis)
    #
    # print("line_axes.patches ", line_axes.patch)
    #
    # for rect in line_axes.patches:
    #     ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
    #     print("x0: {}, y0:{}".format(x0, y0))
    #     print("x1: {}, y1:{}".format(x1, y1))
    #
    # line_axes_figure = line_axes.figure
    # print('line axes figures dict ', line_axes_figure.__dict__)
    # print("x ", line_axes_figure._align_label_groups)
    # exit()
    # fig = line_axes_figure.figure
    #
    # print("type fig ", type(fig))
    # print("fig dict ", fig.__dict__)
    # fig_axes = fig.get_axes()
    #
    # axes_subplot = fig_axes[0]
    #
    # print("axes_subplot ", type(axes_subplot))
    #
    # v1_figure.show()


    #fig.savefig(os.path.join(dir, "variogram.pdf"))
    #V1.scattergram()
    #V1.distance_difference_plot(ax=ax2)
    #fig2.savefig(os.path.join(dir, "dist_differences.pdf"))

    # print("V1 coordinates ", V1.coordinates)
    #
    # print("len(lag groups)", len(V1.lag_groups()))
    # print("lag groups", V1.lag_groups())
    # print("V1.data()", V1.data())
    # print("V1.bins", V1.bins)
    # print("V1.diff() ", V1._diff)
    # print("V.metric_space ", V1.metric_space.dist_metric)
    #
    #
    # print("mode V1 distance matrix ", stats.mode(V1.distance_matrix, axis=None))
    #
    # distances = distance.cdist(cond_field_xy_arr, cond_field_xy_arr, 'euclidean')
    # print("distances ", distances)
    # print("len(distances.flatten) ", len(distances.flatten()))
    # print("len distances ", len(distances))
    #
    # print("len V1.data ", len(V1.data()))
    #
    # for lag_class in V1.lag_classes():
    #     print("lag class ", lag_class)
    # exit()

    ## Directional variogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    Vnorth = skg.DirectionalVariogram(cond_field_xy_arr, scalar_cond_log_arr, maxlag=maxlag, azimuth=90,
                                      tolerance=90, bin_func=bin_func)

    # print("Vnorth ", Vnorth)
    # exit()



    #Vnorth.plot(axes=ax,show=True)
    #Vnorth.distance_difference_plot()


    #fig.savefig(os.path.join(dir,"variogram_vnorth.pdf"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    Veast = skg.DirectionalVariogram(cond_field_xy_arr, scalar_cond_log_arr, maxlag=maxlag, azimuth=0,
                                     tolerance=90, bin_func=bin_func)
    #Veast.plot(axes=ax, show=True)
    #Veast.distance_difference_plot()


    #fig.savefig(os.path.join(dir,"variogram_veast.pdf"))

    # V2 = skg.DirectionalVariogram(cond_field_xy_arr, scalar_cond_log_arr, azimuth=45, tolerance=30)
    # V2.plot(show=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
    #ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
    #ax.plot(V2.bins, V2.experimental, '.--g')
    ax.set_xlabel('lag [m]')
    ax.set_ylabel('semi-variance (matheron)')
    #plt.legend(loc='upper left')

    #fig.show()
    #fig.savefig(os.path.join(dir, "directional_semi_variance.pdf"))

    return V1, Vnorth, Veast


def results_for_seed(seed):

    subdirs = ["n_10_s_100_100_step_5", "n_10_s_200_200_step_5", "n_10_s_300_300_step_5"]
    #subdirs = ["n_10_s_200_200_step_5", "n_10_s_300_300_step_5"]
    #subdirs = ["n_10_s_100_100_step_5"]
    #subdirs = [""]
    results = {}

    for subdir in subdirs:
        i = 0
        dir = "seed_{}/{}".format(seed, subdir)
        dir = "seed_{}/bulk_10_3/{}".format(seed, subdir)
        dir = "seed_{}/p_32_4/{}".format(seed, subdir)
        dir = "seed_{}/aperture_10_4/{}".format(seed, subdir)
        dir = "seed_{}/aperture_10_4/new_data/{}".format(seed, subdir)

        #dir = "seed_{}/comparison/sigma_10/charon_samples/rho_5".format(seed)
        #dir = "seed_{}/aperture_10_4/test/{}".format(seed, subdir)
        #dir = "seed_{}/aperture_10_4/test_2/{}".format(seed, subdir)
        #dir = "seed_{}/aperture_10_4/test_identic_mesh/{}".format(seed, subdir)
        #dir = "test_summary"

        #dir = "no_fractures/{}".format(subdir)



        if not os.path.exists(dir):
            print("dir not exists: {}".format(dir))
            continue

        cond_field_xy = []
        scalar_cond_log = []

        #i = 6 #@TODO: remove asap

        while True:
            i += 1
            # if i not in [2]:
            #     continue
            dir_name = os.path.join(os.path.join(dir, "fine_{}".format(i)))
            #print("cwd ", os.getcwd())
            #print("dir name ", dir_name)

            if os.path.exists(dir_name):
                print("dir exists ")
                with open(os.path.join(dir_name, "summary.yaml"), "r") as f:
                    summary_dict = yaml.load(f, Loader=yaml.FullLoader)
                print("summary_dict ", summary_dict)
                print("summary_dict flux ", summary_dict["fine"]["flux"])
                print("summary_dict pressure matrix ", summary_dict["fine"]["pressure_matrix"])
                print("   ...store")
                fine_cond_tn = np.array(summary_dict['fine']['cond_tn'])
                print("find cond tn ", fine_cond_tn)
                e_val, e_vec = np.linalg.eigh(fine_cond_tn)
                print("eval ", e_val)
                #print("eval ", type(np.squeeze(e_val)[1]))
                if np.min(e_val) <= 0:
                    e_val = [FlowProblem.line_fit(np.array(summary_dict["fine"]["flux"]))]

                cond_field_xy.append(np.array(summary_dict['fine']['pos']))

                print("np.mean(fine_cond_tn[0][:][0]) ", np.mean(fine_cond_tn[0][:][0]))
                #K_eff = np.mean(fine_cond_tn[0][:, 0])/10**-6
                K_eff = np.mean(e_val)#/10**-6

                print("K_eff ", K_eff)

                print("eval to mean ", e_val)
                #scalar_cond_log.append(np.log10(np.mean(e_val)))
                scalar_cond_log.append(K_eff)

                #scalar_cond_log.append(np.mean(e_val))
                print("scalar cond log ", scalar_cond_log)

                fractures_cond, bulk_cond = loads_fields_file(os.path.join(dir_name, "fields_fine.msh"))
            else:
                print("dir not exists ", dir_name)
                break

        results[subdir] = (cond_field_xy, scalar_cond_log, fractures_cond, bulk_cond)

    return results


def loads_fields_file(file):
    #print("file ", file)

    fractures_cond = []
    bulk_cond = []
    if os.path.exists(file):
        out_mesh = gmsh_io.GmshIO()
        with open(file, "r") as f:
            out_mesh.read(f)
        c_tensor = out_mesh.element_data['conductivity_tensor']

        c_tensor_0 = c_tensor[0]
        c_tensor_0_1 = c_tensor_0[1]

        for key, value in c_tensor_0_1.items():
            if value[1] == 0:
                fractures_cond.append(value[0])
            else:
                bulk_cond.append(value[0])
    else:
        raise FileNotFoundError

    return fractures_cond, bulk_cond


def process_mult_samples(sample_dict, work_dir=None):
    seeds = sample_dict.get("seeds", [])
    if len(seeds) == 0:
        seeds = [12345]

    #seeds = [0]

    seed = [26339]

    seeds_cond_field_xy = []
    seeds_scalar_cond_log = []
    seeds_results = {}
    rho_list = np.arange(0.5, 10, step=0.25)
    #rho_list = [2.0, 10.0]
    results = {}

    data_dir = "seed_{}/comparison/sigma_1000/samples/".format(seed[0])
    #data_dir = "seed_{}/ap/sigma_1000/samples/".format(seed[0])

    if work_dir is None:
        work_dir = data_dir


    all_eigen_values = []

    for rh in rho_list:
        i = 0
        # dir = "seed_{}/comparison/fr_cond_10_4/rho_{}".format(seed[0], rho)
        # dir = "seed_{}/comparison/sigma_10/charon_samples/rho_{}".format(seed[0], rho)
        # # dir = "seed_{}/aperture_10_4/test/{}".format(seed, subdir)
        # # dir = "seed_{}/aperture_10_4/test_2/{}".format(seed, subdir)
        # # dir = "seed_{}/aperture_10_4/test_identic_mesh/{}".format(seed, subdir)
        # dir = "test_summary"


        # # dir = "no_fractures/{}".format(subdir)
        # if not os.path.exists(dir):
        #     print("dir not exists: {}".format(dir))
        #     continue

        # i = 6 #@TODO: remove asap

        fr_rho_area = False

        while True:
            i += 1
            # if i not in [2]:
            #     continue
            dir_name = os.path.join(os.path.join(data_dir, "rho_{}_fine_{}".format(rh, i)))
            # print("cwd ", os.getcwd())
            # print("dir name ", dir_name)

            if not os.path.exists(dir_name) and i > 10:
                print("dir not exists: {}".format(dir_name))
                break

            if os.path.exists(dir_name):
                if not os.path.exists(os.path.join(dir_name, "summary.yaml")):
                    continue
                # print("dir: {} exists ".format(dir_name))
                with open(os.path.join(dir_name, "summary.yaml"), "r") as f:
                    summary_dict = yaml.load(f, Loader=yaml.Loader)
                # print("summary_dict ", summary_dict)
                # print("summary_dict flux ", summary_dict["fine"]["flux"])
                # print("summary_dict pressure matrix ", summary_dict["fine"]["pressure_matrix"])
                # print("   ...store")
                # print("summary_dict[fine] keys ", summary_dict["fine"].keys())
                # print("fracture lines length ", summary_dict["fine"]["fracture_length"])
                fine_cond_tn = np.array(summary_dict['fine']['cond_tn'])
                fracture_cs = summary_dict["fine"]["fracture_cs"]
                fracture_len = summary_dict["fine"]["fracture_len"]
                fracture_area = np.sum(np.multiply(fracture_cs, fracture_len))
                # print("find cond tn ", fine_cond_tn)
                # print("fracture_cs ", fracture_cs[:10])
                # print("fracture_len ", fracture_len[:10])
                # exit()
                e_val, e_vec = np.linalg.eigh(fine_cond_tn)

                # print("eval ", type(np.squeeze(e_val)[1]))
                if np.min(e_val) <= 0:
                    e_val = [FlowProblem.line_fit(np.array(summary_dict["fine"]["flux"]))]

                print("eval ", np.squeeze(e_val))

                all_eigen_values.extend(list(np.squeeze(e_val)))
                print("all eigen values ", all_eigen_values)

                # cond_field_xy.append(np.array(summary_dict['fine']['pos']))

                # print("np.mean(fine_cond_tn[0][:][0]) ", np.mean(fine_cond_tn[0][:][0]))
                # K_eff = np.mean(fine_cond_tn[0][:, 0])/10**-6
                K_eff = np.mean(e_val) #/ (10 ** -6)

                # print("K_eff ", K_eff)
                domain_box = sample_dict["geometry"]["domain_box"]
                area = domain_box[0] * domain_box[1]

                # print("area ", area)
                #
                # print("fracture len ", len(fracture_len))

                A_ex = np.mean(fracture_len) ** 2

                # pow_law_exp_3d = 3
                # A_ex = BothSample.excluded_area(np.min(fracture_len), np.max(fracture_len),
                #                                 kappa=pow_law_exp_3d - 1, coef=np.pi / 2)

                # print("A_ex ", A_ex)
                #
                # print("(len(fracture_len) / area) ", (len(fracture_len) / area))
                # print(" np.mean(fracture_len) ** 2 ",  np.mean(fracture_len) ** 2)


                rho = (len(fracture_len) / area) * A_ex#
                #
                # print("rho ", rho)
                # exit()

                #if fr_rho_area == False:
                    # print("calculated rho ", rho)
                fracture_area_ratio = fracture_area / area
                print("fracture are ratio ", fracture_area_ratio)

                fr_rho_area = True
                #fracture_area_ratio = 1

                results.setdefault(rho, []).append((K_eff - 1, fracture_area_ratio))

                # print("eval to mean ", e_val)
                # scalar_cond_log.append(np.log10(np.mean(e_val)))
                #
                # # scalar_cond_log.append(np.mean(e_val))
                # print("scalar cond log ", scalar_cond_log)
                #
                # fractures_cond, bulk_cond = loads_fields_file(os.path.join(dir_name, "fields_fine.msh"))
            else:
                print("dir not exists ", dir_name)
                break

    plot_hist_eval(all_eigen_values)

    plot_results(results, work_dir)

def plot_results(results, work_dir):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax2 = axes.twinx()
    print("plot results ", results.items())

    color_keff = 'tab:red'
    color_fr_area = 'tab:blue'


    for rho_k_eff_fr_area_ratio  in results.items():
        rho = rho_k_eff_fr_area_ratio[0]
        k_eff_fr_area_ratio = np.array(rho_k_eff_fr_area_ratio[1])
        k_eff = k_eff_fr_area_ratio[:, 0]
        fr_area_ratio = k_eff_fr_area_ratio[:, 1]

        k_eff = list(k_eff)
        fr_area_ratio = list(fr_area_ratio)
        axes.scatter(rho * np.ones(len(k_eff)), k_eff, color=color_keff, alpha=0.4)
        ax2.scatter(rho * np.ones(len(k_eff)), fr_area_ratio, marker='x', color=color_fr_area, alpha=0.4)

    axes.tick_params(axis='y', labelcolor=color_keff)
    axes.set_yscale('log')
    axes.set_ylabel(r"$K^{\prime}_{eff} - 1$", color=color_keff)
    axes.set_xlabel(r"$\rho^{\prime}_{2D}$")
    ax2.set_ylabel(r"fr area/total area (x)", color=color_fr_area)
    ax2.tick_params(axis='y', labelcolor=color_fr_area)

    fig.savefig(os.path.join(work_dir, "conductivity_sigma_10_test"))
    fig.show()


def process_results(sample_dict, work_dir=None):
    seeds = sample_dict.get("seeds", [])
    if len(seeds) == 0:
        seeds = [12345]

    #seeds = [0]
    print("seeds ", seeds)

    seeds_cond_field_xy = []
    seeds_scalar_cond_log = []
    seeds_results = {}
    for seed in seeds:
        subdir_results = results_for_seed(seed)
        print("subdir results ", subdir_results)
        # seeds_cond_field_xy.append(cond_field_xy)
        # seeds_scalar_cond_log.append(scalar_cond_log)
        seeds_results[seed] = subdir_results

    analyze_results(seeds_results)

    # if len(seeds_cond_field_xy) > 0:  # List of samples, every sample have conductivity tensor for every coarse mesh element.
    #     for idx, (s_cond_field_xy, s_scalar_cond_log) in enumerate(zip(seeds_cond_field_xy, seeds_scalar_cond_log)):
    #         # self.precompute()
    #         # self.test_homogenity()
    #         # self.test_homogenity_xy()
    #         # self.test_isotropy()
    #         # self.test_isotropy_alt()
    #         print("s cond field xy ", s_cond_field_xy)
    #         print("s scalar cond log ", s_scalar_cond_log)
    #
    #         dir="seed_{}/{}".format(seeds[idx], subdir)
    #         print("dir variogram ", dir)
    #         compute_variogram(s_cond_field_xy, s_scalar_cond_log, dir=dir)
    #
    #         # self.eigenvals_correlation()
    #         # self.calculate_field_parameters()

    #compute_anova(seeds_scalar_cond_log)

    # print("seeds cond field xy ", seeds_cond_field_xy)
    # print("seeds scalar cond log ", seeds_scalar_cond_log)


def analyze_results(seeds_results):
    hist_data = []
    V1_variograms = []
    Vnorth_variograms = []
    Veast_variograms = []
    seed_hist_data = {}
    for seed, subdir_results in seeds_results.items():
        for subdir, (cond_field_xy, scalar_cond_log, fractures_cond, bulk_cond) in subdir_results.items():
            if subdir not in seed_hist_data:
                seed_hist_data[subdir] = {"scalar_cond_log": [], "cond_field_xy": [],
                                          "fractures_cond": [], "bulk_cond": []}
                #variograms_sub_dirs[subdir] = {}

            print("subdir ", subdir)
            #print("cond field xy ", cond_field_xy)
            # print("scalar cond log ", scalar_cond_log)
            # print("fractures cond", fractures_cond)
            # print("bulk cond", bulk_cond)
            seed_hist_data[subdir]["cond_field_xy"].extend(cond_field_xy)
            seed_hist_data[subdir]["scalar_cond_log"].extend(scalar_cond_log)
            seed_hist_data[subdir]["fractures_cond"].extend(fractures_cond)
            seed_hist_data[subdir]["bulk_cond"].extend(bulk_cond)

    seed = "all_seeds"
    for subdir, data in seed_hist_data.items():
        #V1, Vnorth, Veast = compute_variogram(seed_hist_data[subdir]["cond_field_xy"], seed_hist_data[subdir]["scalar_cond_log"], dir=subdir)

        # print("subdir ", subdir)
        # exit()


        subdir_split = subdir.split("_")
        label = "S: {} x {}".format(subdir_split[3], subdir_split[3])

    #     V1_variograms.append((V1, "{}".format(label), subdir))
    #     Vnorth_variograms.append((Vnorth, "{}".format(label), subdir))
    #     Veast_variograms.append((Veast, "{}".format(label), subdir))
    #
    # matplotlib_variogram_plot(V1_variograms, hist=False, show=True, title="{}_{}".format(seed, "diff_squares"), min_dist=50)
    # matplotlib_variogram_plot(Vnorth_variograms, hist=False, show=True,
    #                           title="{}_{}".format(seed, "north_diff_squares"), min_dist=50)
    # matplotlib_variogram_plot(Veast_variograms, hist=False, show=True,
    #                           title="{}_{}".format(seed, "east_diff_squares"), min_dist=50)

    # V1_variograms.append((V1, "{}".format(subdir), subdir))
    # Vnorth_variograms.append((Vnorth, "{}".format(subdir), subdir))
    # Veast_variograms.append((Veast, "{}".format(subdir), subdir))

    plot_histogram(seed, seed_hist_data)

    plot_hist_eval(all_eigen_values)

    #matplotlib_variogram_plot([(V1, "variogram", subdir), (Vnorth, "variogram north", subdir), (Veast, "variogram east", subdir)], hist=False, show=True, title="{}_{}_{}".format(seed, "V_north_east", subdir))

    #     hist_data.append(seed_hist_data)
    #
    #     # print("subdir variograms ", subdir_variograms)
    #     # variograms = np.array(V1_Vnorth_Veast)
    #
    #     #print("variograms shape ", variograms.shape)
    #     #print("V1 variograms ", V1_variograms)
    #     #V1_variograms = []
    # matplotlib_variogram_plot(V1_variograms, hist=False, show=True, title="{}_{}".format(seed, "diff_squares"))
    # matplotlib_variogram_plot(Vnorth_variograms, hist=False, show=True, title="{}_{}".format(seed, "north_diff_squares"))
    # matplotlib_variogram_plot(Veast_variograms, hist=False, show=True,
    #                           title="{}_{}".format(seed, "east_diff_squares"))
#
    # plot_histogram(hist_data)



# Process results for each seed
# def analyze_results(seeds_results):
#     hist_data = []
#     for seed, subdir_results in seeds_results.items():
#         V1_variograms = []
#         Vnorth_variograms = []
#         Veast_variograms = []
#
#         print("seed ", seed)
#
#         seed_hist_data = {}
#         for subdir, (cond_field_xy, scalar_cond_log, fractures_cond, bulk_cond) in subdir_results.items():
#             if subdir not in seed_hist_data:
#                 seed_hist_data[subdir] = {"scalar_cond_log": [], "fractures_cond": [], "bulk_cond": []}
#
#             print("subdir ", subdir)
#             #print("cond field xy ", cond_field_xy)
#             # print("scalar cond log ", scalar_cond_log)
#             # print("fractures cond", fractures_cond)
#             # print("bulk cond", bulk_cond)
#
#             seed_hist_data[subdir]["scalar_cond_log"].extend(scalar_cond_log)
#             seed_hist_data[subdir]["fractures_cond"].extend(fractures_cond)
#             seed_hist_data[subdir]["bulk_cond"].extend(bulk_cond)
#
#             V1, Vnorth, Veast = compute_variogram(cond_field_xy, scalar_cond_log, dir=subdir)
#             V1_variograms.append((V1, "{}".format(subdir), subdir))
#             Vnorth_variograms.append((Vnorth, "{}".format(subdir), subdir))
#             Veast_variograms.append((Veast, "{}".format(subdir), subdir))
#
#         plot_histogram(seed, seed_hist_data)
#
#             #matplotlib_variogram_plot([(V1, "variogram", subdir), (Vnorth, "variogram north", subdir), (Veast, "variogram east", subdir)], hist=False, show=True, title="{}_{}_{}".format(seed, "V_north_east", subdir))
#
#     #     hist_data.append(seed_hist_data)
#     #
#     #     # print("subdir variograms ", subdir_variograms)
#     #     # variograms = np.array(V1_Vnorth_Veast)
#     #
#     #     #print("variograms shape ", variograms.shape)
#     #     #print("V1 variograms ", V1_variograms)
#     #     #V1_variograms = []
#     #     matplotlib_variogram_plot(V1_variograms, hist=False, show=True, title="{}_{}".format(seed, "diff_squares"))
#     #     matplotlib_variogram_plot(Vnorth_variograms, hist=False, show=True, title="{}_{}".format(seed, "north_diff_squares"))
#     #     matplotlib_variogram_plot(Veast_variograms, hist=False, show=True,
#     #                               title="{}_{}".format(seed, "east_diff_squares"))
#     #
#     # plot_histogram(hist_data)


def plot_hist_eval(all_eigen_values):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axes#[0]
    # ax.set_xscale('log')

    print("all eigen values ", all_eigen_values)
    ax.hist(all_eigen_values, label=r"$K_{eq}$", color="blue", bins=25)
    # ax.hist(hist_data["fractures_cond"], label="fractures cond")
    # ax.hist(hist_data["bulk_cond"], label="bulk cond")
    #ax.set_yscale('log')
    # fig.legend()
    # fig.show()
    # fig.savefig("{}_{}_histogram.pdf".format(seed, square_size))
    # plt.tight_layout()

    #ax.set_yscale('log')
    fig.legend()
    fig.show()
    fig.savefig("eval_hist.pdf")
    # plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axes  # [0]
    ax.set_xscale('log')
    ax.hist(all_eigen_values, label=r"$K_{eq}$", color="blue", bins=25, density=True)
    # ax.hist(hist_data["fractures_cond"], label="fractures cond")
    # ax.hist(hist_data["bulk_cond"], label="bulk cond")
    # ax.set_yscale('log')
    # fig.legend()
    # fig.show()
    # fig.savefig("{}_{}_histogram.pdf".format(seed, square_size))
    # plt.tight_layout()

    # ax.set_yscale('log')
    fig.legend()
    fig.show()
    fig.savefig("eval_hist_log.pdf")
    # plt.tight_layout()


def plot_histogram(seed, hist_data):
    print("plt histogram")

    #fig, axs = plt.subplots(1, len(hist_data) + 1, sharey=True, sharex=True)

    print("np.array(*hist_data).flatten().shape ", np.array(hist_data).flatten().shape)
    #hist_data.append(np.array(hist_data).flatten())

    # the histogram of the data
    # n, bins, patches = plt.hist(hist_data["scalar_cond_log"], 50, density=True)
    # plt.show()

    all_eigen_values = []

    for square_size, h_data in hist_data.items():
        print("scalar cond log ", h_data["scalar_cond_log"])
        print("fractures cond", h_data["fractures_cond"])
        print("bulk cond", h_data["bulk_cond"])

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        ax = axes[0]
        #ax.set_xscale('log')
        ax.hist(h_data["bulk_cond"], label="bulk cond", color="blue", density=True)
        # ax.hist(hist_data["fractures_cond"], label="fractures cond")
        # ax.hist(hist_data["bulk_cond"], label="bulk cond")
        ax.set_yscale('log')
        #fig.legend()
        #fig.show()
        #fig.savefig("{}_{}_histogram.pdf".format(seed, square_size))
        #plt.tight_layout()

        #fig, ax = plt.subplots(1, 1)
        ax = axes[1]
        ax.hist(h_data["fractures_cond"], label="fractures cond", color="green", density=True)
        # ax.hist(hist_data["fractures_cond"], label="fractures cond")
        # ax.hist(hist_data["bulk_cond"], label="bulk cond")
        ax.set_yscale('log')
        ax.set_xscale('log')
        #fig.legend()
        #fig.show()
        #fig.savefig("{}_{}_histogram_bulk_cond.pdf".format(seed, square_size))
        #plt.tight_layout()

        #fig, ax = plt.subplots(1, 1)
        ax = axes[2]
        ax.hist(h_data["scalar_cond_log"], label="effective cond", color="red", density=True)
        # ax.hist(hist_data["fractures_cond"], label="fractures cond")
        # ax.hist(hist_data["bulk_cond"], label="bulk cond")
        # ax.set_xscale('log')
        ax.set_yscale('log')
        fig.legend()
        fig.show()
        fig.savefig("{}_{}_histogram".format(seed, square_size))
        #plt.tight_layout()

        all_eigen_values.extend(h_data["scalar_cond_log"])


    # for (scalar_cond_log, fractures_cond, bulk_cond), axes in zip(hist_data, axs):
    #     fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    #     print("scalar cond log ", scalar_cond_log)
    #     print("fractures cond ", fractures_cond)
    #     print("bulk cond ", bulk_cond)
    #     axs.hist(scalar_cond_log, label="eq conductivity")
    #     axs.hist(fractures_cond, label="fractures cond")
    #     axs.hist(bulk_cond, label="bulk cond")
    #
    #     fig.show()
    # fig.savefig("histogram.pdf")
    import matplotlib
    matplotlib.rcParams.update({'font.size': 13})

    from scipy.stats import lognorm, powerlaw

    shape, loc, scale = lognorm.fit(all_eigen_values)
    shape, loc, scale = powerlaw.fit(all_eigen_values)

    x = np.logspace(-6, -3, 200)
    pdf = lognorm.pdf(x, shape, loc, scale)
    pdf = powerlaw.pdf(x, shape, loc, scale)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axes  # [0]
    # ax.set_xscale('log')
    print("len(all_eigen_values) ", len(all_eigen_values))

    # shape, location, scale = lognorm.fit(all_eigen_values)
    # mu, sigma = np.log(scale), shape

    #print("mu: {}, sigma: {}".format(mu,sigma))

    ax.hist(all_eigen_values, label=r"$K_{eq}$", color="blue", alpha=0.5, density=True)
    ax.plot(x, pdf, 'r')

    # ax.hist(hist_data["fractures_cond"], label="fractures cond")
    # ax.hist(hist_data["bulk_cond"], label="bulk cond")
    # ax.set_yscale('log')
    # fig.legend()
    # fig.show()
    # fig.savefig("{}_{}_histogram.pdf".format(seed, square_size))
    # plt.tight_layout()

    ax.set_ylabel('frequency')
    ax.set_xlabel(r"$K_{eq}$")
    #fig.legend(loc='upper right', ncol=2, borderaxespad=7)
    fig.show()
    fig.savefig("eval_hist.pdf")
    # plt.tight_layout()

    import matplotlib
    matplotlib.rcParams.update({'font.size': 13})

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axes  # [0]
    #ax.set_xscale('log')
    print("len(all_eigen_values) ", len(all_eigen_values))

    shape, loc, scale = lognorm.fit(all_eigen_values)

    pdf = lognorm.pdf(x, shape, loc, scale)



    ax.hist(np.log(all_eigen_values), label=r"$K_{eq}$", color="blue", alpha=0.5, density=True)
    ax.plot(np.log(x), np.log(pdf), 'r')
    # ax.hist(hist_data["fractures_cond"], label="fractures cond")
    # ax.hist(hist_data["bulk_cond"], label="bulk cond")
    # ax.set_yscale('log')
    # fig.legend()
    # fig.show()
    # fig.savefig("{}_{}_histogram.pdf".format(seed, square_size))
    # plt.tight_layout()

    ax.set_ylabel('frequency')
    ax.set_xlabel(r"$K_{eq}$")
    # fig.legend(loc='upper right', ncol=2, borderaxespad=7)
    fig.show()
    fig.savefig("eval_hist_log.pdf")
    # plt.tight_layout()


def compute_anova(seeds_scalar_cond_log):
    import scipy.stats as stats
    print("compute anova")
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    print("seeds_scalar_cond_log shape ", np.array(seeds_scalar_cond_log).shape)

    seeds_scalar_cond_log_arr = np.array(seeds_scalar_cond_log)
    print("seeds_scalar_cond_log_arr", seeds_scalar_cond_log_arr.shape)
    groups = []
    for i in range(seeds_scalar_cond_log_arr.shape[1]):
        groups.append(seeds_scalar_cond_log_arr[:, i])

    print("groups shape ", np.array(groups).shape)
    print(groups)

    fvalue, pvalue = stats.f_oneway(*groups) # (group 1 samples, group 2 sampels, ...)
    print(fvalue, pvalue)

    significant_level = 0.05
    if pvalue > significant_level:
        print("Cannot reject the null hypothesis that the population means are all equal")
    else:
        print("The null hypothesis is rejected - not all of population means are equal.")

    exit()


def subdomains(sample_dict, work_dir=None, command="run"):
    if command == "process":
        process_results(sample_dict, work_dir)
    elif command == "process_mult_samples":
        process_mult_samples(sample_dict, work_dir)
    elif command == "run_mult_samples":
        run_multiple_samples(work_dir, sample_dict)
    else:
        run_samples(work_dir, sample_dict)


def run_multiple_samples(work_dir, sample_dict, n_samples=10):
    print("sample dict ", sample_dict)
    print(sample_dict["seed"])

    # work_dir = "seed_{}".format(sample_dict["seed"])

    work_dir = os.path.join("/home/martin/Documents/Endorse/ms-homogenization/seed_26339/comparison/sigma_10/LMH/", "rho_1")
    work_dir = "/home/martin/Documents/Endorse/ms-homogenization/test_summary"

    #n_subdomains = sample_dict["geometry"].get("n_subdomains", 4)
    # domain_box = sample_dict["geometry"]["domain_box"]
    # subdomain_box = sample_dict["geometry"]["subdomain_box"]
    # lx, ly = domain_box

    n_samples = 25
    i = 0
    good_i = 0

    while good_i < n_samples:
        try:
            os.remove("fields_fine.msh")
        except OSError:
            pass

        try:
            os.remove("summary.yaml")
        except OSError:
            pass

        try:
            os.remove("mesh_fine.msh")
        except OSError:
            pass

        try:
            os.remove("mesh_fine.brep")
        except OSError:
            pass

        try:
            os.remove("mesh_fine.tmp.geo")
        except OSError:
            pass

        try:
            os.remove("mesh_fine.tmp.geo")
        except OSError:
            pass

        #try:

        sample_dict["seed"] = i
        sample_dict["geometry"]["rho_2D"] = False#1
        bs = BothSample(sample_dict)
        bs.calculate()
        good_i += 1
        print("calculate ")

        # except:
        #     pass
        i += 1

        dir_name = os.path.join(work_dir, "fine_{}".format(i))
        sample_dict["dir_name"] = dir_name
        print("dir name ", dir_name)

        #try:
        shutil.move("fine", dir_name)
        if os.path.exists(os.path.join(dir_name, "fields_fine.msh")):
            os.remove(os.path.join(dir_name, "fields_fine.msh"))
        shutil.move("fields_fine.msh", dir_name)
        shutil.move("summary.yaml", dir_name)
        shutil.move("mesh_fine.msh", dir_name)
        shutil.move("mesh_fine.brep", dir_name)
        shutil.move("mesh_fine.tmp.geo", dir_name)
        shutil.move("mesh_fine.tmp.msh", dir_name)
        if os.path.exists("fine"):
            shutil.rmtree("fine")
        # except:
        #     pass

        exit()


def run_samples(work_dir, sample_dict):
    domain_box = sample_dict["geometry"]["domain_box"]
    subdomain_box = sample_dict["geometry"]["subdomain_box"]
    subdomain_overlap = np.array([0, 0])  # np.array([50, 50])

    print("sample dict ", sample_dict)
    print(sample_dict["seed"])

    #work_dir = "seed_{}".format(sample_dict["seed"])

    work_dir = os.path.join("/home/martin/Documents/Endorse/ms-homogenization/seed_26339/aperture_10_4/test_density", "n_10_s_100_100_step_5_2")

    work_dir = "/home/martin/Documents/Endorse/ms-homogenization/test_summary"

    print("work dir ", work_dir)
    lx, ly = domain_box

    bottom_left_corner = [-lx / 2, -ly / 2]
    bottom_right_corner = [+lx / 2, -ly / 2]
    top_right_corner = [+lx / 2, +ly / 2]
    top_left_corner = [-lx / 2, +ly / 2]
    #               bottom-left corner  bottom-right corner  top-right corner    top-left corner
    complete_polygon = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]

    plt.scatter(*zip(*complete_polygon))

    n_subdomains = sample_dict["geometry"].get("n_subdomains", 4)

    domain_box = sample_dict["geometry"]["domain_box"]
    subdomain_box = sample_dict["geometry"]["subdomain_box"]
    lx, ly = domain_box

    k = 0
    for i in range(n_subdomains):
        center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2
        for j in range(n_subdomains):
            k += 1
            print("i: {}, j:{}, k:{}".format(i, j, k))
            # if k != 24:pop
            #     continue
            # if k != 1:
            #     continue
            # if k not in [1]:
            #     continue
            # if k < 88:
            #     continue
            center_y = subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_subdomains - 1) * j - lx / 2

            bl_corner = [center_x - subdomain_box[0] / 2, center_y - subdomain_box[1] / 2]
            br_corner = [center_x + subdomain_box[0] / 2, center_y - subdomain_box[1] / 2]
            tl_corner = [center_x - subdomain_box[0] / 2, center_y + subdomain_box[1] / 2]
            tr_corner = [center_x + subdomain_box[0] / 2, center_y + subdomain_box[1] / 2]

            outer_polygon = [copy.deepcopy(bl_corner), copy.deepcopy(br_corner), copy.deepcopy(tr_corner),
                             copy.deepcopy(tl_corner)]
            #print("outer polygon ", outer_polygon)

            plt.scatter(*zip(*outer_polygon))

            sample_dict["geometry"]["outer_polygon"] = outer_polygon

            sample_dict["work_dir"] = work_dir

            bs = BothSample(sample_dict)
            bs.calculate()

            dir_name = os.path.join(work_dir, "fine_{}".format(k))
            sample_dict["dir_name"] = dir_name

            print("dir name ", dir_name)

            try:
                shutil.move("fine", dir_name)
                if os.path.exists(os.path.join(dir_name, "fields_fine.msh")):
                    os.remove(os.path.join(dir_name, "fields_fine.msh"))
                shutil.move("fields_fine.msh", dir_name)
                shutil.move("summary.yaml", dir_name)
                shutil.move("mesh_fine.msh", dir_name)
                shutil.move("mesh_fine.brep", dir_name)
                shutil.move("mesh_fine.tmp.geo", dir_name)
                shutil.move("mesh_fine.tmp.msh", dir_name)
                shutil.rmtree("fine")
            except:
                pass

    #plt.show()

    # bottom_left_corner = [-lx / 2, -ly / 2]
    # bottom_right_corner = [+lx / 2, -ly / 2]
    # top_right_corner = [+lx / 2, +ly / 2]
    # top_left_corner = [-lx / 2, +ly / 2]
    # #               bottom-left corner  bottom-right corner  top-right corner    top-left corner
    # complete_polygon = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]
    # # bs = BothSample(sample_dict)
    # # bs.calculate()
    #
    # plt.scatter(*zip(*complete_polygon))
    #
    # tl_corner = top_left_corner
    #
    # i = 0
    # while True:
    #     bl_corner = tl_corner - np.array([0, subdomain_box[1]])
    #     bl_corner = np.maximum(bl_corner,
    #                            [bl_corner[0], bottom_right_corner[1]])  # make sure it lies in the complete polygon
    #
    #     br_corner = bl_corner + np.array([subdomain_box[0], 0])
    #     br_corner = np.minimum(br_corner, [bottom_right_corner[0], br_corner[1]])
    #
    #     tr_corner = tl_corner + np.array([subdomain_box[0], 0])
    #     tr_corner = np.minimum(tr_corner, [bottom_right_corner[0], tr_corner[1]])
    #
    #     outer_polygon = [copy.deepcopy(bl_corner), copy.deepcopy(br_corner), copy.deepcopy(tr_corner),
    #                      copy.deepcopy(tl_corner)]
    #     print("outer polygon ", outer_polygon)
    #
    #     sample_dict["geometry"]["outer_polygon"] = outer_polygon
    #
    #     # plt.scatter(*zip(*outer_polygon))
    #
    #     # Set new top-left corner including overlap on x axes
    #     # Do not move corner if at the end of x axis
    #     if tr_corner[0] - tl_corner[0] == subdomain_box[0]:
    #         tl_corner = tr_corner - np.array([subdomain_overlap[0], 0])
    #     else:
    #         tl_corner = tr_corner
    #
    #     # top-left corner is on the left-hand edge
    #     if tl_corner[0] >= bottom_right_corner[0]:
    #         tl_corner[0] = bottom_left_corner[0]
    #         tl_corner[1] = tl_corner[1] - subdomain_box[1]
    #         # y-axis overlap
    #         tl_corner += np.array([0, subdomain_overlap[1] if tl_corner[1] < top_left_corner[1] else 0])
    #
    #     # bs = BothSample(sample_dict)
    #     # bs.calculate()
    #
    #     dir_name = "fine_{}".format(i)
    #     sample_dict["dir_name"] = dir_name
    #     try:
    #         os.remove("mesh_fine.msh")
    #         os.remove("mesh_fine.brep")
    #         os.remove("mesh_fine.tmp.geo")
    #         os.remove("mesh_fine.tmp.msh")
    #         import shutil
    #         shutil.move("fine", dir_name)
    #         shutil.copy("summary.yaml", dir_name)
    #         shutil.rmtree("fine")
    #     except:
    #         pass
    #
    #
    #     # top-left corner is on the bottom edge
    #     if tl_corner[1] <= bottom_left_corner[1]:
    #         break
    #
    #     i += 1
    #
    # plt.show()

if __name__ == "__main__":
    start_time = time.time()
    atexit.register(finished, start_time)
    work_dir = sys.argv[1]
    sample_config = sys.argv[2]
    #try:
    with open(sample_config, "r") as f:
        sample_dict = yaml.load(f, Loader=yaml.FullLoader)
    #except Exception as e:
    #    print("cwd: ", os.getcwd(), "sample config: ", sample_config)

    #subdomains(sample_dict, work_dir, command="run")
    #subdomains(sample_dict, work_dir, command="run_mult_samples")
    subdomains(sample_dict, command="process")
    #subdomains(sample_dict, command="process_mult_samples")
