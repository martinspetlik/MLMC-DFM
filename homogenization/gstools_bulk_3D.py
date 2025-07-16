from typing import *
from abc import *

import warnings
import logging

import joblib
import os
import sys
import glob
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
from sklearn.preprocessing import QuantileTransformer
#import seaborn as sns
from bgem.stochastic.fr_set import FractureSet


from typing import *
from abc import *

import warnings
import logging

import joblib
import os
import sys
import glob
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
from sklearn.preprocessing import QuantileTransformer
#import seaborn as sns
from bgem.stochastic.fr_set import FractureSet


# @attr.s(auto_attribs=True)
# class GSToolsBulk3DVectorField:
#     mean_log_conductivity: Tuple[float, float]
#     cov_log_conductivity: Optional[List[List[float]]]
#     angle_mean: float = attr.ib(converter=float)
#     angle_concentration: float = attr.ib(converter=float)
#     corr_lengths_x: Optional[List[float]]  # [15, 0, 0]
#     corr_lengths_y: Optional[List[float]]  # = [3, 0, 0]
#     corr_lengths_z: Optional[List[float]]  # = [3, 0, 0]
#     anis_angles: Optional[List[float]]  # = [0, 0, 0]
#     anisotropy: float = None
#     rotation: float = None
#     mode_no: int = 1000
#     angle_var = 1
#     _rf_sample = None
#     _mesh_data= None
#     log = False
#     seed= None
#     mode: Optional[str] = None
#
#     def _pca(self, mean_k_xx_yy_zz, cov_matrix_k_xx_yy_zz):
#         n_samples = 10000
#
#         # print("mean k xx yy ", mean_k_xx_yy)
#         # print("cov matrix k xx yy ", cov_matrix_k_xx_yy)
#         # print("self. angle var ", self.angle_var)
#         np.random.seed(seed=self.seed)
#         gstools.random.RNG(seed=self.seed)
#
#         samples = np.random.multivariate_normal(mean=mean_k_xx_yy_zz, cov=cov_matrix_k_xx_yy_zz, size=n_samples)
#
#         # sample_means = [np.mean(samples[:, 0]), np.mean(samples[:, 1])]
#         # sample_vars = [np.var(samples[:, 0]), np.var(samples[:, 1])]
#         # print("sample means: {} vars: {}".format(sample_means, sample_vars))
#         covariance_matrix = np.cov(samples.T)
#         #print("covariance matrix ", covariance_matrix)
#         eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#         self._projection_matrix = (eigen_vectors.T[:][:]).T
#
#         #print("cov_matrix_k_xx_yy_zz ", cov_matrix_k_xx_yy_zz)
#
#         eigen_values, eigen_vectors = np.linalg.eig(cov_matrix_k_xx_yy_zz)
#         self._projection_matrix = (eigen_vectors.T[:][:]).T
#         p_components = samples.dot(self._projection_matrix)
#
#         #print("p components shape ", p_components.shape)
#
#         pc_means = [np.mean(p_components[:, 0]), np.mean(p_components[:, 1]), np.mean(p_components[:, 2])]
#         pc_vars = [np.var(p_components[:, 0]), np.var(p_components[:, 1]), np.var(p_components[:, 2])]
#         #print("pc means: {} vars: {}".format(pc_means, pc_vars))
#
#
#         if self.corr_lengths_x[0] == 0:
#             len_scale = 1e-15
#         else:
#             len_scale = [self.corr_lengths_x[0], self.corr_lengths_y[0], self.corr_lengths_z[0]]
#         self._model_k_xx_yy_zz = gstools.Exponential(dim=3, var=pc_vars[0], len_scale=len_scale, angles=self.anis_angles[0])
#
#         # ##########
#         # ## K_xx ##
#         # ##########
#         # if self.corr_lengths_x[0] == 0:
#         #     len_scale = 1e-15
#         # else:
#         #     len_scale = [self.corr_lengths_x[0], self.corr_lengths_y[0], self.corr_lengths_z[0]]
#         #
#         # self._model_k_xx = gstools.Exponential(dim=3, var=pc_vars[0], len_scale=len_scale, angles=self.anis_angles[0])
#         # # self._model_k_xx_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale, angles=self.anis_angles[0])
#         #
#         # ##########
#         # ## K_yy ##
#         # ##########
#         # if self.corr_lengths_x[1] == 0:
#         #     len_scale = 1e-15
#         # else:
#         #     len_scale = [self.corr_lengths_x[1], self.corr_lengths_y[1], self.corr_lengths_z[1]]
#         #
#         # #print("len scale ", len_scale)
#         #
#         # self._model_k_yy = gstools.Exponential(dim=3, var=pc_vars[1], len_scale=len_scale, angles=self.anis_angles[1])
#         # # self._model_k_yy_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
#         # #                                            angles=self.anis_angles[1])
#         #
#         # ##########
#         # ## K_zz ##
#         # ##########
#         # if self.corr_lengths_x[2] == 0:
#         #     len_scale = 1e-15
#         # else:
#         #     len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2], self.corr_lengths_z[2]]
#         # # print("len scale ", len_scale)
#         #
#         # self._model_k_zz = gstools.Exponential(dim=3, var=pc_vars[2], len_scale=len_scale, angles=self.anis_angles[2])
#         # # self._model_k_zz_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
#         # #                                            angles=self.anis_angles[2])
#
#
#         ###########
#         ## Angle ##
#         ###########
#         if self.corr_lengths_x[2] == 0:
#             len_scale = 1e-15
#         else:
#             len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2]]
#
#         self._R_xy = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         self._N_xyz = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#
#         # #model_angle = gstools.Gaussian(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         # self._R_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         # self._R_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         #
#         # #self._R_x = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         # #self._R_y = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         #
#         # self._N_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         # self._N_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#         # self._N_z = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
#
#         # print("self._model_k_xx ", self._model_k_xx)
#         # print("self._model_k_yy ", self._model_k_yy)
#         # print("self._model_angle ", self._model_angle)
#
#         self._pc_means = pc_means
#
#     def _create_field(self, mean_log_conductivity, cov_log_conductivity):
#         self._pca(mean_log_conductivity, cov_log_conductivity)
#
#         field_k_xx_yy_zz = cf.Field('k_xx_yy_zz', cf.GSToolsSpatialCorrelatedVectorField(self._model_k_xx_yy_zz, log=self.log,
#                                                                              mean=(0,0,0),
#                                                                        #sigma=np.sqrt(self.cov_log_conductivity[0,0]),
#                                                                        mode_no=self.mode_no,
#                                                                        #mu=pca_means[0]
#                                                                        seed=self.seed + 100, mode=self.mode
#                                                                        ))
#
#         # field_k_yy = cf.Field('k_yy', cf.GSToolsSpatialCorrelatedField(self._model_k_yy, log=self.log,
#         #                                                                #sigma=np.sqrt(self.cov_log_conductivity[1,1]),
#         #                                                                mode_no=self.mode_no,
#         #                                                                #mu=pca_means[1]
#         #                                                                seed=self.seed + 200, mode=self.mode
#         #                                                                ))
#         #
#         # field_k_zz = cf.Field('k_zz', cf.GSToolsSpatialCorrelatedField(self._model_k_zz, log=self.log,
#         #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
#         #                                                                mode_no=self.mode_no,
#         #                                                                # mu=pca_means[1]
#         #                                                                seed=self.seed + 300, mode=self.mode
#         #                                                                ))
#
#         # field_k_xx_new = cf.Field('k_xx_new', cf.GSToolsSpatialCorrelatedField(self._model_k_xx_new, log=self.log,
#         #                                                                # sigma=np.sqrt(self.cov_log_conductivity[0,0]),
#         #                                                                mode_no=self.mode_no,
#         #                                                                # mu=pca_means[0]
#         #                                                                seed=self.seed + 1100
#         #                                                                ))
#         #
#         # field_k_yy_new = cf.Field('k_yy_new', cf.GSToolsSpatialCorrelatedField(self._model_k_yy_new, log=self.log,
#         #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
#         #                                                                mode_no=self.mode_no,
#         #                                                                # mu=pca_means[1]
#         #                                                                seed=self.seed + 1200
#         #                                                                ))
#         #
#         # field_k_zz_new = cf.Field('k_zz_new', cf.GSToolsSpatialCorrelatedField(self._model_k_zz_new, log=self.log,
#         #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
#         #                                                                mode_no=self.mode_no,
#         #                                                                # mu=pca_means[1]
#         #                                                                seed=self.seed + 1300
#         #                                                                ))
#
#         field_R_xy = cf.Field('R_xy', cf.GSToolsSpatialCorrelatedVectorField(self._R_xy,
#                                                                          mean=(0,0,0),
#                                                                          #sigma=(1,1,1),
#                                                                          mode_no=self.mode_no,
#                                                                          seed=self.seed + 400, mode=self.mode
#                                                                          ))
#
#         field_N_xyz = cf.Field('N_xyz', cf.GSToolsSpatialCorrelatedVectorField(self._N_xyz,
#                                                                              mean=(0, 0, 0),
#                                                                              #sigma=(1, 1, 1),
#                                                                              mode_no=self.mode_no,
#                                                                              seed=self.seed + 400, mode=self.mode
#                                                                              ))
#
#         # field_R_y = cf.Field('R_y', cf.GSToolsSpatialCorrelatedField(self._R_x,
#         #                                                                  sigma=np.sqrt(self.angle_var),
#         #                                                                  mode_no=self.mode_no,
#         #                                                                  seed=self.seed + 500, mode=self.mode
#         #                                                                  ))
#         #
#         # field_N_x = cf.Field('N_x', cf.GSToolsSpatialCorrelatedField(self._N_x,
#         #                                                              sigma=np.sqrt(self.angle_var),
#         #                                                              mode_no=self.mode_no,
#         #                                                              seed=self.seed + 600, mode=self.mode
#         #                                                              ))
#         #
#         # field_N_y = cf.Field('N_y', cf.GSToolsSpatialCorrelatedField(self._N_y,
#         #                                                              sigma=np.sqrt(self.angle_var),
#         #                                                              mode_no=self.mode_no,
#         #                                                              seed=self.seed + 700, mode=self.mode
#         #                                                              ))
#         #
#         # field_N_z = cf.Field('N_z', cf.GSToolsSpatialCorrelatedField(self._N_z,
#         #                                                              sigma=np.sqrt(self.angle_var),
#         #                                                              mode_no=self.mode_no,
#         #                                                              seed=self.seed + 800, mode=self.mode
#         #                                                              ))
#
#         self._fields = cf.Fields([field_k_xx_yy_zz, field_R_xy, field_N_xyz])
#         #print("self._fields ", self._fields)
#
#     def generate_field(self, barycenters):
#         if self._rf_sample is None:
#             #if self.model_k_xx is None:
#             self._create_field(self.mean_log_conductivity, self.cov_log_conductivity)
#
#             #grid_barycenters = fem_grid.grid.barycenters()
#
#             print("barycenters ", barycenters.shape)
#
#             #self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
#             #print("mesh data ", list(self._mesh_data.keys()))
#             # print("mesh data ele ids ", self._mesh_data["ele_ids"])
#             # print("mesh_data['points'] ", self._mesh_data['points'])
#             #self._fields.set_points(self._mesh_data['points'], self._mesh_data['point_region_ids'], self._mesh_data['region_map'])
#
#             self._fields.set_points(barycenters)
#             self._rf_sample = self._fields.sample()
#
#             k_xx = self._rf_sample["k_xx_yy_zz"][0]
#             k_yy = self._rf_sample["k_xx_yy_zz"][1]
#             k_zz = self._rf_sample["k_xx_yy_zz"][2]
#
#             # print(self._rf_sample["k_xx_yy_zz"].shape)
#             # exit()
#
#             # # print("self._pc_mean ", self._pc_means)
#             # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
#             # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
#             # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))
#
#             k_xx += self._pc_means[0]
#             k_yy += self._pc_means[1]
#             k_zz += self._pc_means[2]
#
#             # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
#             # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
#             # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))
#
#             srf_data = np.array([k_xx, k_yy, k_zz])
#
#             inv_srf_data = np.matmul(srf_data.T, self._projection_matrix.T).T
#
#             k_xx = inv_srf_data[0, :]
#             k_yy = inv_srf_data[1, :]
#             k_zz = inv_srf_data[2, :]
#
#         # k_xx = self._rf_sample["k_xx"]
#         # k_yy = self._rf_sample["k_yy"]
#         # k_zz = self._rf_sample["k_zz"]
#
#         # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
#         # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
#         # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))
#         #
#         # print("var k xx ", np.var(self._rf_sample["k_xx"]))
#         # print("var k yy ", np.var(self._rf_sample["k_yy"]))
#         # print("var k zz ", np.var(self._rf_sample["k_zz"]))
#
#         # data = np.vstack([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])
#         #
#         # # Calculate the covariance matrix
#         # cov_matrix = np.cov(data)
#         # # print("cov matrix ", cov_matrix)
#         # # exit()
#
#         # print("R_x ", self._rf_sample["R_x"])
#         #
#         plt.figure(figsize=(8, 6))
#         plt.hist(self._rf_sample["R_xy"][0], bins=50, density=True, alpha=0.6, edgecolor='black')
#         plt.title("Histogram of R_x")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.show()
#
#         print("R_y ", self._rf_sample["R_y"])
#
#         plt.figure(figsize=(8, 6))
#         plt.hist(self._rf_sample["R_xy"][1], bins=50, density=True, alpha=0.6, edgecolor='black')
#         plt.title("Histogram of R_y")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.show()
#
#
#         R_vectors = np.stack([self._rf_sample["R_xy"][0],  self._rf_sample["R_xy"][1]])
#         R_vectors = R_vectors / np.linalg.norm(R_vectors, axis=0)
#
#
#         #print("R vectors ", R_vectors)
#
#         #angle_radians = np.arctan2(R_vectors[0, :], R_vectors[1, :])
#
#         # print("angle radians ", angle_radians)
#         #
#         # plt.figure(figsize=(8, 6))
#         # plt.hist(angle_radians, bins=50, density=True, alpha=0.6, edgecolor='black')
#         # plt.title("Histogram of angle_radians")
#         # plt.xlabel("Value")
#         # plt.ylabel("Density")
#         # plt.show()
#         #
#         #
#         # print("barycenters ", grid_barycenters.shape)
#         # print("angle_radians shape ", angle_radians.shape)
#
#         # fig = plt.figure(figsize=(10, 8))
#         # ax = fig.add_subplot(111, projection='3d')
#
#         # # Scatter plot with color mapping
#         # scatter = ax.scatter(grid_barycenters[:, 0], grid_barycenters[:, 1], grid_barycenters[:, 2],
#         #                      c=angle_radians, cmap='viridis', marker='o')
#         #
#         # # Add a color bar
#         # cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
#         # cbar.set_label('Scalar Values', rotation=270, labelpad=15)
#
#         # # Set axis labels
#         # ax.set_xlabel('X')
#         # ax.set_ylabel('Y')
#         # ax.set_zlabel('Z')
#
#         # Extract X, Y, Z coordinates
#         #x, y, z = barycenters[:, 0], barycenters[:, 1], barycenters[:, 2]
#
#         # from mayavi import mlab
#         #
#         # # Create a 3D scatter plot
#         # mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # White background
#         # scatter = mlab.points3d(x, y, z, angle_radians,
#         #                         scale_mode='none',  # Fix point size
#         #                         scale_factor=0.02,  # Adjust size
#         #                         colormap='viridis')  # Choose colormap
#         #
#         # # Add color bar
#         # mlab.colorbar(scatter, title="Scalar Values", orientation="vertical")
#         #
#         # # Set axis labels
#         # mlab.xlabel("X")
#         # mlab.ylabel("Y")
#         # mlab.zlabel("Z")
#
#         N_vectors = np.stack([self._rf_sample["N_xyz"][0],  self._rf_sample["N_xyz"][1], self._rf_sample["N_xyz"][2]])
#         N_vectors = N_vectors / np.linalg.norm(N_vectors, axis=0)
#
#         #phi = np.arctan2(N_vectors[1, :], N_vectors[0, :])
#
#         #magnitude = np.sqrt(N_vectors[0, :] ** 2 + N_vectors[1, :] ** 2 + N_vectors[2, :] ** 2)
#
#         #print("magnitude ", magnitude)
#
#         # Compute the polar angle (theta) in radians
#         #theta = np.arccos(N_vectors[2,:] / magnitude)
#
#         # plt.figure(figsize=(8, 6))
#         # plt.hist(theta, bins=50, density=True, alpha=0.6, edgecolor='black')
#         # plt.title("Histogram of theta")
#         # plt.xlabel("Value")
#         # plt.ylabel("Density")
#         # plt.show()
#         #
#         # plt.figure(figsize=(8, 6))
#         # plt.hist(phi, bins=50, density=True, alpha=0.6, edgecolor='black')
#         # plt.title("Histogram of phi")
#         # plt.xlabel("Value")
#         # plt.ylabel("Density")
#         # plt.show()
#
#         N_vectors = N_vectors.T
#         R_vectors = R_vectors.T
#
#         # Stack the arrays into a 2D array (27 x 3)
#         stacked_arrays = np.power(10, np.vstack((k_xx, k_yy, k_zz)).T)
#
#         # Create diagonal matrices using broadcasting with einsum
#         diagonal_matrices = np.zeros((N_vectors.shape[0], 3, 3))
#         indices = np.arange(3)
#         diagonal_matrices[:, indices, indices] = stacked_arrays
#
#         radius = np.ones((N_vectors.shape[0], 2))
#
#         transform_matrices = FractureSet.transform_mat_static(normal=N_vectors, shape_axis=R_vectors, radius=radius)
#
#         cond_3d = transform_matrices @ diagonal_matrices @ np.transpose(transform_matrices, axes=(0, 2, 1))
#
#         n_tensors = cond_3d.shape[0]
#
#         # Step 1: Compute eigenvalues and eigenvectors for all tensors
#         eigenvalues, eigenvectors = np.linalg.eigh(cond_3d)
#
#         # eigenvalues: Shape (4096, 3)
#         # eigenvectors: Shape (4096, 3, 3) where each row contains one eigenvector set
#
#         dominant_eigenvectors = eigenvectors[np.arange(n_tensors), :, np.argmax(eigenvalues, axis=1)]
#         # Step 3: Compute azimuth and elevation
#         # Azimuth: Angle in the XY-plane
#         azimuths = np.arctan2(dominant_eigenvectors[:, 1], dominant_eigenvectors[:, 0])  # Shape (4096,)
#
#         # Elevation: Angle from the Z-axis
#         elevations = np.arccos(
#             dominant_eigenvectors[:, 2] / np.linalg.norm(dominant_eigenvectors, axis=1))  # Shape (4096,)
#
#         #print("azimuths shape ", azimuths.shape)
#
#         #single_angles = np.sqrt(azimuths_deg ** 2 + elevations_deg ** 2)
#
#         plt.figure(figsize=(8, 6))
#         plt.hist(azimuths, bins=50, density=True, alpha=0.6, edgecolor='black')
#         plt.title("Histogram of azimuths")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.show()
#
#         plt.figure(figsize=(8, 6))
#         plt.hist(elevations, bins=50, density=True, alpha=0.6, edgecolor='black')
#         plt.title("Histogram of elevations")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.show()
#         print("cond_3d.shape ", cond_3d.shape)
#         eigenvalues = np.linalg.eigvals(cond_3d)
#         assert np.all(eigenvalues > 0)
#         return cond_3d, barycenters




@attr.s(auto_attribs=True)
class GSToolsBulk3D:
    mean_log_conductivity: Tuple[float, float]
    cov_log_conductivity: Optional[List[List[float]]]
    angle_mean: float = attr.ib(converter=float)
    angle_concentration: float = attr.ib(converter=float)
    corr_lengths_x: Optional[List[float]]  # [15, 0, 0]
    corr_lengths_y: Optional[List[float]]  # = [3, 0, 0]
    corr_lengths_z: Optional[List[float]]  # = [3, 0, 0]
    anis_angles: Optional[List[float]]  # = [0, 0, 0]
    anisotropy: float = None
    rotation: float = None
    mode_no: int = 1000
    angle_var = 1
    _rf_sample = None
    _mesh_data= None
    log = False
    seed= None
    mode: Optional[str] = None

    def _pca(self, mean_k_xx_yy_zz, cov_matrix_k_xx_yy_zz):
        n_samples = 10000

        # print("mean k xx yy ", mean_k_xx_yy)
        # print("cov matrix k xx yy ", cov_matrix_k_xx_yy)
        # print("self. angle var ", self.angle_var)
        np.random.seed(seed=self.seed)
        gstools.random.RNG(seed=self.seed)

        samples = np.random.multivariate_normal(mean=mean_k_xx_yy_zz, cov=cov_matrix_k_xx_yy_zz, size=n_samples)

        # sample_means = [np.mean(samples[:, 0]), np.mean(samples[:, 1])]
        # sample_vars = [np.var(samples[:, 0]), np.var(samples[:, 1])]
        # print("sample means: {} vars: {}".format(sample_means, sample_vars))
        covariance_matrix = np.cov(samples.T)
        #print("covariance matrix ", covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        self._projection_matrix = (eigen_vectors.T[:][:]).T

        #print("cov_matrix_k_xx_yy_zz ", cov_matrix_k_xx_yy_zz)

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix_k_xx_yy_zz)
        self._projection_matrix = (eigen_vectors.T[:][:]).T
        p_components = samples.dot(self._projection_matrix)

        #print("p components shape ", p_components.shape)

        pc_means = [np.mean(p_components[:, 0]), np.mean(p_components[:, 1]), np.mean(p_components[:, 2])]
        pc_vars = [np.var(p_components[:, 0]), np.var(p_components[:, 1]), np.var(p_components[:, 2])]
        #print("pc means: {} vars: {}".format(pc_means, pc_vars))

        ##########
        ## K_xx ##
        ##########
        if self.corr_lengths_x[0] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[0], self.corr_lengths_y[0], self.corr_lengths_z[0]]

        self._model_k_xx = gstools.Exponential(dim=3, var=pc_vars[0], len_scale=len_scale, angles=self.anis_angles[0])
        # self._model_k_xx_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale, angles=self.anis_angles[0])

        ##########
        ## K_yy ##
        ##########
        if self.corr_lengths_x[1] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[1], self.corr_lengths_y[1], self.corr_lengths_z[1]]

        #print("len scale ", len_scale)

        self._model_k_yy = gstools.Exponential(dim=3, var=pc_vars[1], len_scale=len_scale, angles=self.anis_angles[1])
        # self._model_k_yy_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
        #                                            angles=self.anis_angles[1])

        ##########
        ## K_zz ##
        ##########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2], self.corr_lengths_z[2]]
        # print("len scale ", len_scale)

        self._model_k_zz = gstools.Exponential(dim=3, var=pc_vars[2], len_scale=len_scale, angles=self.anis_angles[2])
        # self._model_k_zz_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
        #                                            angles=self.anis_angles[2])


        ###########
        ## Angle ##
        ###########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2]]

        #model_angle = gstools.Gaussian(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._R_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._R_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

        #self._R_x = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        #self._R_y = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

        self._N_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._N_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        self._N_z = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

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
                                                                       seed=self.seed + 100, mode=self.mode
                                                                       ))

        field_k_yy = cf.Field('k_yy', cf.GSToolsSpatialCorrelatedField(self._model_k_yy, log=self.log,
                                                                       #sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       #mu=pca_means[1]
                                                                       seed=self.seed + 200, mode=self.mode
                                                                       ))

        field_k_zz = cf.Field('k_zz', cf.GSToolsSpatialCorrelatedField(self._model_k_zz, log=self.log,
                                                                       # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       # mu=pca_means[1]
                                                                       seed=self.seed + 300, mode=self.mode
                                                                       ))

        # field_k_xx_new = cf.Field('k_xx_new', cf.GSToolsSpatialCorrelatedField(self._model_k_xx_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[0,0]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[0]
        #                                                                seed=self.seed + 1100
        #                                                                ))
        #
        # field_k_yy_new = cf.Field('k_yy_new', cf.GSToolsSpatialCorrelatedField(self._model_k_yy_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[1]
        #                                                                seed=self.seed + 1200
        #                                                                ))
        #
        # field_k_zz_new = cf.Field('k_zz_new', cf.GSToolsSpatialCorrelatedField(self._model_k_zz_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[1]
        #                                                                seed=self.seed + 1300
        #                                                                ))

        field_R_x = cf.Field('R_x', cf.GSToolsSpatialCorrelatedField(self._R_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no,
                                                                         seed=self.seed + 400, mode=self.mode
                                                                         ))

        field_R_y = cf.Field('R_y', cf.GSToolsSpatialCorrelatedField(self._R_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no,
                                                                         seed=self.seed + 500, mode=self.mode
                                                                         ))

        field_N_x = cf.Field('N_x', cf.GSToolsSpatialCorrelatedField(self._N_x,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 600, mode=self.mode
                                                                     ))

        field_N_y = cf.Field('N_y', cf.GSToolsSpatialCorrelatedField(self._N_y,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 700, mode=self.mode
                                                                     ))

        field_N_z = cf.Field('N_z', cf.GSToolsSpatialCorrelatedField(self._N_z,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 800, mode=self.mode
                                                                     ))

        self._fields = cf.Fields([field_k_xx, field_k_yy, field_k_zz, field_R_x, field_R_y, field_N_x, field_N_y, field_N_z])
        #print("self._fields ", self._fields)

    def generate_field(self, barycenters):
        if self._rf_sample is None:
            #if self.model_k_xx is None:
            self._create_field(self.mean_log_conductivity, self.cov_log_conductivity)

            #grid_barycenters = fem_grid.grid.barycenters()

            #print("barycenters ", barycenters.shape)

            #self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            #print("mesh data ", list(self._mesh_data.keys()))
            # print("mesh data ele ids ", self._mesh_data["ele_ids"])
            # print("mesh_data['points'] ", self._mesh_data['points'])
            #self._fields.set_points(self._mesh_data['points'], self._mesh_data['point_region_ids'], self._mesh_data['region_map'])

            self._fields.set_points(barycenters)
            self._rf_sample = self._fields.sample()

            # # print("self._pc_mean ", self._pc_means)
            # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
            # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
            # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))

            self._rf_sample["k_xx"] += self._pc_means[0]
            self._rf_sample["k_yy"] += self._pc_means[1]
            self._rf_sample["k_zz"] += self._pc_means[2]

            # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
            # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
            # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))

            srf_data = np.array([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])

            inv_srf_data = np.matmul(srf_data.T, self._projection_matrix.T).T

            self._rf_sample["k_xx"] = inv_srf_data[0, :]
            self._rf_sample["k_yy"] = inv_srf_data[1, :]
            self._rf_sample["k_zz"] = inv_srf_data[2, :]

        k_xx = self._rf_sample["k_xx"]
        k_yy = self._rf_sample["k_yy"]
        k_zz = self._rf_sample["k_zz"]

        # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
        # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
        # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))
        #
        # print("var k xx ", np.var(self._rf_sample["k_xx"]))
        # print("var k yy ", np.var(self._rf_sample["k_yy"]))
        # print("var k zz ", np.var(self._rf_sample["k_zz"]))

        # data = np.vstack([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])
        #
        # # Calculate the covariance matrix
        # cov_matrix = np.cov(data)
        # # print("cov matrix ", cov_matrix)
        # # exit()

        # print("R_x ", self._rf_sample["R_x"])
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(self._rf_sample["R_x"], bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of R_x")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # print("R_y ", self._rf_sample["R_y"])
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(self._rf_sample["R_y"], bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of R_y")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()


        R_vectors = np.stack([self._rf_sample["R_x"],  self._rf_sample["R_y"]])
        R_vectors = R_vectors / np.linalg.norm(R_vectors, axis=0)


        #print("R vectors ", R_vectors)

        #angle_radians = np.arctan2(R_vectors[0, :], R_vectors[1, :])

        # print("angle radians ", angle_radians)
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(angle_radians, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of angle_radians")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        #
        # print("barycenters ", grid_barycenters.shape)
        # print("angle_radians shape ", angle_radians.shape)

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Scatter plot with color mapping
        # scatter = ax.scatter(grid_barycenters[:, 0], grid_barycenters[:, 1], grid_barycenters[:, 2],
        #                      c=angle_radians, cmap='viridis', marker='o')
        #
        # # Add a color bar
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        # cbar.set_label('Scalar Values', rotation=270, labelpad=15)

        # # Set axis labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # Extract X, Y, Z coordinates
        #x, y, z = barycenters[:, 0], barycenters[:, 1], barycenters[:, 2]

        # from mayavi import mlab
        #
        # # Create a 3D scatter plot
        # mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # White background
        # scatter = mlab.points3d(x, y, z, angle_radians,
        #                         scale_mode='none',  # Fix point size
        #                         scale_factor=0.02,  # Adjust size
        #                         colormap='viridis')  # Choose colormap
        #
        # # Add color bar
        # mlab.colorbar(scatter, title="Scalar Values", orientation="vertical")
        #
        # # Set axis labels
        # mlab.xlabel("X")
        # mlab.ylabel("Y")
        # mlab.zlabel("Z")

        N_vectors = np.stack([self._rf_sample["N_x"],  self._rf_sample["N_y"], self._rf_sample["N_z"]])
        N_vectors = N_vectors / np.linalg.norm(N_vectors, axis=0)

        #phi = np.arctan2(N_vectors[1, :], N_vectors[0, :])

        #magnitude = np.sqrt(N_vectors[0, :] ** 2 + N_vectors[1, :] ** 2 + N_vectors[2, :] ** 2)

        #print("magnitude ", magnitude)

        # Compute the polar angle (theta) in radians
        #theta = np.arccos(N_vectors[2,:] / magnitude)

        # plt.figure(figsize=(8, 6))
        # plt.hist(theta, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of theta")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(phi, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of phi")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()

        N_vectors = N_vectors.T
        R_vectors = R_vectors.T

        # Stack the arrays into a 2D array (27 x 3)
        stacked_arrays = np.power(10, np.vstack((k_xx, k_yy, k_zz)).T)

        # Create diagonal matrices using broadcasting with einsum
        diagonal_matrices = np.zeros((N_vectors.shape[0], 3, 3))
        indices = np.arange(3)
        diagonal_matrices[:, indices, indices] = stacked_arrays

        radius = np.ones((N_vectors.shape[0], 2))

        transform_matrices = FractureSet.transform_mat_static(normal=N_vectors, shape_axis=R_vectors, radius=radius)

        cond_3d = transform_matrices @ diagonal_matrices @ np.transpose(transform_matrices, axes=(0, 2, 1))

        # n_tensors = cond_3d.shape[0]
        #
        # # Step 1: Compute eigenvalues and eigenvectors for all tensors
        # eigenvalues, eigenvectors = np.linalg.eigh(cond_3d)
        #
        # # eigenvalues: Shape (4096, 3)
        # # eigenvectors: Shape (4096, 3, 3) where each row contains one eigenvector set
        #
        # dominant_eigenvectors = eigenvectors[np.arange(n_tensors), :, np.argmax(eigenvalues, axis=1)]
        #
        # # Step 3: Compute azimuth and elevation
        # # Azimuth: Angle in the XY-plane
        # azimuths = np.arctan2(dominant_eigenvectors[:, 1], dominant_eigenvectors[:, 0])  # Shape (4096,)
        #
        # # Elevation: Angle from the Z-axis
        # elevations = np.arccos(
        #     dominant_eigenvectors[:, 2] / np.linalg.norm(dominant_eigenvectors, axis=1))  # Shape (4096,)
        #
        # print("azimuths shape ", azimuths.shape)
        #
        # #single_angles = np.sqrt(azimuths_deg ** 2 + elevations_deg ** 2)
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(azimuths, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of azimuths")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(elevations, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of elevations")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        # print("cond_3d.shape ", cond_3d.shape)
        # eigenvalues = np.linalg.eigvals(cond_3d)
        # assert np.all(eigenvalues > 0)
        return cond_3d, barycenters

@attr.s(auto_attribs=True)
class GSToolsBulk3DEffective:
    mean_log_conductivity: Tuple[float, float]
    cov_log_conductivity: Optional[List[List[float]]]
    angle_mean: float = attr.ib(converter=float)
    angle_concentration: float = attr.ib(converter=float)
    corr_lengths_x: Optional[List[float]]  # [15, 0, 0]
    corr_lengths_y: Optional[List[float]]  # = [3, 0, 0]
    corr_lengths_z: Optional[List[float]]  # = [3, 0, 0]
    anis_angles: Optional[List[float]]  # = [0, 0, 0]
    anisotropy: float = None
    rotation: float = None
    mode_no: int = 1000
    angle_var = 1
    _rf_sample = None
    _mesh_data= None
    log = False
    seed= None
    mode: Optional[str] = None
    structured: Optional[bool] = False

    def _pca(self, mean_k_xx_yy_zz, cov_matrix_k_xx_yy_zz):
        n_samples = 10000

        # print("mean k xx yy ", mean_k_xx_yy)
        # print("cov matrix k xx yy ", cov_matrix_k_xx_yy)
        # print("self. angle var ", self.angle_var)
        np.random.seed(seed=self.seed)
        gstools.random.RNG(seed=self.seed)

        samples = np.random.multivariate_normal(mean=mean_k_xx_yy_zz, cov=cov_matrix_k_xx_yy_zz, size=n_samples)

        # sample_means = [np.mean(samples[:, 0]), np.mean(samples[:, 1])]
        # sample_vars = [np.var(samples[:, 0]), np.var(samples[:, 1])]
        # print("sample means: {} vars: {}".format(sample_means, sample_vars))
        covariance_matrix = np.cov(samples.T)
        #print("covariance matrix ", covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        self._projection_matrix = (eigen_vectors.T[:][:]).T

        #print("cov_matrix_k_xx_yy_zz ", cov_matrix_k_xx_yy_zz)

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix_k_xx_yy_zz)
        self._projection_matrix = (eigen_vectors.T[:][:]).T
        p_components = samples.dot(self._projection_matrix)

        #print("p components shape ", p_components.shape)

        pc_means = [np.mean(p_components[:, 0]), np.mean(p_components[:, 1]), np.mean(p_components[:, 2])]
        pc_vars = [np.var(p_components[:, 0]), np.var(p_components[:, 1]), np.var(p_components[:, 2])]
        #print("pc means: {} vars: {}".format(pc_means, pc_vars))

        ##########
        ## K_xx ##
        ##########
        if self.corr_lengths_x[0] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[0], self.corr_lengths_y[0], self.corr_lengths_z[0]]

        self._model_k_xx = gstools.Exponential(dim=3, var=pc_vars[0], len_scale=len_scale, angles=self.anis_angles[0])
        # self._model_k_xx_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale, angles=self.anis_angles[0])

        ##########
        ## K_yy ##
        ##########
        if self.corr_lengths_x[1] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[1], self.corr_lengths_y[1], self.corr_lengths_z[1]]

        #print("len scale ", len_scale)

        self._model_k_yy = gstools.Exponential(dim=3, var=pc_vars[1], len_scale=len_scale, angles=self.anis_angles[1])
        # self._model_k_yy_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
        #                                            angles=self.anis_angles[1])

        ##########
        ## K_zz ##
        ##########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2], self.corr_lengths_z[2]]
        # print("len scale ", len_scale)

        self._model_k_zz = gstools.Exponential(dim=3, var=pc_vars[2], len_scale=len_scale, angles=self.anis_angles[2])
        # self._model_k_zz_new = gstools.Exponential(dim=3, var=1, len_scale=len_scale,
        #                                            angles=self.anis_angles[2])

        ###########
        ## Angle ##
        ###########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2]]

        #model_angle = gstools.Gaussian(dim=2, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

        self._cov_model_angle = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

        # self._R_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        # self._R_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        #
        # #self._R_x = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        # #self._R_y = gstools.Gaussian(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        #
        # self._N_x = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        # self._N_y = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])
        # self._N_z = gstools.Exponential(dim=3, var=self.angle_var, len_scale=len_scale, angles=self.anis_angles[2])

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
                                                                       seed=self.seed + 100, mode=self.mode,
                                                                       structured=self.structured
                                                                       ))

        field_k_yy = cf.Field('k_yy', cf.GSToolsSpatialCorrelatedField(self._model_k_yy, log=self.log,
                                                                       #sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       #mu=pca_means[1]
                                                                       seed=self.seed + 200, mode=self.mode,
                                                                       structured=self.structured
                                                                       ))

        field_k_zz = cf.Field('k_zz', cf.GSToolsSpatialCorrelatedField(self._model_k_zz, log=self.log,
                                                                       # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       # mu=pca_means[1]
                                                                       seed=self.seed + 300, mode=self.mode,
                                                                       structured=self.structured
                                                                       ))

        # field_k_xx_new = cf.Field('k_xx_new', cf.GSToolsSpatialCorrelatedField(self._model_k_xx_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[0,0]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[0]
        #                                                                seed=self.seed + 1100
        #                                                                ))
        #
        # field_k_yy_new = cf.Field('k_yy_new', cf.GSToolsSpatialCorrelatedField(self._model_k_yy_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[1]
        #                                                                seed=self.seed + 1200
        #                                                                ))
        #
        # field_k_zz_new = cf.Field('k_zz_new', cf.GSToolsSpatialCorrelatedField(self._model_k_zz_new, log=self.log,
        #                                                                # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
        #                                                                mode_no=self.mode_no,
        #                                                                # mu=pca_means[1]
        #                                                                seed=self.seed + 1300
        #                                                                ))

        angle_srf = cf.GSToolsSpatialCorrelatedField(self._cov_model_angle,
                                         sigma=np.sqrt(self.angle_var),
                                         mode_no=self.mode_no, mode=self.mode, structured=self.structured)

        field_R_x = cf.Field('R_x', angle_srf)
        field_R_y = cf.Field('R_y', angle_srf)
        field_N_x = cf.Field('N_x', angle_srf)
        field_N_y = cf.Field('N_y', angle_srf)
        field_N_z = cf.Field('N_z', angle_srf)

        # field_R_x = cf.Field('R_x', cf.GSToolsSpatialCorrelatedField(self._R_x,
        #                                                                  sigma=np.sqrt(self.angle_var),
        #                                                                  mode_no=self.mode_no,
        #                                                                  seed=self.seed + 400, mode=self.mode
        #                                                                  ))
        #
        # field_R_y = cf.Field('R_y', cf.GSToolsSpatialCorrelatedField(self._R_x,
        #                                                                  sigma=np.sqrt(self.angle_var),
        #                                                                  mode_no=self.mode_no,
        #                                                                  seed=self.seed + 500, mode=self.mode
        #                                                                  ))
        #
        # field_N_x = cf.Field('N_x', cf.GSToolsSpatialCorrelatedField(self._N_x,
        #                                                              sigma=np.sqrt(self.angle_var),
        #                                                              mode_no=self.mode_no,
        #                                                              seed=self.seed + 600, mode=self.mode
        #                                                              ))
        #
        # field_N_y = cf.Field('N_y', cf.GSToolsSpatialCorrelatedField(self._N_y,
        #                                                              sigma=np.sqrt(self.angle_var),
        #                                                              mode_no=self.mode_no,
        #                                                              seed=self.seed + 700, mode=self.mode
        #                                                              ))
        #
        # field_N_z = cf.Field('N_z', cf.GSToolsSpatialCorrelatedField(self._N_z,
        #                                                              sigma=np.sqrt(self.angle_var),
        #                                                              mode_no=self.mode_no,
        #                                                              seed=self.seed + 800, mode=self.mode
        #                                                              ))

        self._fields = cf.Fields([field_k_xx, field_k_yy, field_k_zz])
        self._angle_fields = cf.Fields([field_R_x, field_R_y, field_N_x, field_N_y, field_N_z],
                                       seeds=[self.seed + 400, self.seed + 500, self.seed + 600,
                                              self.seed + 700, self.seed + 800])
        #print("self._fields ", self._fields)

    def generate_field(self, barycenters):
        if self._rf_sample is None:
            #if self.model_k_xx is None:
            self._create_field(self.mean_log_conductivity, self.cov_log_conductivity)

            #grid_barycenters = fem_grid.grid.barycenters()

            #print("barycenters ", barycenters.shape)

            #self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            #print("mesh data ", list(self._mesh_data.keys()))
            # print("mesh data ele ids ", self._mesh_data["ele_ids"])
            # print("mesh_data['points'] ", self._mesh_data['points'])
            #self._fields.set_points(self._mesh_data['points'], self._mesh_data['point_region_ids'], self._mesh_data['region_map'])

            self._fields.set_points(barycenters)
            self._angle_fields.set_points(barycenters)

            self._rf_sample = self._fields.sample()
            self._angle_fields_sample = self._angle_fields.sample()

            #print("angle fields sample shape ", self._angle_fields_sample["angle"].shape)

            # # print("self._pc_mean ", self._pc_means)
            # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
            # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
            # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))

            self._rf_sample["k_xx"] += self._pc_means[0]
            self._rf_sample["k_yy"] += self._pc_means[1]
            self._rf_sample["k_zz"] += self._pc_means[2]

            # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
            # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
            # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))

            srf_data = np.array([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])

            inv_srf_data = np.matmul(srf_data.T, self._projection_matrix.T).T

            self._rf_sample["k_xx"] = inv_srf_data[0, :]
            self._rf_sample["k_yy"] = inv_srf_data[1, :]
            self._rf_sample["k_zz"] = inv_srf_data[2, :]

        k_xx = self._rf_sample["k_xx"]
        k_yy = self._rf_sample["k_yy"]
        k_zz = self._rf_sample["k_zz"]

        # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
        # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))
        # print("mean k zz ", np.mean(self._rf_sample["k_zz"]))
        #
        # print("var k xx ", np.var(self._rf_sample["k_xx"]))
        # print("var k yy ", np.var(self._rf_sample["k_yy"]))
        # print("var k zz ", np.var(self._rf_sample["k_zz"]))

        # data = np.vstack([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])
        #
        # # Calculate the covariance matrix
        # cov_matrix = np.cov(data)
        # # print("cov matrix ", cov_matrix)
        # # exit()

        R_x = self._angle_fields_sample["R_x"]
        R_y = self._angle_fields_sample["R_y"]
        N_x = self._angle_fields_sample["N_x"]
        N_y = self._angle_fields_sample["N_y"]
        N_z = self._angle_fields_sample["N_z"]

        #print("R_x ", self._rf_sample["R_x"])

        # plt.figure(figsize=(8, 6))
        # plt.hist(R_x, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of R_x")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # #print("R_y ", self._rf_sample["R_y"])
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(R_y, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of R_y")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()


        R_vectors = np.stack([R_x,  R_y])
        R_vectors = R_vectors / np.linalg.norm(R_vectors, axis=0)


        #print("R vectors ", R_vectors)

        #angle_radians = np.arctan2(R_vectors[0, :], R_vectors[1, :])

        # print("angle radians ", angle_radians)
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(angle_radians, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of angle_radians")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        #
        # print("barycenters ", grid_barycenters.shape)
        # print("angle_radians shape ", angle_radians.shape)

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Scatter plot with color mapping
        # scatter = ax.scatter(grid_barycenters[:, 0], grid_barycenters[:, 1], grid_barycenters[:, 2],
        #                      c=angle_radians, cmap='viridis', marker='o')
        #
        # # Add a color bar
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        # cbar.set_label('Scalar Values', rotation=270, labelpad=15)

        # # Set axis labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # Extract X, Y, Z coordinates
        #x, y, z = barycenters[:, 0], barycenters[:, 1], barycenters[:, 2]

        # from mayavi import mlab
        #
        # # Create a 3D scatter plot
        # mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # White background
        # scatter = mlab.points3d(x, y, z, angle_radians,
        #                         scale_mode='none',  # Fix point size
        #                         scale_factor=0.02,  # Adjust size
        #                         colormap='viridis')  # Choose colormap
        #
        # # Add color bar
        # mlab.colorbar(scatter, title="Scalar Values", orientation="vertical")
        #
        # # Set axis labels
        # mlab.xlabel("X")
        # mlab.ylabel("Y")
        # mlab.zlabel("Z")

        N_vectors = np.stack([N_x,  N_y, N_z])
        N_vectors = N_vectors / np.linalg.norm(N_vectors, axis=0)

        #phi = np.arctan2(N_vectors[1, :], N_vectors[0, :])

        #magnitude = np.sqrt(N_vectors[0, :] ** 2 + N_vectors[1, :] ** 2 + N_vectors[2, :] ** 2)

        #print("magnitude ", magnitude)

        # Compute the polar angle (theta) in radians
        #theta = np.arccos(N_vectors[2,:] / magnitude)

        # plt.figure(figsize=(8, 6))
        # plt.hist(theta, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of theta")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(phi, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of phi")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()

        N_vectors = N_vectors.T
        R_vectors = R_vectors.T

        # Stack the arrays into a 2D array (27 x 3)
        stacked_arrays = np.power(10, np.vstack((k_xx, k_yy, k_zz)).T)

        # Create diagonal matrices using broadcasting with einsum
        diagonal_matrices = np.zeros((N_vectors.shape[0], 3, 3))
        indices = np.arange(3)
        diagonal_matrices[:, indices, indices] = stacked_arrays

        radius = np.ones((N_vectors.shape[0], 2))

        transform_matrices = FractureSet.transform_mat_static(normal=N_vectors, shape_axis=R_vectors, radius=radius)

        cond_3d = transform_matrices @ diagonal_matrices @ np.transpose(transform_matrices, axes=(0, 2, 1))

        # if isinstance(barycenters, tuple) and len(barycenters) == 3:
        #     X, Y, Z = np.meshgrid(barycenters[0], barycenters[1], barycenters[2], indexing='ij')  # shape (18, 18, 18)
        #     barycenters = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # shape (5832, 3)
        #
        #     #barycenters = np.stack([barycenters[0], barycenters[1], barycenters[2]], axis=-1).reshape(-1, 3)
        #
        #
        # #print("barycenters[0].shape ", barycenters[0].shape)
        #
        # #barycenters = np.stack([barycenters[0], barycenters[1], barycenters[2]], axis=-1).reshape(-1, 3)
        #
        #
        # print("barycenters ", barycenters.shape)

        # n_tensors = cond_3d.shape[0]
        #
        # # Step 1: Compute eigenvalues and eigenvectors for all tensors
        # eigenvalues, eigenvectors = np.linalg.eigh(cond_3d)
        #
        # # eigenvalues: Shape (4096, 3)
        # # eigenvectors: Shape (4096, 3, 3) where each row contains one eigenvector set
        #
        # dominant_eigenvectors = eigenvectors[np.arange(n_tensors), :, np.argmax(eigenvalues, axis=1)]
        #
        # # Step 3: Compute azimuth and elevation
        # # Azimuth: Angle in the XY-plane
        # azimuths = np.arctan2(dominant_eigenvectors[:, 1], dominant_eigenvectors[:, 0])  # Shape (4096,)
        #
        # # Elevation: Angle from the Z-axis
        # elevations = np.arccos(
        #     dominant_eigenvectors[:, 2] / np.linalg.norm(dominant_eigenvectors, axis=1))  # Shape (4096,)
        #
        # print("azimuths shape ", azimuths.shape)
        # #single_angles = np.sqrt(azimuths_deg ** 2 + elevations_deg ** 2)
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(azimuths, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of azimuths")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        #
        # plt.figure(figsize=(8, 6))
        # plt.hist(elevations, bins=50, density=True, alpha=0.6, edgecolor='black')
        # plt.title("Histogram of elevations")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.show()
        # print("cond_3d.shape ", cond_3d.shape)
        # eigenvalues = np.linalg.eigvals(cond_3d)
        # assert np.all(eigenvalues > 0)
        return cond_3d, barycenters

