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
from bgem_lib_.src.bgem.stochastic.fr_set import FractureSet



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

        ##########
        ## K_yy ##
        ##########
        if self.corr_lengths_x[1] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[1], self.corr_lengths_y[1], self.corr_lengths_z[1]]

        #print("len scale ", len_scale)

        self._model_k_yy = gstools.Exponential(dim=3, var=pc_vars[1], len_scale=len_scale, angles=self.anis_angles[1])

        ##########
        ## K_zz ##
        ##########
        if self.corr_lengths_x[2] == 0:
            len_scale = 1e-15
        else:
            len_scale = [self.corr_lengths_x[2], self.corr_lengths_y[2], self.corr_lengths_z[2]]
        # print("len scale ", len_scale)

        self._model_k_zz = gstools.Exponential(dim=3, var=pc_vars[2], len_scale=len_scale, angles=self.anis_angles[2])


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
                                                                       seed=self.seed + 100
                                                                       ))

        field_k_yy = cf.Field('k_yy', cf.GSToolsSpatialCorrelatedField(self._model_k_yy, log=self.log,
                                                                       #sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       #mu=pca_means[1]
                                                                       seed=self.seed + 200
                                                                       ))

        field_k_zz = cf.Field('k_zz', cf.GSToolsSpatialCorrelatedField(self._model_k_zz, log=self.log,
                                                                       # sigma=np.sqrt(self.cov_log_conductivity[1,1]),
                                                                       mode_no=self.mode_no,
                                                                       # mu=pca_means[1]
                                                                       seed=self.seed + 300
                                                                       ))

        field_R_x = cf.Field('R_x', cf.GSToolsSpatialCorrelatedField(self._R_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no,
                                                                         seed=self.seed + 400
                                                                         ))

        field_R_y = cf.Field('R_y', cf.GSToolsSpatialCorrelatedField(self._R_x,
                                                                         sigma=np.sqrt(self.angle_var),
                                                                         mode_no=self.mode_no,
                                                                         seed=self.seed + 500
                                                                         ))

        field_N_x = cf.Field('N_x', cf.GSToolsSpatialCorrelatedField(self._N_x,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 600
                                                                     ))

        field_N_y = cf.Field('N_y', cf.GSToolsSpatialCorrelatedField(self._N_y,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 700
                                                                     ))

        field_N_z = cf.Field('N_z', cf.GSToolsSpatialCorrelatedField(self._N_z,
                                                                     sigma=np.sqrt(self.angle_var),
                                                                     mode_no=self.mode_no,
                                                                     seed=self.seed + 800
                                                                     ))

        self._fields = cf.Fields([field_k_xx, field_k_yy, field_k_zz, field_R_x, field_R_y, field_N_x, field_N_y, field_N_z])
        #print("self._fields ", self._fields)

    def generate_field(self, fem_grid):
        if self._rf_sample is None:
            #if self.model_k_xx is None:
            self._create_field(self.mean_log_conductivity, self.cov_log_conductivity)


            grid_barycenters = fem_grid.grid.barycenters()

            #self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            #print("mesh data ", list(self._mesh_data.keys()))
            # print("mesh data ele ids ", self._mesh_data["ele_ids"])
            # print("mesh_data['points'] ", self._mesh_data['points'])
            #self._fields.set_points(self._mesh_data['points'], self._mesh_data['point_region_ids'], self._mesh_data['region_map'])

            self._fields.set_points(grid_barycenters)

            self._rf_sample = self._fields.sample()


            # print("self._pc_mean ", self._pc_means)
            # print("mean k xx ", np.mean(self._rf_sample["k_xx"]))
            # print("mean k yy ", np.mean(self._rf_sample["k_yy"]))

            self._rf_sample["k_xx"] += self._pc_means[0]
            self._rf_sample["k_yy"] += self._pc_means[1]
            self._rf_sample["k_zz"] += self._pc_means[2]

            srf_data = np.array([self._rf_sample["k_xx"], self._rf_sample["k_yy"], self._rf_sample["k_zz"]])

            inv_srf_data = np.matmul(srf_data.T, self._projection_matrix.T).T

            self._rf_sample["k_xx"] = inv_srf_data[0, :]
            self._rf_sample["k_yy"] = inv_srf_data[1, :]
            self._rf_sample["k_zz"] = inv_srf_data[2, :]


        k_xx = self._rf_sample["k_xx"]
        k_yy = self._rf_sample["k_yy"]
        k_zz = self._rf_sample["k_zz"]


        R_vectors = np.stack([self._rf_sample["R_x"],  self._rf_sample["R_y"]])
        R_vectors = R_vectors / np.linalg.norm(R_vectors, axis=0)

        N_vectors = np.stack([self._rf_sample["N_x"],  self._rf_sample["N_y"], self._rf_sample["N_z"]])
        N_vectors = N_vectors / np.linalg.norm(N_vectors, axis=0)

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

        eigenvalues = np.linalg.eigvals(cond_3d)
        assert np.all(eigenvalues > 0)
        return cond_3d

