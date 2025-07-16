import numpy as np
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
import os
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

#
# logging.getLogger('bgem').disabled = True
#
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings('ignore')



class SRFFromTensorPopulation:
    def __init__(self, config_dict):
        self._cond_tns = None
        self._config_dict = config_dict
        self._get_tensors(config_dict)
        self._srf_gstools_model = None
        self._svd_model = {}

        self.mean_log_conductivity = None
        if "mean_log_conductivity" in config_dict:
            self.mean_log_conductivity = config_dict["mean_log_conductivity"]

        self._rf_sample = None

        # if config_dict["sim_config"]["bulk_fine_sample_model"] == "choice":
        #     srf_model = SpatialCorrelatedFieldHomChoice()
        #     srf_model._cond_tensors = self._cond_tns
        #     field_cond_tn = cf.Field('cond_tn', srf_model)
        #     self._fields = cf.Fields([field_cond_tn])
        #
        # elif config_dict["sim_config"]["bulk_fine_sample_model"] == "srf_gstools":
        #     self._srf_gstools_model = FineHomSRFGstools(self._cond_tns, config_dict)
        #     self._fields = self._srf_gstools_model._fields
        #
        # elif config_dict["sim_config"]["bulk_fine_sample_model"] == "svd":
        #     avg_len_scale_list, avg_var_list = FineHomSRFGstools._calc_var_len_scale(config_dict)
        #     print("avg len scale list ", avg_len_scale_list)
        #     print("avg var list ", avg_var_list)
        #     cond_tn_values, transform_obj = FineHomSRFGstools.normalize_cond_tns(self._cond_tns)
        #     self._svd_model["transform_obj"] = transform_obj
        #     print("corr length ", np.mean(avg_len_scale_list))
        #     srf_model = SpatialCorrelatedFieldHomSVD(corr_exp="exp", corr_length=np.mean(avg_len_scale_list))
        #     srf_model._cond_tensors = cond_tn_values
        #     field_cond_tn = cf.Field('cond_tn', srf_model)
        #     self._fields = cf.Fields([field_cond_tn])
        #
        # else:  # Distr fit by GaussianMixtures
        #     srf_model = SpatialCorrelatedFieldHomGMM()
        #     srf_model._cond_tensors = self._cond_tns
        #     field_cond_tn = cf.Field('cond_tn', srf_model)
        #     self._fields = cf.Fields([field_cond_tn])
        #     print("SRF MODEL corr length: {}, sigma: {}".format(srf_model._corr_length, srf_model.sigma))


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
        self._cond_tns = np.load(cond_pop_file)

    def generate_field(self, barycenters):
        print("barycenters ", barycenters)
        print("self._cond_tns shape ", self._cond_tns.shape)

        barycenters = np.array([barycenters[0], barycenters[1], barycenters[2]]).T

        print("barycenters shape ", barycenters.shape)

        return cond_3d, barycenters