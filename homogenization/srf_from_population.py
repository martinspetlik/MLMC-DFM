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


logging.getLogger('bgem').disabled = True

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')



class BulkHomogenizationFineSample(BulkBase):
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

        if config_dict["sim_config"]["bulk_fine_sample_model"] == "choice":
            srf_model = SpatialCorrelatedFieldHomChoice()
            srf_model._cond_tensors = self._cond_tns
            field_cond_tn = cf.Field('cond_tn', srf_model)
            self._fields = cf.Fields([field_cond_tn])

        elif config_dict["sim_config"]["bulk_fine_sample_model"] == "srf_gstools":
            self._srf_gstools_model = FineHomSRFGstools(self._cond_tns, config_dict)
            self._fields = self._srf_gstools_model._fields

        elif config_dict["sim_config"]["bulk_fine_sample_model"] == "svd":
            avg_len_scale_list, avg_var_list = FineHomSRFGstools._calc_var_len_scale(config_dict)
            print("avg len scale list ", avg_len_scale_list)
            print("avg var list ", avg_var_list)
            cond_tn_values, transform_obj = FineHomSRFGstools.normalize_cond_tns(self._cond_tns)
            self._svd_model["transform_obj"] = transform_obj
            print("corr length ", np.mean(avg_len_scale_list))
            srf_model = SpatialCorrelatedFieldHomSVD(corr_exp="exp", corr_length=np.mean(avg_len_scale_list))
            srf_model._cond_tensors = cond_tn_values
            field_cond_tn = cf.Field('cond_tn', srf_model)
            self._fields = cf.Fields([field_cond_tn])

        else:  # Distr fit by GaussianMixtures
            srf_model = SpatialCorrelatedFieldHomGMM()
            srf_model._cond_tensors = self._cond_tns
            field_cond_tn = cf.Field('cond_tn', srf_model)
            self._fields = cf.Fields([field_cond_tn])
            print("SRF MODEL corr length: {}, sigma: {}".format(srf_model._corr_length, srf_model.sigma))


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

    def element_data(self, mesh, eid):
        if self._rf_sample is None:
            self._mesh_data = BulkFieldsGSTools.extract_mesh(mesh)
            self._fields.set_points(self._mesh_data['points'],
                                    self._mesh_data['point_region_ids'],
                                    self._mesh_data['region_map'])
            self._rf_sample = self._fields.sample()

            if self._srf_gstools_model is not None:
                self._rf_sample["k_xx"] += self._srf_gstools_model._pc_means[0]
                self._rf_sample["k_xy"] += self._srf_gstools_model._pc_means[1]
                self._rf_sample["k_yy"] += self._srf_gstools_model._pc_means[2]

                srf_data = np.array([self._rf_sample["k_xx"], self._rf_sample["k_xy"], self._rf_sample["k_yy"]])
                inv_srf_data = np.matmul(srf_data.T, self._srf_gstools_model.projection_matrix.T)

                inv_trf_data = np.empty((inv_srf_data.shape)).T
                if len(self._srf_gstools_model.transform_obj) > 0:
                    for i in range(inv_srf_data.shape[1]):
                        if self._srf_gstools_model.transform_obj[i] is not None:
                            inv_transformed_data = self._srf_gstools_model.transform_obj[i].inverse_transform(inv_srf_data[:, i].reshape(-1, 1))
                            if i != 1:
                                inv_trf_data[i][...] = np.exp(np.reshape(inv_transformed_data, inv_srf_data[:, i].shape))
                            else:
                                inv_trf_data[i][...] = np.reshape(inv_transformed_data, inv_srf_data[:, i].shape)
                        else:
                            inv_trf_data[i][...] = inv_srf_data[:, i]
                    inv_srf_data = inv_trf_data.T

                self._rf_sample["k_xx"] = inv_srf_data[:, 0]
                self._rf_sample["k_xy"] = inv_srf_data[:, 1]
                self._rf_sample["k_yy"] = inv_srf_data[:, 2]

                # bin_center, gamma = gstools.vario_estimate((self._mesh_data['points'][:, 0], self._mesh_data['points'][:, 1]),self._rf_sample["k_xx"])
                # fit_model = gstools.Exponential(dim=2)
                # k_xx_params = fit_model.fit_variogram(bin_center, gamma, nugget=False)
                # print("inv srf k xx params ", k_xx_params)
            elif len(self._svd_model) > 0:
                transform_obj = self._svd_model["transform_obj"]
                inv_srf_data = self._rf_sample["cond_tn"]
                inv_trf_data = np.empty((inv_srf_data.shape)).T
                if len(transform_obj) > 0:
                    for i in range(inv_srf_data.shape[1]):
                        if transform_obj[i] is not None:
                            inv_transformed_data = transform_obj[i].inverse_transform(
                                inv_srf_data[:, i].reshape(-1, 1))
                            if i != 1:
                                inv_trf_data[i][...] = np.exp(
                                    np.reshape(inv_transformed_data, inv_srf_data[:, i].shape))
                            else:
                                inv_trf_data[i][...] = np.reshape(inv_transformed_data, inv_srf_data[:, i].shape)
                        else:
                            inv_trf_data[i][...] = inv_srf_data[:, i]
                    inv_srf_data = inv_trf_data.T
                self._rf_sample["cond_tn"] = inv_srf_data

        ele_idx = np.where(self._mesh_data["ele_ids"] == eid)

        if self._srf_gstools_model is None:
            cond_tn = self._rf_sample["cond_tn"][ele_idx]
            cond_tn = [[cond_tn[0][0], cond_tn[0][1]], [cond_tn[0][1], cond_tn[0][2]]]
        else:
            k_xx = self._rf_sample["k_xx"][ele_idx][0]
            k_xy = self._rf_sample["k_xy"][ele_idx][0]
            k_yy = self._rf_sample["k_yy"][ele_idx][0]
            cond_tn = [[k_xx, k_xy], [k_xy, k_yy]]

        e_val, e_vec = np.linalg.eigh(cond_tn)

        # Check if any eigenvalues are non-positive
        while any(e_val <= 0):
            # Add a positive definite matrix (e.g., identity matrix scaled by a small positive constant)
            eps = np.mean([cond_tn[0][0], cond_tn[1][1]]) * 0.2
            cond_tn = cond_tn + eps * np.eye(2)
            e_val, e_vec = np.linalg.eigh(cond_tn)

        #print("cond tn ", cond_tn)
        #cond_tn = [[cond_tn[0][0], cond_tn[0][1]], [cond_tn[0][1], cond_tn[0][2]]]

        # cond_tn = cond_tn.reshape(2, 2)
        # sym_value = (cond_tn[0,1] + cond_tn[1,0])/2
        # cond_tn[0,1] = cond_tn[1,0] = sym_value

        cond_tn[0][0] = np.abs(cond_tn[0][0])
        cond_tn[1][1] = np.abs(cond_tn[1][1])

        return 1.0, cond_tn