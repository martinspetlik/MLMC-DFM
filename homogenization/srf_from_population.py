import os
import numpy as np
from itertools import product


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

        fine_step = config_dict["fine"]["step"]
        print("fine step ", fine_step)
        print("coarse step ", config_dict["coarse"]["step"])

        coarse_step = config_dict["coarse"]["step"]

        hom_block_size = int(fine_step * 3)

        level_parameters = list(np.squeeze(config_dict["sim_config"]["level_parameters"], axis=1))
        current_level_index = list(np.squeeze(config_dict["sim_config"]["level_parameters"], axis=1)).index(fine_step)
        previous_level_fine_step = level_parameters[current_level_index + 1]

        previous_level_hom_block_size = int(previous_level_fine_step * 3)

        if "hom_box_fine_step_mult" in config_dict["sim_config"]:
            hom_block_size = int(fine_step * config_dict["sim_config"]["hom_box_fine_step_mult"])
            previous_level_hom_block_size = int(previous_level_fine_step * config_dict["sim_config"]["hom_box_fine_step_mult"])

        if coarse_step == 0:
            orig_domain_box = config_dict["sim_config"]["geometry"]["orig_domain_box"]
            print("orig_domain_box ", orig_domain_box)
            larger_domain_size = orig_domain_box[0] + previous_level_hom_block_size + previous_level_fine_step
        else:
            orig_domain_box = config_dict["sim_config"]["geometry"]["orig_domain_box"]
            print("orig_domain_box ", orig_domain_box)
            larger_domain_size = orig_domain_box[0] + hom_block_size + fine_step

        # larger_domain_reduced_by_homogenization = larger_domain_size
        # for hb_size in hom_block_sizes:
        #     larger_domain_reduced_by_homogenization -= hb_size

        print("new larger_domain_size ", larger_domain_size)
        #print("larger_domain_reduced_by_homogenization ", larger_domain_reduced_by_homogenization)
        print("hom block size ", hom_block_size)
        print("previous level hom block size ", previous_level_hom_block_size)

        self.centers_3d = SRFFromTensorPopulation.calculate_all_centers(larger_domain_size, previous_level_hom_block_size, overlap=previous_level_hom_block_size/2)

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
    def calculate_all_centers(domain_size, block_size, overlap):
        start = -domain_size / 2 #+ block_size / 2
        end = domain_size / 2 #- block_size / 2

        length = end - start
        stride = block_size - overlap

        # Calculate number of intervals, rounding up to cover full domain
        num_intervals = int(np.ceil(length / stride))
        n_centers = num_intervals + 1  # total centers

        # Recalculate exact stride to cover full domain exactly
        exact_stride = length / (n_centers - 1)

        centers_1d = start + exact_stride * np.arange(n_centers)
        print("centers_1d ", centers_1d)

        # Generate 3D centers meshgrid
        z, y, x = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing='ij')
        centers_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        print("centers_3d.shape ", centers_3d.shape)

        return centers_3d

    @staticmethod
    def symmetrize_cond_tns(cond_tns):
        sym_values = (cond_tns[:, 1] + cond_tns[:, 2])/2
        cond_tns = np.delete(cond_tns, 2, 1)
        cond_tns[:, 1] = sym_values
        return cond_tns

    def _get_tensors(self, config_dict):
        cond_pop_file = config_dict["fine"]['cond_tn_pop_file']
        cond_pop_coords_file = config_dict["fine"]['cond_tn_pop_coords_file']
        self._cond_tns = np.load(cond_pop_file)
        self._cond_tns_coords = np.load(cond_pop_coords_file)

    def generate_field(self):
        # print("barycenters ", barycenters)
        # print("self._cond_tns shape ", self._cond_tns.shape)
        #
        # print("self._cond_tns_coords.shape ", self._cond_tns_coords.shape)
        #
        # barycenters = np.array([barycenters[0], barycenters[1], barycenters[2]]).T

        #print("barycenters shape ", barycenters.shape)

        # Sample N indices independently with replacement from the M tensor samples
        indices = np.random.choice(self._cond_tns.shape[0], size=self.centers_3d.shape[0], replace=True)

        # Select sampled tensors
        sampled_tensors = self._cond_tns[indices]  # shape: (N, 3, 3)

        print("sampled_tensors ", sampled_tensors.shape)

        return sampled_tensors, self.centers_3d