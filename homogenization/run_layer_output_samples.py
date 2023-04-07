import os
import os.path
import subprocess
import numpy as np
import shutil
import time
import os
import re
import copy
import torch
import yaml
from homogenization.sim_sample import DFMSim
from homogenization.both_sample import FlowProblem, BothSample


def calculate_sample(s_dir):
    config = {'fine': {'step': 4.325, 'common_files_dir': '/home/martin/Documents/MLMC-DFM/test/01_cond_field/l_step_4.325_common_files'},
            'coarse': {'step': 10.0, 'common_files_dir': '/home/martin/Documents/MLMC-DFM/test/01_cond_field/l_step_10.0_common_files'},
              'gmsh': '/home/martin/gmsh/bin/gmsh', 'flow123d': ['flow123d/flow123d-gnu:3.1.0', 'flow123d'],
              'fields_params': {'model': 'exp', 'corr_length': 0.1, 'sigma': 1, 'mode_no': 10000}}

    work_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/"
    with open(os.path.join(work_dir, "sim_config.yaml"), "r") as f:
        sim_config = yaml.load(f, Loader=yaml.FullLoader)
        #print("sim_config_dict ", sim_config_dict)

    config["sim_config"] = sim_config
    config["sim_config"]["geometry"]["n_subdomains"] = 1

    input_tensor = np.load(os.path.join(s_dir, "input_tensor.npy")) #  (channels, x_pixels, y_pixels)

    os.chdir(s_dir)
    print("os.getcwd() ", os.getcwd())

    fractures = DFMSim.generate_fractures(config)

    #print("fine_step", config["fine"]["step"])
    #print("coarse_step", config["coarse"]["step"])

    times = {}
    fine_res = 0

    # try:
    #shutil.rmtree("fine")
    if os.path.exists("fields_fine.msh"):
        os.remove("fields_fine.msh")
    if os.path.exists("fields_coarse.msh"):
        os.remove("fields_coarse.msh")
    if os.path.exists("summary.yaml"):
        os.remove("summary.yaml")
    if os.path.exists("mesh_fine.msh"):
        os.remove("mesh_fine.msh")
    if os.path.exists("mesh_fine.brep"):
        os.remove("mesh_fine.brep")
    if os.path.exists("mesh_fine.tmp.geo"):
        os.remove("mesh_fine.tmp.geo")
    if os.path.exists("mesh_fine.tmp.msh"):
        os.remove("mesh_fine.tmp.msh")
    if os.path.exists("mesh_coarse.msh"):
        os.remove("mesh_coarse.msh")
    if os.path.exists("mesh_coarse.brep"):
        os.remove("mesh_coarse.brep")
    if os.path.exists("mesh_coarse.tmp.geo"):
        os.remove("mesh_coarse.tmp.geo")
    if os.path.exists("mesh_coarse.tmp.msh"):
        os.remove("mesh_coarse.tmp.msh")
    if os.path.exists("water_balance.txt"):
        os.remove("water_balance.txt")
    if os.path.exists("water_balance.yaml"):
        os.remove("water_balance.yaml")
    if os.path.exists("flow_fields.msh"):
        os.remove("flow_fields.msh")
    # if os.path.exists("fields_fine.msh")
    #shutil.rmtree("fine")
    # except:

    ####################
    ### fine problem ###
    ####################
    fine_flow = FlowProblem.make_fine((config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
    fine_flow.fr_range = [config["fine"]["step"], fine_flow.fr_range[1]]
    #fine_flow.fr_range = [25, fine_flow.fr_range[1]]

    #print("fine_flow.fr_range ", fine_flow.fr_range)

    make_mesh_start = time.time()


    if config["sim_config"]["geometry"]["n_subdomains"] == 1:
        mesh_file = "/home/martin/Desktop/mesh_fine.msh"
        shutil.copy(mesh_file, os.getcwd())
        positions = {17: [2, 0], 18: [2, 0],
                     19: [2, 1], 20: [2, 1],
                     21: [2, 2], 22: [2, 2],
                     23: [1, 0], 24: [1,0],
                     25: [1, 1], 26: [1,1],
                     27: [1,2], 28: [1,2],
                     29: [0,0], 30: [0,0],
                     31: [0,1], 32: [0,1],
                     33: [0,2], 34: [0,2]}
        elids_same_value = {18: 17, 20: 19, 22: 21, 24: 23, 26: 25, 28: 27, 30: 29, 32: 31, 34: 33}
        fine_flow.make_mesh(mesh_file)
        times['make_mesh'] = time.time() - make_mesh_start
        make_fields_start = time.time()
        fine_flow.make_fields(elids_same_value=elids_same_value, positions=positions, input_tensors=input_tensor)
        times['make_fields'] = time.time() - make_fields_start

    #fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
    #print("run samples ")
    fine_res, status = DFMSim._run_sample(fine_flow, config)
    cond_tensors = DFMSim.homogenization(copy.deepcopy(config))
    DFMSim._save_tensors(cond_tensors)


if __name__ == "__main__":
    data_dir = "/home/martin/Documents/MLMC-DFM_data/layer_outputs/3_3_from_9_9"
    for s_dir in os.listdir(data_dir):
        try:
            l = re.findall(r'sample_[0-9]*', s_dir)[0]
        except IndexError:
            continue
        calculate_sample(os.path.join(data_dir, s_dir))
