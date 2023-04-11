import os
import os.path
import numpy as np
import yaml
import gmsh_io
from rasterization import rasterize
import time
#from matplotlib import pyplot as plt

MESH_FILE = "mesh_fine.msh"
FIELDS_MESH_FILE = "fields_fine.msh"
SUMMARY_FILE = "summary.yaml"


def create_output(sample_dir, symmetrize=True):
    """
    Load tensor form yaml file and save it as compressed numpy array
    :param sample_dir: sample directory
    :param symmetrize: bool, if True symmetrize conductivity tensor
    :return: None
    """
    with open(os.path.join(sample_dir, SUMMARY_FILE), "r") as f:
        summary_dict = yaml.load(f, Loader=yaml.Loader)
    cond_tn = np.array(summary_dict['fine']['cond_tn'])[0]

    if cond_tn.shape[0] == 2:
        if symmetrize:
            cond_tn[0,1] = (cond_tn[0,1] + cond_tn[1,0])/2

        tn3d = np.eye(3)
        tn3d[0:2, 0:2] = cond_tn
        cond_tn = tn3d

    elif symmetrize:
        cond_tn[0, 1] = (cond_tn[0, 1] + cond_tn[1, 0]) / 2
        cond_tn[0, 2] = (cond_tn[0, 2] + cond_tn[2, 0]) / 2
        cond_tn[1, 2] = (cond_tn[1, 2] + cond_tn[2, 1]) / 2

    up_tr = cond_tn[np.triu_indices(3)]
    np.save(os.path.join(sample_dir, "output_tensor"), up_tr)
    return cond_tn
    # np.savez_compressed(os.path.join(sample_dir, "output_tensor_compressed"), data=cond_tn.ravel)


def create_input(sample_dir, n_pixels_x=256, feature_names=[['conductivity_tensor'], ['cross_section']]):
    """
    Create inputs - get mesh properties and corresponding values of random fields
                  - rasterize bulk and fracture properties separately
                  - save the rasterized arrays in a compressed way, i.e. np.saved_compressed
    :param sample_dir: sample directory
    :param n_pixels_x: number of pixels in one dimension, we consider square inputs
    :param feature_names: features to
    :return: None
    """
    n_tn_elements = 6  # number of tensor elements in use - upper triangular matrix of 3x3 tensor
    mesh = os.path.join(sample_dir, MESH_FILE)

    mesh_nodes, triangles, lines, ele_ids = extract_mesh_gmsh_io(mesh, image=True)

    field_mesh = os.path.join(sample_dir, FIELDS_MESH_FILE)
    if os.path.exists(field_mesh):
        features, all_fields = get_node_features(field_mesh, feature_names)

        cond_tn_elements_triangles = []
        cond_tn_elements_lines = []
        cs_lines = []
        skip_lower_tr_indices = [3, 6, 7]
        for _ in range(n_tn_elements):
            cond_tn_elements_triangles.append(dict(zip(triangles.keys(), np.zeros(len(triangles.keys())))))
            cond_tn_elements_lines.append(dict(zip(lines.keys(), np.zeros(len(lines.keys())))))
            cs_lines = dict()

        for feature_name in feature_names:
            feature_name = feature_name[0]
            if feature_name == "conductivity_tensor":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                    val = np.delete(val, skip_lower_tr_indices)
                    if e_id in triangles:
                        for idx, v in enumerate(val):
                            cond_tn_elements_triangles[idx][e_id] = v

                    elif e_id in lines:
                        for idx, v in enumerate(val):
                            cond_tn_elements_lines[idx][e_id] = v

            elif feature_name == "cross_section":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                    if e_id in lines:
                        cs_lines.setdefault(val[0], []).append(e_id)

        bulk_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))
        fractures_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))

        for k in range(n_tn_elements):
            trimesh, cvs_lines = rasterize(mesh_nodes, triangles, cond_tn_elements_triangles[k],
                      lines, cond_tn_elements_lines[k], cs_lines, n_pixels_x, save_image=True)

            bulk_data_array[k] = np.flip(trimesh, axis=0)
            fractures_data_array[k] = np.flip(cvs_lines, axis=0)

        np.savez_compressed(os.path.join(sample_dir, "bulk"), data=bulk_data_array)
        np.savez_compressed(os.path.join(sample_dir, "fractures"), data=fractures_data_array)

        return bulk_data_array, fractures_data_array


def join_fields(fields, f_names):
    if len(f_names) > 0:
        x_name = len(set([*fields[f_names[0]]]))
    assert all(x_name == len(set([*fields[f_n]])) for f_n in f_names)

    joint_dict = {}
    for f_n in f_names:
        for key, item in fields[f_n].items():
            joint_dict.setdefault(key, 0)
            if joint_dict[key] != 0 and np.squeeze(item) != 0:
                raise ValueError("Just one field value should be non zero for each element")
            joint_dict[key] += np.squeeze(item)

    return joint_dict


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


def extract_mesh_gmsh_io(mesh_file, get_points=False, image=False):
    """
    Extract mesh from file
    :param mesh_file: Mesh file path
    :return: Dict
    """
    mesh = gmsh_io.GmshIO(mesh_file)
    is_bc_region = {}
    region_map = {}
    for name, (id, _) in mesh.physical.items():
        unquoted_name = name.strip("\"'")
        is_bc_region[id] = (unquoted_name[0] == '.')
        region_map[unquoted_name] = id

    bulk_elements = []
    triangles = {}
    lines = {}

    for id, el in mesh.elements.items():
        _, tags, i_nodes = el
        region_id = tags[0]
        if not is_bc_region[region_id]:
            bulk_elements.append(id)

    n_bulk = len(bulk_elements)
    centers = np.empty((n_bulk, 3))
    ele_ids = np.zeros(n_bulk, dtype=int)
    ele_nodes = {}
    point_region_ids = np.zeros(n_bulk, dtype=int)

    for i, id_bulk in enumerate(bulk_elements):
        _, tags, i_nodes = mesh.elements[id_bulk]
        region_id = tags[0]
        centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
        point_region_ids[i] = region_id
        ele_ids[i] = id_bulk
        ele_nodes[id_bulk] = i_nodes

        if len(i_nodes) == 2:
            lines[ele_ids[i]] = i_nodes
        elif len(i_nodes) == 3:
            triangles[ele_ids[i]] = i_nodes

    if get_points:
        min_pt = np.min(centers, axis=0)
        max_pt = np.max(centers, axis=0)
        diff = max_pt - min_pt
        min_axis = np.argmin(diff)
        non_zero_axes = [0, 1, 2]

        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}

    if image:
        return mesh.nodes, triangles, lines, ele_ids
    return ele_nodes


if __name__ == "__main__":
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_no_fractures"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/nn_data/homogenization_samples_dfm"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/nn_data/homogenization_samples_no_fractures"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/nn_data/homogenization_samples_6_5_8_c_2"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/nn_data/charon_samples_no_fractures/test_data"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_charon/"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_3_3_charon/"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_5LMC_L4/"

    start_time = time.time()

    n_pixels_x = 256

    i = 0
    while True:
        sample_dir = os.path.join(data_dir, "sample_{}".format(i))
        if os.path.exists(sample_dir):
            if not os.path.exists(os.path.join(sample_dir, MESH_FILE)) \
                    or not os.path.exists(os.path.join(sample_dir, SUMMARY_FILE)):
                i += 1
                continue

            create_input(sample_dir, n_pixels_x=n_pixels_x)
            create_output(sample_dir, symmetrize=True)
            i += 1

        else:
            break

    stop_time = time.time()
    print("total time: {}, time per sample: {}".format(stop_time - start_time, (stop_time-start_time)/(i+1)))
