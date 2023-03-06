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


def element_volume(mesh, nodes):
    nodes = np.array([mesh.nodes[nid] for nid in nodes])
    if len(nodes) == 1:
        return 0
    elif len(nodes) == 2:
        return np.linalg.norm(nodes[1] - nodes[0])
    elif len(nodes) == 3:
        return np.linalg.norm(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0]))
    else:
        assert False


def read_flow_fields(sample_dir):
    print("sample dir ", sample_dir)
    sym_condition = True
    #sample_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/output/L01_S0000001/homogenization/i_0_j_2_k_3"

    with open(os.path.join(sample_dir, SUMMARY_FILE), "r") as f:
        summary_dict = yaml.load(f, Loader=yaml.Loader)
    bulk_regions = summary_dict['fine']['bulk_regions']
    pressure_loads = summary_dict['fine']['pressure_loads']
    regions = summary_dict['fine']['regions']

    flow_fields_file = os.path.join(sample_dir, "flow_fields.msh")
    if not os.path.exists(flow_fields_file):
        raise FileNotFoundError

    out_mesh = gmsh_io.GmshIO()
    with open(flow_fields_file, "r") as f:
        out_mesh.read(f)
    time_idx = 0
    time, field_cs = out_mesh.element_data['cross_section'][time_idx]

    ele_reg_vol = {eid: (tags[0] - 10000, element_volume(out_mesh, nodes))
                   for eid, (tele, tags, nodes) in out_mesh.elements.items()}

    assert len(field_cs) == len(ele_reg_vol)
    velocity_field = out_mesh.element_data['velocity_p0']

    print("time ", time)
    print("field_cs ", field_cs)

    assert len(field_cs) == len(ele_reg_vol)
    velocity_field = out_mesh.element_data['velocity_p0']
    # print("velocity field ", velocity_field)

    loads = pressure_loads

    print("loads ", loads)
    group_idx = {group_id: i_group for i_group, group_id in enumerate(set(bulk_regions.values()))}
    n_groups = len(group_idx)
    print("n groups ", n_groups)
    group_labels = n_groups * ['_']
    for reg_id, group_id in bulk_regions.items():
        i_group = group_idx[group_id]
        old_label = group_labels[i_group]
        new_label = regions[reg_id].name
        group_labels[i_group] = old_label if len(old_label) > len(new_label) else new_label

    n_directions = len(loads)
    flux_response = np.zeros((n_groups, n_directions, 2))
    area = np.zeros((n_groups, n_directions))
    print("Averaging velocities ...")
    #print("velocity field ", velocity_field)

    velocities = {}
    for i_time, (time, velocity) in velocity_field.items():
        i_time = int(i_time)
        print("i time ", i_time)

        velocities[i_time] = {}

        for eid, ele_vel in velocity.items():
            print("eid: {}, ele_vel:{}".format(eid, ele_vel))

            velocities[i_time][eid] = ele_vel
            reg_id, vol = ele_reg_vol[eid]
            cs = field_cs[eid][0]
            print("vol: {}, cs: {}".format(vol, cs))
            volume = cs * vol
            i_group = group_idx[bulk_regions[reg_id]]
            print("i group ", i_group)
            print("i time ", i_time)
            flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:2]))
            area[i_group, i_time] += volume


    return velocities, pressure_loads, flux_response.shape

    # print("flux response ", flux_response)
    # print("area ", area)
    #
    # flux_response /= area[:, :, None]
    # cond_tensors = {}
    # print("Fitting tensors ...")
    #
    # print("groud_idx ", group_idx.items())
    # for group_id, i_group in group_idx.items():
    #     flux = flux_response[i_group]
    #     print("flux ", flux)
    #     # least square fit for the symmetric conductivity tensor
    #     rhs = flux.flatten()
    #     # columns for the tensor values: C00, C01, C11
    #     pressure_matrix = np.zeros((len(rhs), 4))
    #     if sym_condition:
    #         pressure_matrix = np.zeros((len(rhs) + 1, 4))
    #
    #     for i_load, (p0, p1) in enumerate(loads):
    #         i0 = 2 * i_load
    #         i1 = i0 + 1
    #         # @TODO: skalarni soucin rychlosti s (p0, p1)
    #         # pressure_matrix[i0] = [p0, p1, 0]
    #         # pressure_matrix[i1] = [0, p0, p1]
    #         pressure_matrix[i0] = [p0, p1, 0, 0]
    #         pressure_matrix[i1] = [0, 0, p0, p1]
    #
    #     if sym_condition:
    #         pressure_matrix[-1] = [0, 1, -1, 0]
    #         rhs = np.append(rhs, [0])
    #
    #     print("pressure matrix ", pressure_matrix)
    #     print("rhs ", rhs)
    #     print("rhs type ", type(rhs))
    #     print("rhs ", rhs)
    #
    #     C, residuals, rank, sing_values = np.linalg.lstsq(pressure_matrix, rhs)
    #     #
    #     #
    #
    #     print("residuals ", residuals)
    #     print("rank ", rank)
    #     print("sing values ", sing_values)
    #     # exit()
    #
    #     # C = np.linalg.lstsq(pressure_matrix, rhs)[0]
    #     print("C ", C)
    #     print("(C[1]-C[2])/(C[1]+C[2]/2) ", np.abs((C[1] - C[2]) / ((C[1] + C[2] / 2))))
    #     # C[1] = C[1]
    #     # cond_tn = np.array([[C[0], C[1]], [C[1], C[2]]])
    #     cond_tn = np.array([[C[0], C[1]], [C[2], C[3]]])
    #     diff = np.abs((C[1] - C[2]) / ((C[1] + C[2] / 2)))
    #
    #     # exit()
    #
    #     e_val = None
    #     if residuals > 10 ** (-8):
    #         exit()
    #         e_val = FlowProblem.line_fit(flux)
    #
    #     if i_group < 10:
    #         # if flux.shape[0] < 5:
    #         print("Plot tensor for eid: ", group_id)
    #         print("Fluxes: \n", flux)
    #         print("pressures: \n", loads)
    #         print("cond: \n", cond_tn)
    #         # if e_val is None:
    #         e_val, e_vec = np.linalg.eigh(cond_tn)
    #         print("e val ", e_val)
    #
    #         # print("log mean eval cond", np.log10(np.mean(e_val)))
    #         #self.plot_effective_tensor(flux, cond_tn, self.basename + "_" + group_labels[i_group])
    #         # exit()
    #         # print(cond_tn)
    #     cond_tensors[group_id] = cond_tn
    #
    # print("self.cond_tensors ", cond_tensors)
    # print("self.flux ", flux)
    # print("self. pressure matrix ", pressure_matrix)
    #
    #
    # exit()

def _calculate_effective_tensor(time_cond_bulk, time_velocity_bulk, time_neg_pressure_bulk, pressure_loads,
                                flux_response_shape,
                                upper_corner):

    sym_condition = True
    #sample_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/output/L01_S0000001/homogenization/i_0_j_2_k_3"

    # with open(os.path.join(sample_dir, SUMMARY_FILE), "r") as f:
    #     summary_dict = yaml.load(f, Loader=yaml.Loader)
    # bulk_regions = summary_dict['fine']['bulk_regions']
    # pressure_loads = summary_dict['fine']['pressure_loads']
    # regions = summary_dict['fine']['regions']

    # flow_fields_file = os.path.join(sample_dir, "flow_fields.msh")
    # if not os.path.exists(flow_fields_file):
    #     raise FileNotFoundError
    #
    # out_mesh = gmsh_io.GmshIO()
    # with open(flow_fields_file, "r") as f:
    #     out_mesh.read(f)
    # time_idx = 0
    # time, field_cs = out_mesh.element_data['cross_section'][time_idx]

    # ele_reg_vol = {eid: (tags[0] - 10000, element_volume(out_mesh, nodes))
    #                for eid, (tele, tags, nodes) in out_mesh.elements.items()}
    #
    # assert len(field_cs) == len(ele_reg_vol)
    # velocity_field = out_mesh.element_data['velocity_p0']

    # print("time ", time)
    # print("field_cs ", field_cs)

    # assert len(field_cs) == len(ele_reg_vol)
    # velocity_field = out_mesh.element_data['velocity_p0']
    # # print("velocity field ", velocity_field)

    loads = pressure_loads

    # print("loads ", loads)
    # group_idx = {group_id: i_group for i_group, group_id in enumerate(set(bulk_regions.values()))}
    # n_groups = len(group_idx)
    # print("n groups ", n_groups)
    # group_labels = n_groups * ['_']
    # for reg_id, group_id in bulk_regions.items():
    #     i_group = group_idx[group_id]
    #     old_label = group_labels[i_group]
    #     new_label = regions[reg_id].name
    #     group_labels[i_group] = old_label if len(old_label) > len(new_label) else new_label

    n_directions = len(loads)
    print("flux_response_shape ", flux_response_shape)
    flux_response = np.zeros(flux_response_shape)
    #area = np.zeros((*flux_response_shape[:2]))
    print("Averaging velocities ...")
    #print("velocity field ", velocity_field)

    for i_time, velocity in enumerate(time_velocity_bulk):
        for i in range(3):
            for j in range(3):
                print("velocity.shape ", velocity.shape)
                pixel_vel = velocity[upper_corner[0] + i, upper_corner[1] + j]
                print("pixel_vel ", pixel_vel)
                exit()

                velocities[eid] = ele_vel
                #reg_id, vol = ele_reg_vol[eid]
                #cs = field_cs[eid][0]
                # print("vol: {}, cs: {}".format(vol, cs))
                volume = 1 #cs * vol
                #i_group = group_idx[bulk_regions[reg_id]]
                #print("i group ", i_group)
                #print("i time ", i_time)
                flux_response[i_time, :] += -(volume * np.array(ele_vel[0:2]))
                #area[i_group, i_time] += volume


    velocities = {}
    for i_time, (time, velocity) in velocity_field.items():
        i_time = int(i_time)
        for eid, ele_vel in velocity.items():
            print("eid: {}, ele_vel:{}".format(eid, ele_vel))

            velocities[eid] = ele_vel
            reg_id, vol = ele_reg_vol[eid]
            cs = field_cs[eid][0]
            #print("vol: {}, cs: {}".format(vol, cs))
            volume = cs * vol
            i_group = group_idx[bulk_regions[reg_id]]
            print("i group ", i_group)
            print("i time ", i_time)
            flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:2]))
            area[i_group, i_time] += volume

    # print("flux response ", flux_response)
    # print("area ", area)
    #
    # flux_response /= area[:, :, None]
    # cond_tensors = {}
    # print("Fitting tensors ...")
    #
    # print("groud_idx ", group_idx.items())
    # for group_id, i_group in group_idx.items():
    #     flux = flux_response[i_group]
    #     print("flux ", flux)
    #     # least square fit for the symmetric conductivity tensor
    #     rhs = flux.flatten()
    #     # columns for the tensor values: C00, C01, C11
    #     pressure_matrix = np.zeros((len(rhs), 4))
    #     if sym_condition:
    #         pressure_matrix = np.zeros((len(rhs) + 1, 4))
    #
    #     for i_load, (p0, p1) in enumerate(loads):
    #         i0 = 2 * i_load
    #         i1 = i0 + 1
    #         # @TODO: skalarni soucin rychlosti s (p0, p1)
    #         # pressure_matrix[i0] = [p0, p1, 0]
    #         # pressure_matrix[i1] = [0, p0, p1]
    #         pressure_matrix[i0] = [p0, p1, 0, 0]
    #         pressure_matrix[i1] = [0, 0, p0, p1]
    #
    #     if sym_condition:
    #         pressure_matrix[-1] = [0, 1, -1, 0]
    #         rhs = np.append(rhs, [0])
    #
    #     print("pressure matrix ", pressure_matrix)
    #     print("rhs ", rhs)
    #     print("rhs type ", type(rhs))
    #     print("rhs ", rhs)
    #
    #     C, residuals, rank, sing_values = np.linalg.lstsq(pressure_matrix, rhs)
    #     #
    #     #
    #
    #     print("residuals ", residuals)
    #     print("rank ", rank)
    #     print("sing values ", sing_values)
    #     # exit()
    #
    #     # C = np.linalg.lstsq(pressure_matrix, rhs)[0]
    #     print("C ", C)
    #     print("(C[1]-C[2])/(C[1]+C[2]/2) ", np.abs((C[1] - C[2]) / ((C[1] + C[2] / 2))))
    #     # C[1] = C[1]
    #     # cond_tn = np.array([[C[0], C[1]], [C[1], C[2]]])
    #     cond_tn = np.array([[C[0], C[1]], [C[2], C[3]]])
    #     diff = np.abs((C[1] - C[2]) / ((C[1] + C[2] / 2)))
    #
    #     # exit()
    #
    #     e_val = None
    #     if residuals > 10 ** (-8):
    #         exit()
    #         e_val = FlowProblem.line_fit(flux)
    #
    #     if i_group < 10:
    #         # if flux.shape[0] < 5:
    #         print("Plot tensor for eid: ", group_id)
    #         print("Fluxes: \n", flux)
    #         print("pressures: \n", loads)
    #         print("cond: \n", cond_tn)
    #         # if e_val is None:
    #         e_val, e_vec = np.linalg.eigh(cond_tn)
    #         print("e val ", e_val)
    #
    #         # print("log mean eval cond", np.log10(np.mean(e_val)))
    #         #self.plot_effective_tensor(flux, cond_tn, self.basename + "_" + group_labels[i_group])
    #         # exit()
    #         # print(cond_tn)
    #     cond_tensors[group_id] = cond_tn
    #
    # print("self.cond_tensors ", cond_tensors)
    # print("self.flux ", flux)
    # print("self. pressure matrix ", pressure_matrix)
    #
    #
    # exit()


def create_input(sample_dir, n_pixels_x=256, feature_names=[['conductivity_tensor'], ['cross_section']], velocities=None):
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
        neg_pressure_elements_triangles = []
        velocity_elements_triangles = []
        cond_tn_elements_lines = []
        neg_pressure_elements_lines = []
        velocity_elements_lines = []
        #cs_triangles = []
        cs_lines = []
        skip_lower_tr_indices = [3, 6, 7]
        for _ in range(n_tn_elements):
            cond_tn_elements_triangles.append(dict(zip(triangles.keys(), np.zeros(len(triangles.keys())))))
            neg_pressure_elements_triangles.append(dict(zip(triangles.keys(), np.zeros(len(triangles.keys())))))
            velocity_elements_triangles.append(dict(zip(triangles.keys(), np.zeros(len(triangles.keys())))))
            cond_tn_elements_lines.append(dict(zip(lines.keys(), np.zeros(len(lines.keys())))))
            neg_pressure_elements_lines.append(dict(zip(lines.keys(), np.zeros(len(lines.keys())))))
            velocity_elements_lines.append(dict(zip(lines.keys(), np.zeros(len(lines.keys())))))
            cs_lines = dict()

        for feature_name in feature_names:
            feature_name = feature_name[0]
            if feature_name == "conductivity_tensor":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                    cond_tn = np.array(val).reshape(3, 3)
                    val = np.delete(val, skip_lower_tr_indices)
                    velocity = [0,0,0]
                    neg_pressure = [0,0,0]

                    if e_id in velocities:
                        velocity = velocities[e_id]
                        neg_pressure = np.matmul(cond_tn, velocities[e_id])
                    
                    if e_id in triangles:
                        for idx, v in enumerate(neg_pressure):
                            neg_pressure_elements_triangles[idx][e_id] = v
                        for idx, v in enumerate(velocity):
                            velocity_elements_triangles[idx][e_id] = v
                        for idx, v in enumerate(val):
                            cond_tn_elements_triangles[idx][e_id] = v

                    elif e_id in lines:
                        for idx, v in enumerate(val):
                            cond_tn_elements_lines[idx][e_id] = v
                        for idx, v in enumerate(neg_pressure):
                            neg_pressure_elements_lines[idx][e_id] = v
                        for idx, v in enumerate(velocity):
                            velocity_elements_lines[idx][e_id] = v

            elif feature_name == "cross_section":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                    # if e_id in triangles:
                    #     cs_triangles.setdefault(val[0], []).append(e_id)
                    if e_id in lines:
                        cs_lines.setdefault(val[0], []).append(e_id)

        cond_bulk_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))
        cond_fractures_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))

        velocity_bulk_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))
        velocity_fractures_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))

        neg_pressure_bulk_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))
        neg_pressure_fractures_data_array = np.empty((n_tn_elements, n_pixels_x, n_pixels_x))

        for k in range(n_tn_elements):
            ###
            # Conductivity
            ###
            trimesh, cvs_lines = rasterize(mesh_nodes, triangles, cond_tn_elements_triangles[k],
                      lines, cond_tn_elements_lines[k], cs_lines, n_pixels_x, save_image=True)

            cond_bulk_data_array[k] = np.flip(trimesh, axis=0)
            cond_fractures_data_array[k] = np.flip(cvs_lines, axis=0)

            ###
            # Velocity
            ###
            trimesh, cvs_lines = rasterize(mesh_nodes, triangles, velocity_elements_triangles[k],
                                           lines, velocity_elements_lines[k], cs_lines, n_pixels_x, save_image=True)

            velocity_bulk_data_array[k] = np.flip(trimesh, axis=0)
            velocity_fractures_data_array[k] = np.flip(cvs_lines, axis=0)

            ###
            # Pressure
            ###
            trimesh, cvs_lines = rasterize(mesh_nodes, triangles, neg_pressure_elements_triangles[k],
                                           lines, neg_pressure_elements_lines[k], cs_lines, n_pixels_x, save_image=True)


            velocity_bulk_data_array[k] = np.flip(trimesh, axis=0)
            velocity_fractures_data_array[k] = np.flip(cvs_lines, axis=0)



        np.savez_compressed(os.path.join(sample_dir, "bulk_{}".format(n_pixels_x)), data=cond_bulk_data_array)
        np.savez_compressed(os.path.join(sample_dir, "fractures_{}".format(n_pixels_x)), data=cond_fractures_data_array)

        return cond_bulk_data_array, velocity_bulk_data_array, neg_pressure_bulk_data_array

        # np.savez_compressed(os.path.join(sample_dir, "velocity_bulk_{}".format(n_pixels_x)), data=velocity_bulk_data_array)
        # np.savez_compressed(os.path.join(sample_dir, "velocity_fractures_{}".format(n_pixels_x)), data=velocity_fractures_data_array)
        #
        # np.savez_compressed(os.path.join(sample_dir, "neg_pressure_bulk_{}".format(n_pixels_x)),
        #                     data=neg_pressure_bulk_data_array)
        # np.savez_compressed(os.path.join(sample_dir, "neg_pressure_fractures_{}".format(n_pixels_x)),
        #                     data=neg_pressure_fractures_data_array)


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


def _prepare_input(sample_dir, n_pixels_x):
    velocities_dict, pressure_loads, flux_response_shape = read_flow_fields(sample_dir)
    time_cond = []
    time_velocity = []
    time_neg_pressure = []

    for i_time, velocities in velocities_dict.items():
        print("veloctities ", velocities)

        cond_bulk_data_array, velocity_bulk_data_array, neg_pressure_bulk_data_array = create_input(sample_dir,
                                                                                                    n_pixels_x=n_pixels_x,
                                                                                                    velocities=velocities)

        time_cond.append(cond_bulk_data_array)
        time_velocity.append(velocity_bulk_data_array)
        time_neg_pressure.append(neg_pressure_bulk_data_array)


    for i in range(0, n_pixels_x, 3):
        for j in range(0, n_pixels_x, 3):
            print("i: {} j: {}".format(i, j))



            _calculate_effective_tensor(time_cond, time_velocity, time_neg_pressure, pressure_loads, flux_response_shape, upper_corner=[i, j])
    print("velocities ", velocities)



if __name__ == "__main__":
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_no_fractures"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/nn_data/homogenization_samples_dfm"
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/nn_data/homogenization_samples_no_fractures"
    data_dir = "/home/martin/Documents/MLMC-DFM/test/nn_data/homogenization_samples_6_5_8_c_2"
    data_dir = "/home/martin/Documents/MLMC-DFM/test/nn_data/charon_samples_no_fractures/test_data"

    start_time = time.time()

    i = 0
    while True:
        sample_dir = os.path.join(data_dir, "sample_{}".format(i))
        if os.path.exists(sample_dir):
            if not os.path.exists(os.path.join(sample_dir, MESH_FILE)) \
                    or not os.path.exists(os.path.join(sample_dir, SUMMARY_FILE)):
                i += 1
                continue

            _prepare_input(sample_dir, n_pixels_x=256)
            create_output(sample_dir, symmetrize=True)
            i += 1

        else:
            break

    stop_time = time.time()
    print("total time: {}, time per sample: {}".format(stop_time - start_time, (stop_time-start_time)/(i+1)))


