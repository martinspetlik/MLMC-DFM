import os
import os.path
import numpy as np
import yaml
import gmsh_io
from rasterization import rasterize
from matplotlib import pyplot as plt

MESH_FILE = "mesh_fine.msh"
FIELDS_MESH_FILE = "fields_fine.msh"


def get_output(data_dir):
    with open(os.path.join(data_dir, "summary.yaml"), "r") as f:
        summary_dict = yaml.load(f, Loader=yaml.Loader)
    cond_tn = np.array(summary_dict['fine']['cond_tn'])
    return cond_tn[0]


def image_creator(data_dir, feature_names=[['conductivity_tensor'], ['cross_section']], quantity_name="conductivity"):
    print("image creator ")
    mesh = os.path.join(data_dir, MESH_FILE)

    print("mesh ", mesh)
    mesh_nodes, triangles, lines, ele_ids = extract_mesh_gmsh_io(mesh, image=True)

    print("lines ", lines)

    # for key, (node_0, node_1) in lines.items():
    #     print("key ", key)
    #     print("node_0 id: {} node_1 id: {} ".format(node_0, node_1))
    #     print("node_0: {} node_1 id: {} ".format(mesh_nodes[node_0], mesh_nodes[node_1]))
    #
    #
    #     max_key = np.max(list(mesh_nodes.keys()))
    #
    #     mesh_nodes[max_key + 1] = []
    #
    #     print("max_key ", max_key)
    #     exit()

        #mesh_nodes[nodes]

    # sample_storage = SampleStorageHDF(file_path=hdf_path)
    # sample_storage.chunk_size = 1e8
    # result_format = sample_storage.load_result_format()
    # root_quantity = make_root_quantity(sample_storage, result_format)
    #
    # conductivity = root_quantity[quantity_name]
    # time = conductivity[1]  # times: [1]
    # location = time['0']  # locations: ['0']
    # q_value = location[0, 0]

    # hdf = HDF5(file_path=hdf_path, load_from_file=True)
    # level_group = hdf.add_level_group(level_id=str(level))
    #
    # chunk_spec = next(sample_storage.chunks(level_id=level, n_samples=sample_storage.get_n_collected()[int(level)]))
    # collected_values = q_value.samples(chunk_spec=chunk_spec)[0]
    #
    # collected_ids = sample_storage.collected_ids(level_id=level)
    #
    # indices = np.ones(len(collected_values))
    # collected = zip(collected_ids, collected_values)

    #graphs = []
    #data = []
    #i = 0
    #while True:
    n_tn_elements = 9

    field_mesh = os.path.join(data_dir, FIELDS_MESH_FILE)
    if os.path.exists(field_mesh):
        # i += 1
        # if i > 2:
        #     break
        features, all_fields = get_node_features(field_mesh, feature_names)

        print("all fields ", all_fields)
        # print("all_fields cross_section", all_fields["cross_section"])
        # print("all_fields cross_section", all_fields[1]["cross_section"])


        cond_tn_elements_triangles = []
        cond_tn_elements_lines = []
        cs_lines = []
        for _ in range(n_tn_elements):
            cond_tn_elements_triangles.append(dict(zip(triangles.keys(), np.zeros(len(triangles.keys())))))
            cond_tn_elements_lines.append(dict(zip(lines.keys(), np.zeros(len(lines.keys())))))
            cs_lines = dict()

        for feature_name in feature_names:
            feature_name = feature_name[0]
            if feature_name == "conductivity_tensor":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                        if e_id in triangles:
                            #print("e_id: {} in triangles ".format(e_id))
                            for idx, v in enumerate(val):
                                cond_tn_elements_triangles[idx][e_id] = v

                        elif e_id in lines:
                            #print("e_id: {} in lines ".format(e_id))
                            for idx, v in enumerate(val):
                                cond_tn_elements_lines[idx][e_id] = v

            elif feature_name == "cross_section":
                for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                    if e_id in lines:
                        cs_lines.setdefault(val[0], []).append(e_id)

        print("cs lines ", cs_lines)

        trimesh, cvs_lines = rasterize(mesh_nodes, triangles, cond_tn_elements_triangles[0],
                  lines, cond_tn_elements_lines[0], cs_lines, save_image=True)

        cvs_lines = np.flip(cvs_lines, axis=0)
        trimesh = np.flip(trimesh, axis=0)

        np.save(os.path.join(sample_dir, "trimesh_bypixel_256"), trimesh)
        np.save(os.path.join(sample_dir, "lines_bypixel_256"), cvs_lines)

        trimes_values = np.load(os.path.join(sample_dir, "trimesh_bypixel_256.npy"))
        lines_values = np.load(os.path.join(sample_dir, "lines_bypixel_256.npy"))

        trimes_values = np.array(trimes_values.data)
        #print("trimes_values ", trimes_values)

        flatten_trimes = trimes_values.reshape(-1)
        flatten_lines_values = lines_values.reshape(-1)

        not_nan_indices = np.argwhere(~np.isnan(flatten_lines_values))
        flatten_trimes[not_nan_indices] = flatten_lines_values[not_nan_indices]
        trimes_values = flatten_trimes.reshape((int(np.sqrt(len(flatten_trimes))),
                                                int(np.sqrt(len(flatten_trimes)))))


        from PIL import Image as im

        print("trimes values ", trimes_values)


        # img_data = im.fromarray(trimes_values)
        # img_data.save('dfm_model_from_array.png')

        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True

        #arr = np.random.rand(5, 5)
        plt.gray()
        plt.imshow(trimes_values)

        plt.show()

        plt.gray()
        plt.imshow(lines_values)

        plt.show()



        # print("np.isnan ", np.isnan(lines_values))
        # exit()
        #
        # print("trimes values ", trimes_values)
        # print("lines values ", lines_values)
        # print("lines values shape ", lines_values.shape)
        # for i in range(lines_values.shape[0]):
        #     for j in range(lines_values.shape[1]):
        #         if not np.isnan(lines_values[i][j]):
        #             print("lines_values[i][j] ", lines_values[i][j])
        # exit()

        output_value = get_output(data_dir)
        print("output value ", output_value)

        #np.save(os.path.join(sample_dir, "image"), rgb_image)
        np.save(os.path.join(sample_dir, "output"), output_value)


def join_fields(fields, f_names):
    if len(f_names) > 0:
        x_name = len(set([*fields[f_names[0]]]))
    assert all(x_name == len(set([*fields[f_n]])) for f_n in f_names)

    # # Using defaultdict
    # c = [collections.Counter(fields[f_n]) for f_n in f_names]
    # Cdict = collections.defaultdict(int)

    joint_dict = {}
    for f_n in f_names:
        for key, item in fields[f_n].items():
            #print("key: {}, item: {}".format(key, np.squeeze(item)))
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
    all_joint_features = []
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
        # TODO: be able to use this mesh_dimension in fields
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}

    if image:
        return mesh.nodes, triangles, lines, ele_ids
    return ele_nodes


if __name__ == "__main__":
    #data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_no_fractures"
    data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_fractures"

    i = 1
    while True:
        sample_dir = os.path.join(data_dir, "sample_{}".format(i))
        if os.path.exists(sample_dir):
            image_creator(sample_dir)
            i += 1
        else:
            break