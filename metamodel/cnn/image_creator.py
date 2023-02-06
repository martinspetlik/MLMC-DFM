import os
import os.path
import numpy as np
import yaml
import gmsh_io
from rasterization import rasterize

MESH_FILE = "mesh_fine.msh"
FIELDS_MESH_FILE = "fields_fine.msh"


def get_output(data_dir):
    with open(os.path.join(data_dir, "summary.yaml"), "r") as f:
        summary_dict = yaml.load(f, Loader=yaml.Loader)
    cond_tn = np.array(summary_dict['fine']['cond_tn'])
    return cond_tn[0]


def image_creator(data_dir, feature_names=[['conductivity_tensor']], quantity_name="conductivity"):
    print("image creator ")
    mesh = os.path.join(data_dir, MESH_FILE)

    print("mesh ", mesh)

    mesh_nodes, triangles, ele_ids = extract_mesh_gmsh_io(mesh, image=True)


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

        print("features ", features)
        print("all fields ", all_fields[0])

        cond_tn_elements = []
        for _ in range(n_tn_elements):
            cond_tn_elements.append(dict(zip(ele_ids, np.zeros(len(ele_ids)))))

        for feature_name in feature_names[0]:
            if len(all_fields) > 1:
                raise NotImplementedError

            for e_id, val in zip(ele_ids, list(all_fields[0][feature_name].values())):
                for idx, v in enumerate(val):
                    cond_tn_elements[idx][e_id] = v

        rasterize(mesh_nodes, triangles, cond_tn_elements[0])
        # rasterize_values = []
        # for e in range(n_tn_elements):
        #     rasterize_values.append(rasterize(mesh_nodes, triangles, cond_tn_elements[e]))
        #
        # rasterize_values = np.array(rasterize_values)
        # print("rasterize_values.shape ", rasterize_values.shape)
        # np.savez_compressed(os.path.join(sample_dir, "bypixel_512"), a=rasterize_values)


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
    print("fields mesh ", fields_mesh)
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
        return mesh.nodes, triangles, ele_ids

    return ele_nodes


if __name__ == "__main__":
    data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_no_fractures_constant_field"

    i = 0
    while True:
        sample_dir = os.path.join(data_dir, "sample_{}".format(i))
        if os.path.exists(sample_dir):
            image_creator(sample_dir)
            i += 1
        else:
            break