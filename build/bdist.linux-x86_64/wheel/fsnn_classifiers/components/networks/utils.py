import sys  # for stderr
import os  # for mkdir
import numpy as np
import nest


def generate_random_state():
    return int.from_bytes(os.urandom(4), sys.byteorder)


def check_nest_objects(network_objects):
    """ Check that the list network_objects
    holds descriptors to existing NEST objects.

    fit() and transform() call this to skip creating the network anew
    for speedup. This checks that network_objects holds
    identifiers of existing NEST objects, which may be not the
    case if the estimator is unpickled, because NEST objects
    live separately from their identifiers in the python API
    and thus are not contained within the estimator object
    and do not survive pickling it.
    """
    nest_object_type = tuple
    if None in network_objects:
        return False
    # Some of the elements of network_objects
    # may be, say, nested dictionaries. Skip them for now.
    objects_to_check = [
        item
        for item in network_objects
        if type(item) is nest_object_type
    ]
    # Check that NEST actually holds an object for these descriptors.
    try:
        _ = nest.GetStatus(objects_to_check)
    except nest.lib.hl_api_exceptions.NESTError:
        return False
    return True


def convert_neuron_ids_to_indices(
    weights,
    connection_descriptors,
    pre_neuron_ids,
    post_neuron_ids,
    delays=None
):
    """
    Convert neurons' IDs in NEST
    to numbers starting with 0,
    numbering neurons with respect to their corresponding layer.
    Returns weights as a structured numpy array
    (pre_neuron_index, post_neuron_index, weight) if delay is None
    or (pre_neuron_index, post_neuron_index, weight, delay).
    """
    if delays is None:
        return_delays = False
        delays = [None] * len(weights)
    else:
        return_delays = True
    # Construct dictionaries that map
    # neurons' NEST IDs to 0-starting indices.
    pre_indices, post_indices = [
        dict(zip(
            nest_ids.tolist(), range(len(nest_ids))
        ))
        for nest_ids in (pre_neuron_ids, post_neuron_ids)
    ]
    weights_in_sparse_format = [
        # connection_descriptor[0] == pre_neuron_id
        # connection_descriptor[1] == post_neuron_id
        (
            pre_indices[pre_neuron_id],
            post_indices[post_neuron_id],
            w,
            delay
        )
        for w, delay, pre_neuron_id, post_neuron_id in zip(
            weights,
            delays,
            connection_descriptors.get('source'),
            connection_descriptors.get('target')
        )
    ]

    weights_in_sparse_format = np.array(
        weights_in_sparse_format,
        dtype=[
            ('pre_index', int),
            ('post_index', int),
            ('weight', float),
            ('delay', float)
        ]
    )
    if return_delays:
        return weights_in_sparse_format
    else:
        return weights_in_sparse_format[[
            'pre_index',
            'post_index',
            'weight',
        ]]


def convert_random_parameters_to_nest(parameters_dict):
    if is_distribution_description(parameters_dict):
        return nest.CreateParameter(**parameters_dict)
    return {
        par_name: (
            convert_random_parameters_to_nest(par_value)
            if isinstance(par_value, dict)
            else par_value
        )
        for par_name, par_value in parameters_dict.items()
    }

is_distribution_description = lambda par_value: (
    isinstance(par_value, dict)
    and 'parametertype' in par_value
    and 'specs' in par_value
)


def create_directory_if_inexistent(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print('Directory {d:s} already exists.'.format(d=dirname), file=sys.stderr)
