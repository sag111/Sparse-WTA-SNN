from copy import deepcopy

from sklearn.preprocessing import minmax_scale

from tqdm import tqdm
import numpy as np
import os

import os
os.environ["PYNEST_QUIET"] = "1"

import nest

from fsnn_classifiers.components.networks.base_spiking_transformer import BaseSpikingTransformer
from fsnn_classifiers.components.networks.utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
    convert_random_parameters_to_nest
)
from fsnn_classifiers.components.networks.common_model_components import disable_plasticity

from fsnn_classifiers.components.networks.configs.diehl_network import snn_parameters, network_objects_tuple


class DiehlNetwork(BaseSpikingTransformer):
    def __init__(
        self,
        synapse_model="stdp_nn_symm_synapse",
        V_th=(-50.0, -40.0),  # (exc_neurons, inh_neurons)
        t_ref=(5.0, 2.0),  # (exc_neurons, inh_neurons)
        tau_m=(100.0, 10.0),  # (exc_neurons, inh_neurons)
        w_syn=(10.4, -17.0),  # (exc_to_inh, inh_to_exc)
        n_neurons=(100, 100),  # (exc_neurons, inh_neurons)
        p=(None, None, None),  # sparcity prob (gen-exc, exc-inh, inh-exc)
        r=(None, None, None),  # sparcity radius (gen-exc, exc-inh, inh-exc)
        spatial="free",
        alpha=0.55,
        nu=0.01,
        high_rate=43.379,
        low_rate=0.018,
        epochs=1,
        time=350,
        random_state=None,
        n_jobs=1,
        warm_start=False,
        quiet=True,
        **kwargs,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.quiet = quiet

        # have to create these for sklearn
        self.V_th = V_th
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.w_syn = w_syn
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.nu = nu
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.time = time
        self.epochs = epochs
        self.synapse_model = synapse_model

        # sparcity
        self.p = p
        self.r = r
        self.spatial = spatial

        self._check_modules(quiet)
        _parameters = self._get_initial_parameters()

        ## set base arguments
        # neuron parameters
        for key, val in zip(["V_th", "t_ref", "tau_m"], [V_th, t_ref, tau_m]):
            if not hasattr(val, "__len__"):
                p = (val, val)
            else:
                p = val
            _parameters["neuron_parameters"]["exc_neurons"][key] = p[0]
            _parameters["neuron_parameters"]["inh_neurons"][key] = p[1]
        # synapse parameters
        if not hasattr(w_syn, "__len__"):
            p = (w_syn, w_syn)
        else:
            p = w_syn
        _parameters["synapse_parameters"]["exc_to_inh"]["weight"] = p[0]
        _parameters["synapse_parameters"]["inh_to_exc"]["weight"] = p[1]
        _parameters["synapse_parameters"]["input_to_exc"][
            "synapse_model"
        ] = synapse_model
        if synapse_model != "stdp_nn_symm_synapse":
            # remove extra parameters for memristive synapse models
            _parameters["synapse_parameters"]["input_to_exc"].pop("Wmax", None)
            _parameters["synapse_parameters"]["input_to_exc"].pop("alpha", None)
            _parameters["synapse_parameters"]["input_to_exc"].pop("lambda", None)
            _parameters["synapse_parameters"]["input_to_exc"].pop("mu_plus", None)
            _parameters["synapse_parameters"]["input_to_exc"].pop("mu_minus", None)
            _parameters["synapse_parameters"]["input_to_exc"].pop("tau_plus", None)
        else:
            _parameters["synapse_parameters"]["input_to_exc"]["alpha"] = alpha
            _parameters["synapse_parameters"]["input_to_exc"]["lambda"] = nu
        # set network parameters
        _parameters["network_parameters"]["high_rate"] = high_rate
        _parameters["network_parameters"]["low_rate"] = low_rate
        _parameters["network_parameters"]["epochs"] = int(epochs)
        _parameters["network_parameters"]["number_of_exc_neurons"] = int(n_neurons[0])
        _parameters["network_parameters"]["number_of_inh_neurons"] = int(n_neurons[1])
        _parameters["network_parameters"]["one_vector_longtitude"] = int(time)

        ## set optional keyword arguments
        # must be in format parameter_type+key1+key2+...
        parameter_types = ["network", "neuron", "synapse"]
        for par_str, val in kwargs.items():
            location = par_str.split("+")
            assert location[0] in parameter_types, "{par_str} -- wrong format!"
            assert len(location) in [
                2,
                3,
            ], f"Wrong number of parameter keys: {len(location)}!"
            if len(location) == 2:
                _parameters[f"{location[0]}_parameters"][location[1]] = val
            else:
                _parameters[f"{location[0]}_parameters"][location[1]][location[2]] = val

        # initialize parameters
        self.network_parameters = _parameters["network_parameters"]
        self.neuron_parameters = _parameters["neuron_parameters"]
        self.synapse_parameters = _parameters["synapse_parameters"]

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        n_threads = self.n_jobs
        random_state = (
            self.random_state
            if not self.random_state is None
            else generate_random_state()
        )
        create_spike_recorders = testing_mode

        # Make a copy because we will tamper with neuron_parameters
        # if creating neurons with pre-recorded thresholds.
        # Also, run nest.CreateParameter on those parameters
        # that are dictionaries describing random distributions.
        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (self.neuron_parameters, self.synapse_parameters),
        )

        if testing_mode:
            # Disable dynamic threshold.
            neuron_parameters["exc_neurons"]["Theta_plus"] = 0.0
            # Disable synaptic plasticity.
            synapse_parameters["input_to_exc"] = disable_plasticity(
                synapse_parameters["input_to_exc"]
            )
        if hasattr(self, "exc_neurons_thresholds_"):
            # Set the recorded dynamic threshold values.
            neuron_parameters["exc_neurons"] = [
                # because, unlike synapse parameters, NEST only accepts
                # setting varying neuron parameters
                # by passing a dictionary per each neuron.
                dict(neuron_parameters["exc_neurons"], V_th=V_th)
                for V_th in exc_neurons_thresholds
            ]

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                "resolution": 0.1,
                "local_num_threads": n_threads,
            }
        )
        nest.rng_seed = random_state

        inputs_ids = nest.Create("parrot_neuron", number_of_inputs)
        generators_ids = nest.Create("poisson_generator", number_of_inputs)

        if any([x is not None for x in self.p]):
            if self.spatial == "grid":
                pos = nest.spatial.grid(
                    shape=[10, 10],  # the number of rows and column in this grid ...
                    extent=[1.0, 1.0],
                )  # the size of the grid in mm

                # Create nodes.
                exc_neuron_ids = nest.Create(
                    "iaf_cond_exp_adaptive",
                    params=neuron_parameters["exc_neurons"],
                    positions=pos,
                )
                inh_neuron_ids = nest.Create(
                    "iaf_cond_exp_adaptive",
                    params=neuron_parameters["inh_neurons"],
                    positions=pos,
                )

                inputs_ids = nest.Create("parrot_neuron", number_of_inputs)

                generators_ids = nest.Create("poisson_generator", number_of_inputs)
            elif self.spatial == "free":
                pos = nest.spatial.free(
                    nest.random.uniform(
                        min=-0.5, max=0.5
                    ),  # using random positions in a uniform distribution
                    num_dimensions=2,
                )  # have to specify number of dimensions

                # Create nodes.
                exc_neuron_ids = nest.Create(
                    "iaf_cond_exp_adaptive",  # ADAPTIVE IN ORIGINAL DIEHL
                    self.network_parameters["number_of_exc_neurons"],
                    params=neuron_parameters["exc_neurons"],
                    positions=pos,
                )
                inh_neuron_ids = nest.Create(
                    "iaf_cond_exp_adaptive",
                    self.network_parameters["number_of_inh_neurons"],
                    params=neuron_parameters["inh_neurons"],
                    positions=pos,
                )
            else:
                raise NotImplementedError(
                    f"Spatial connections of the type {self.spatial} are not implemented."
                )
        else:
            # Create nodes.
            exc_neuron_ids = nest.Create(
                "iaf_cond_exp_adaptive",  # ADAPTIVE IN ORIGINAL DIEHL
                self.network_parameters["number_of_exc_neurons"],
                params=neuron_parameters["exc_neurons"],
            )
            inh_neuron_ids = nest.Create(
                "iaf_cond_exp_adaptive",
                self.network_parameters["number_of_inh_neurons"],
                params=neuron_parameters["inh_neurons"],
            )

        if create_spike_recorders:
            exc_neurons_spike_recorder_id = nest.Create("spike_recorder")
            inh_neurons_spike_recorder_id = nest.Create("spike_recorder")

        populations_to_connect = [
            ("input_to_exc", inputs_ids, exc_neuron_ids),
            ("input_to_inh", inputs_ids, inh_neuron_ids),
            ("exc_to_inh", exc_neuron_ids, inh_neuron_ids),
            ("inh_to_exc", inh_neuron_ids, exc_neuron_ids),
        ]

        # Create connections.
        # ------------------
        # Static synapses from the generators
        # to parrot neurons representing the input.
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec="one_to_one",
            syn_spec="static_synapse",
        )
        # Connections from the input-representing parrot neurons
        # to excitatory and inhibitory neurons.
        if hasattr(self, "weights_"):
            # Re-create connections from the saved weights.
            for exc_or_inh, exc_or_inh_neuron_ids in (
                ("exc", exc_neuron_ids),
                ("inh", inh_neuron_ids),
            ):
                conn_type_name = "input_to_" + exc_or_inh
                sparse_weight_array = self.weights_[conn_type_name]

                synapse_parameters[conn_type_name].update(
                    weight=sparse_weight_array["weight"],
                    delay=sparse_weight_array["delay"],
                )
                nest.Connect(
                    pre=np.array(inputs_ids)[sparse_weight_array["pre_index"]],
                    post=np.array(exc_neuron_ids)[sparse_weight_array["post_index"]],
                    conn_spec="one_to_one",
                    syn_spec=synapse_parameters[conn_type_name],
                )
        else:
            # Plastic synapses from the input parrot neurons
            # to the excitatory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=exc_neuron_ids,
                conn_spec=self._get_connection_spec("all_to_all", self.p[0], self.r[0]),
                syn_spec=synapse_parameters["input_to_exc"],
            )
            # Static synapses from the input parrot neurons
            # to the inhibitory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=inh_neuron_ids,
                conn_spec={
                    "rule": "fixed_total_number",
                    "N": int(
                        self.network_parameters["input_to_inh_connection_prob"]
                        * len(inputs_ids)
                        * len(inh_neuron_ids)
                    ),
                },
                syn_spec=synapse_parameters["input_to_inh"],
            )
        # Static connections from excitatory neurons
        # to their inhibitory counterparts.
        nest.Connect(
            pre=exc_neuron_ids,
            post=inh_neuron_ids,
            conn_spec=self._get_connection_spec("one_to_one", self.p[1], self.r[1]),
            syn_spec=synapse_parameters["exc_to_inh"],
        )
        # Static connections from inhibitory neurons
        # to excitatory ones.
        if self.p[2] is not None:
            nest.Connect(
                pre=inh_neuron_ids,
                post=exc_neuron_ids,
                conn_spec=self._get_connection_spec("all_to_all", self.p[2], self.r[2]),
                syn_spec=synapse_parameters["inh_to_exc"],
            )
        else:
            for current_neuron_number in range(
                self.network_parameters["number_of_inh_neurons"]
            ):
                nest.Connect(
                    pre=inh_neuron_ids[
                        current_neuron_number : current_neuron_number + 1
                    ],
                    post=nest.NodeCollection(
                        list(sorted(list(
                            set(exc_neuron_ids.tolist())
                            - set(
                                exc_neuron_ids[
                                    current_neuron_number : current_neuron_number + 1
                                ].tolist()
                            )
                        )))
                    ),
                    conn_spec="all_to_all",
                    syn_spec=synapse_parameters["inh_to_exc"],
                )

        # Connect neurons to spike detectors.
        if create_spike_recorders:
            nest.Connect(
                exc_neuron_ids, exc_neurons_spike_recorder_id, conn_spec="all_to_all"
            )
            nest.Connect(
                inh_neuron_ids, inh_neurons_spike_recorder_id, conn_spec="all_to_all"
            )

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = {
            conn_type_name: nest.GetConnections(source=pre_ids, target=post_ids)
            for conn_type_name, pre_ids, post_ids in populations_to_connect
        }

        self.network_objects = network_objects_tuple(
            exc_neuron_ids=exc_neuron_ids,
            inh_neuron_ids=inh_neuron_ids,
            generators_ids=generators_ids,
            inputs_ids=inputs_ids,
            all_connection_descriptors=all_connection_descriptors,
            exc_neurons_spike_recorder_id=exc_neurons_spike_recorder_id
            if create_spike_recorders
            else None,
            inh_neurons_spike_recorder_id=inh_neurons_spike_recorder_id
            if create_spike_recorders
            else None,
        )

    def _to_spike_rates(self, X):
        if len(np.ravel(X[X < 0])) > 0:
            # a failsafe in case X can be negative (i.e. if we do not use GRF)
            return minmax_scale(X) * (self.high_rate - self.low_rate) + self.low_rate
        else:
            # note that in this case X will not necessarily range from low_rate to high_rate
            return X * (self.high_rate - self.low_rate) + self.low_rate

    def _get_initial_parameters(self):
        return deepcopy(snn_parameters)

    def _get_connection_spec(self, def_conn, p, r):
        if p == None:
            return def_conn
        else:
            if r == None:
                conn = {
                    "rule": "pairwise_bernoulli",
                    "p": p,
                }
            else:
                conn = {
                    "rule": "pairwise_bernoulli",
                    "p": p,
                    "mask": {"circular": {"radius": r}},
                }

            return conn

    def run_the_simulation(self, X, y_train):
        """Encode X into input spiking rates
        and feed to the network.

        Learning is unsupervised: y_train is only used to indicate
        testing if y_train is None and training otherwise.
        If testing:
        * The simulation duration is fixed at 1 epoch.
        * STDP is disabled.
        * Re-normalization of weights is not applied even if enabled.
        * Output spiking rates are recorded and returned.
        If training:
        * Weights are saved to self.weights_.
        * Nothing is returned.
        """
        testing_mode = y_train is None
        n_epochs = self.network_parameters["epochs"] if not testing_mode else 1
        input_spike_rates = self._to_spike_rates(X)
        record_weights = not testing_mode
        record_spikes = testing_mode

        progress_bar = tqdm(
            total=n_epochs * len(input_spike_rates),
            disable=self.quiet,
        )
        for epoch in range(n_epochs):
            if record_spikes:
                output_spiking_rates = {"exc_neurons": [], "inh_neurons": []}
            for x in input_spike_rates:
                # Weight normalization
                if (
                    not testing_mode
                    and not self.network_parameters[
                        "weight_normalization_during_training"
                    ]
                    is None
                ):
                    for neuron_id in self.network_objects.exc_neuron_ids:
                        this_neuron_input_synapses = nest.GetConnections(
                            source=self.network_objects.inputs_ids, target=[neuron_id]
                        )
                        w = nest.GetStatus(this_neuron_input_synapses, "weight")
                        w = (
                            np.array(w)
                            * self.network_parameters[
                                "weight_normalization_during_training"
                            ]
                            / sum(w)
                        )
                        nest.SetStatus(this_neuron_input_synapses, "weight", w)

                # The simulation itself.
                nest.SetStatus(
                    self.network_objects.generators_ids, [{"rate": r} for r in x]
                )
                nest.Simulate(self.network_parameters["one_vector_longtitude"])

                nest.SetStatus(self.network_objects.generators_ids, {"rate": 0.0})
                nest.Simulate(self.network_parameters["intervector_pause"])

                if record_spikes:
                    for neuron_type_name, neurons_ids, spike_recorder_id in (
                        (
                            "exc_neurons",
                            self.network_objects.exc_neuron_ids,
                            self.network_objects.exc_neurons_spike_recorder_id,
                        ),
                        (
                            "inh_neurons",
                            self.network_objects.inh_neuron_ids,
                            self.network_objects.inh_neurons_spike_recorder_id,
                        ),
                    ):
                        # NEST returns all_spikes == {
                        #   'times': spike_times_array,
                        #   'senders': senders_ids_array
                        # }
                        all_spikes = nest.GetStatus(spike_recorder_id, keys="events")[0]
                        current_input_vector_output_rates = [
                            # 1000.0 * len(all_spikes['times'][
                            #   all_spikes['senders'] == current_neuron
                            # ]) / self.network_parameters['one_vector_longtitude']
                            len(
                                all_spikes["times"][
                                    all_spikes["senders"] == current_neuron
                                ]
                            )
                            for current_neuron in neurons_ids
                        ]
                        output_spiking_rates[neuron_type_name].append(
                            current_input_vector_output_rates
                        )
                        # Empty the detector.
                        nest.SetStatus(spike_recorder_id, {"n_events": 0})
                progress_bar.update()
        progress_bar.close()
        if record_weights:
            exc_neuron_ids = self.network_objects.exc_neuron_ids
            inh_neuron_ids = self.network_objects.inh_neuron_ids
            generators_ids = self.network_objects.generators_ids
            inputs_ids = self.network_objects.inputs_ids
            all_connection_descriptors = self.network_objects.all_connection_descriptors

            weights_of_all_connection_types = {
                conn_type_name: convert_neuron_ids_to_indices(
                    weights=nest.GetStatus(
                        all_connection_descriptors[conn_type_name], "weight"
                    ),
                    delays=nest.GetStatus(
                        all_connection_descriptors[conn_type_name], "delay"
                    ),
                    connection_descriptors=all_connection_descriptors[conn_type_name],
                    pre_neuron_ids=pre_ids,
                    post_neuron_ids=post_ids,
                )
                for conn_type_name, pre_ids, post_ids in (
                    ("input_to_exc", inputs_ids, exc_neuron_ids),
                    ("input_to_inh", inputs_ids, inh_neuron_ids),
                    ("exc_to_inh", exc_neuron_ids, inh_neuron_ids),
                    ("inh_to_exc", inh_neuron_ids, exc_neuron_ids),
                )
            }
            self.weights_ = weights_of_all_connection_types

        if record_spikes:
            return output_spiking_rates["exc_neurons"]


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(as_frame=False, return_X_y=True)

    network = DiehlNetwork(
        quiet=False, p=(0.1, 0.6, 0.5), r=(None, 0.3, 0.5), spatial="free"
    )

    # test all main methods
    network.fit(X, y)

    X_ = network.transform(X)
    X_ = network.fit_transform(X, y)
