import pickle
import numpy as np
import matplotlib.pyplot as plt

for vector_num in range(10):
    with open(f"check_times_test_{vector_num}.pkl", 'rb') as fp:
        inp_time_list, teacher_list, sample_time = pickle.load(fp)

    resolution = 0.1
    if sample_time > 0:
        max_step = np.ceil(sample_time / vector_num / resolution).astype(np.int32)
    else:
        max_step = 188405

    inp_spike_times = [np.array(time_dict["spike_times"]) - sample_time for time_dict in inp_time_list if len(time_dict["spike_times"]) > 0]
    inp_spike_times = np.hstack(inp_spike_times)
    inp_spike_times.sort()
    inp_spike_times = (inp_spike_times * 10).astype(np.int32)

    dt = 2.0

    teacher_spike_times_all = [np.array(time_dict["spike_times"]) - sample_time - dt for time_dict in teacher_list if len(time_dict["spike_times"]) > 0]

    for teacher_spike_times in teacher_spike_times_all:

        teacher_spike_times = (teacher_spike_times * 10).astype(np.int32)
        teacher_spike_times = teacher_spike_times[teacher_spike_times < max_step]

        in_spikes = np.zeros(max_step)
        out_spikes = np.zeros(max_step)

        in_spikes[inp_spike_times] = 1
        out_spikes[teacher_spike_times] = 1

        print(f"Input spikes: {np.sum(in_spikes)}")
        print(f"Output spikes: {np.sum(out_spikes)}")
        print(f"Matching spikes: {np.sum(in_spikes * out_spikes)}")

        p1 = 0
        p2 = 500
        in_spikes = in_spikes[p1:p2]
        out_spikes = out_spikes[p1:p2]

        fig, ax = plt.subplots(3)
        ax[0].bar(range(len(in_spikes)), in_spikes, color='blue')
        ax[1].bar(range(len(in_spikes)), out_spikes, color='red')
        ax[2].bar(range(len(in_spikes)), in_spikes*out_spikes, color='green')
        plt.show()

        break