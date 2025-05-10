import numpy as np
import time


def generate_hierarchical_sequence(inp_vector, ref_seq, N):
    """
    Parameters:
        inp_vector (numpy.ndarray): Входной вектор данных.
        ref_seq (numpy.ndarray): Эталонная последовательность спайков.
        N (int): Число временных шагов.
    Returns:
        numpy.ndarray: Массив последовательностей спайков.
    """

    # На всякий случай преобразуем S0 в плоский массив
    ref_seq = np.ravel(ref_seq)

    # Инициализация массива со спайковыми последовательностями
    out = np.zeros((len(inp_vector), N), dtype=bool)

    # Позиции спайков в S0
    spike_idxs = np.nonzero(ref_seq)[0]
    # Перемешаем их
    np.random.seed(int(time.time()))
    np.random.shuffle(spike_idxs)

    # Сортировка входного вектора с сохранением индексов
    sorted_idxs = np.argsort(inp_vector)
    sorted_inp_vector = inp_vector[sorted_idxs]

    # Количество спайков на элемент (с округлением)
    spikes_per_element = (sorted_inp_vector * ref_seq.sum()).astype(int)

    # Заполнение выходного массива последовательностей спайков
    for i, num_spikes in enumerate(spikes_per_element):

        # Ненулевые элементы спайки не получат
        if sorted_inp_vector[i] == 0:
            continue

        # Индекс входа до сортировки
        orig_idx = sorted_idxs[i]

        # Количество спайков для текущего входа
        current_spike_idxs = spike_idxs[:num_spikes]
        # Запись в выходной массив и обновление маски
        out[orig_idx, current_spike_idxs] = 1

    return out


# def generate_correlated_sequence(inp_vector, ref_seq, spike_p, N, correct_encoding):
#      ref_seq = np.ravel(ref_seq)
#      out = np.zeros((len(inp_vector), N))
#      # позиции спайков в S0
#      one_idxs = np.nonzero(ref_seq)[0]

#      for i, feat in enumerate(inp_vector):
#           num_spikes_to_keep = np.floor(feat * np.sum(ref_seq)).astype(np.int32)
#           np.random.shuffle(one_idxs)
#           cur_one_idxs = one_idxs[:num_spikes_to_keep]
#           out[i, cur_one_idxs] = ref_seq[cur_one_idxs]

#      return out

def spikes_to_times(inp_spikes, sim_time, tau_s, resolution = 0.1):
     spike_times = (np.arange(0, sim_time, resolution).reshape((1,-1)) + resolution + tau_s).round(1)
     spike_times = np.repeat(spike_times, inp_spikes.shape[0], axis=0) # number of neurons
     spike_times = spike_times * inp_spikes
     return spike_times

def get_time_dict(spike_times):
     return [{"spike_times":list(st_i[st_i>0])} for st_i in spike_times]
     

class CorrelationEncoder(object):
     
     def __init__(self, rate, tau_s, sim_time, resolution, interval, use_poisson=False):
          self.rate = rate
          self.sim_time = sim_time
          self.resolution = resolution
          self.interval = interval
          self.tau_s = tau_s
          self.use_poisson = use_poisson

          self.N = int(sim_time /resolution)

          # random_state = int(time.time())
          # np.random.seed(random_state)

          def generate_reference_sequence(rate, N, interval: int = 5, resolution=0.1):
               
               if self.use_poisson:
                    S0 = np.random.poisson(1 / interval, N)
                    S0 = np.clip(S0, 0, 1)

                    mask_zeros = np.where(S0 == 0)[0]
                    mask_ones = np.where(S0 == 1)[0]
                    delta_spike_counts = mask_ones.size - int(N / interval)

                    if delta_spike_counts < 0:
                         np.random.shuffle(mask_zeros)
                         missing_spikes = mask_zeros[:abs(delta_spike_counts)]
                         S0[missing_spikes] = 1

                    elif delta_spike_counts > 0:
                         np.random.shuffle(mask_ones)
                         excessive_spikes = mask_ones[:delta_spike_counts]
                         S0[excessive_spikes] = 0
                    
                    spike_p = (rate*resolution/1000.) * np.exp(-rate*resolution/1000.)
                    return S0.reshape((1,-1)), spike_p

               else:
                    S0 = np.zeros(N, dtype=np.uint8)
                    for i in range(N):
                         # if sum(S0)/N > rate*resolution/1000.:
                         #      break
                         if i % interval == 0:
                              S0[i] = 1
                    
                    spike_p = (rate*resolution/1000.) * np.exp(-rate*resolution/1000.)
                    return S0.reshape((1,-1)), spike_p                    


          self.S0, self.spike_p = generate_reference_sequence(
               self.rate,
               self.N,
               self.interval,
               self.resolution
          )

          self.ref_times = spikes_to_times(self.S0, self.sim_time, self.tau_s, self.resolution)
          self.ref_times = self.ref_times[self.ref_times>0]

     def __call__(self, X: np.ndarray):
          
          X_s = np.empty((*X.shape, self.N), dtype=np.uint8)
          for i, inp_vector in enumerate(X):
               X_s[i] = generate_hierarchical_sequence(inp_vector,
                                                     self.S0,
                                                     self.N,
                                                     )
          return X_s 
               

def debug():
    import matplotlib.pyplot as plt

    sim_time = 200
    resolution = 0.1
    rate = int(sim_time/resolution)
    N = int(sim_time/resolution)
    S0, spike_p = generate_reference_sequence(rate, N)
    print(S0[0][:100])

    freq = sum(S0[0]) * 10000 / int(sim_time/resolution)
    print(freq, spike_p)

    feature = np.array([0.1, 0.3, 0.7, 0.9])

    S1 = generate_correlated_sequence(feature, S0, spike_p, N)


    fig, ax = plt.subplots(1+len(S1))
    ax[0].bar(x=np.arange(0,int(sim_time/resolution),1), height=S0[0], width=1)
    for i, s_ in enumerate(S1):
        ax[i+1].bar(x=np.arange(0,int(sim_time/resolution),1), height=s_, width=1)

        correlation_matrix = np.corrcoef(S0[0], s_)
        correlation_coefficient = correlation_matrix[0, 1]

        print(f"Pearson's correlation coefficient {i}: {correlation_coefficient}")

        t = np.ravel(spikes_to_times(S0.reshape((1,-1)), sim_time, 0.2, 0.1))
        #print(t[t!=0])

    plt.show()

if __name__ == "__main__":
     debug()