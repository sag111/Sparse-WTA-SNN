import numpy as np
import time


def generate_correlated_sequence(inp_vector, ref_seq, spike_p, N, correct_encoding):
     ref_seq = np.ravel(ref_seq)
     out = np.zeros((len(inp_vector), N))

     # позиции спайков в S0
     one_idxs = np.nonzero(ref_seq)[0]
     used_mask = np.zeros_like(ref_seq, dtype=bool)

     # входы куда потом тыкаем недостающие спайки
     non_zero_input_idxs = []

     np.random.seed(int(time.time()))
     np.random.shuffle(one_idxs)

     total_spikes = np.sum(ref_seq)

     for i, feat in enumerate(inp_vector):
          # свободные позиции спайков из S0
          available_one_idxs = one_idxs[~used_mask[one_idxs]]
          feat_x_ref_spikes = feat * total_spikes
          # сколько спайков будет в рассматриваемом входе без учета дробной части
          num_spikes_to_keep = np.floor(feat_x_ref_spikes).astype(np.int32)
          # запоминаем ненулевые входы (у которых есть дробная часть)
          if feat_x_ref_spikes % 1 > 0:
               non_zero_input_idxs.append(i)
          
          # мешаем свободные позиции спайков
          # np.random.shuffle(available_one_idxs)
          # задаем спайки в рассматриваемом входе
          cur_one_idxs = available_one_idxs[:num_spikes_to_keep]
          out[i, cur_one_idxs] = ref_seq[cur_one_idxs]
          used_mask[cur_one_idxs] = True

     # количество недостающих спайков
     k = int(ref_seq.sum()) - int(out.sum())
     # добавляем недостающие спайки при необходимости
     if correct_encoding and k:
          # выбираем случайно k ненулевых входов
          # np.random.shuffle(non_zero_input_idxs)
          # non_zero_input_idxs = non_zero_input_idxs[:k]
          
          # available_one_idxs = one_idxs[~used_mask[one_idxs]]
        
          # for input_idx, one_idx in zip(non_zero_input_idxs, available_one_idxs):
          #      out[input_idx, one_idx] = 1
          #      used_mask[one_idx] = True

          # разом добавляем недостающие спайки
          available_indices = np.random.choice(list(range(len(inp_vector))), size=k, replace=False)
          chosen_spikes = np.random.choice(one_idxs[~used_mask[one_idxs]], size=k, replace=False)

          out[available_indices, chosen_spikes] = 1

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
     
     def __init__(self, rate, tau_s, sim_time, resolution, interval, correct_encoding=True, use_poisson=False):
          self.rate = rate
          self.sim_time = sim_time
          self.resolution = resolution
          self.interval = interval
          self.tau_s = tau_s
          self.correct_encoding = correct_encoding
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
               X_s[i] = generate_correlated_sequence(inp_vector,
                                                     self.S0,
                                                     self.spike_p,
                                                     self.N,
                                                     self.correct_encoding
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