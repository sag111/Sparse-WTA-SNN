import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#def generate_reference_sequence(rate, N, interval: int = 5, resolution=0.1):

#        lam = 1 / rate * (1000 / resolution) # resolution is in ms, rate in Hz

        # Create Poisson distribution and sample inter-spike intervals
        # (incrementing by 1 to avoid zero intervals).
#        rng = np.random.default_rng()

#        intervals = rng.poisson(lam=lam, size=N+1)

#        times = np.cumsum(intervals).astype(np.int32)
#        times[times >= N + 1] = 0.0

#        S0 = np.zeros(N+1, dtype=np.uint8)
#        S0[times] = 1
#        S0 = S0[1:]
        
#        return S0.reshape((1,-1)), lam, times

#def normalize_features(x):
#     if x.sum() == 0:
#          return np.zeros_like(x)
#     return x/x.sum()

#def generate_correlated_sequence(inp_vector, ref_seq, spike_p, N):
     
#     ref_seq = np.ravel(ref_seq)
#     out = np.zeros((len(inp_vector), N))
#     one_idxs = np.nonzero(ref_seq)[0]

#     for i, feat in enumerate(inp_vector):
#          num_spikes_to_keep = np.floor(feat * np.sum(ref_seq)).astype(np.int32)
#          np.random.shuffle(one_idxs)
#          cur_one_idxs = one_idxs[:num_spikes_to_keep]
#          out[i, cur_one_idxs] = ref_seq[cur_one_idxs]
#     return out



def spikes_to_times(inp_spikes, time, tau_s, resolution = 0.1):
     #spike_times = (np.arange(0, time, resolution).reshape((1,-1)) + resolution + tau_s).round(1)
     spike_times = (np.arange(0, time, resolution).reshape((1,-1)) + tau_s).round(1)
     spike_times = np.repeat(spike_times, inp_spikes.shape[0], axis=0) # number of neurons
     spike_times = spike_times[:, :len(inp_spikes[0])] * inp_spikes
     return spike_times

def get_time_dict(spike_times):
     return [{"spike_times":list(st_i[st_i>0])} for st_i in spike_times]
     
class ProbabilisticCorrelationEncoder(BaseEstimator, TransformerMixin):
     
     def __init__(self, rate, resolution, min_spikes = 1):
          self.rate = rate
          self.resolution = resolution
          self.min_spikes = min_spikes

          self.precision = np.floor(-np.log10(resolution)).astype(np.int32)


     def set_global_min_feature(self, X):
          self.global_min = X[X>0].min()

     def normalize_data(self, X):
          return X/X.sum(axis=-1).reshape((-1,1))

     def set_reference_sequence(self, n_spikes):

          time = np.round((1000 * (n_spikes/self.rate)), self.precision) # this many ms

          self.N = int(time/self.resolution)

          spike_prob = (self.rate / 1000) * self.resolution

          self.S0 = (np.random.rand(self.N) <= spike_prob).astype(np.uint8).reshape((1,-1))

          return time

     def count_spikes_per_feature(self,  X: np.ndarray):
          unique_features = np.sort(np.unique(np.ravel(X)))
          #self.feature_spike_map = dict()
          total_spikes = 0
          
          for i in range(0, len(unique_features)):
               total_spikes += int(self.min_spikes * unique_features[i] / self.global_min)

          return total_spikes
     
     def fit(self, X, y=None):
          X = self.normalize_data(X)
          self.set_global_min_feature(X)

          max_spikes = 0
          for x_i in X:
               cur_spikes = self.count_spikes_per_feature(x_i)
               if cur_spikes > max_spikes:
                    max_spikes = cur_spikes

          self.time = self.set_reference_sequence(max_spikes)
          self.ref_times = spikes_to_times(self.S0, self.time, 0, self.resolution)
          self.ref_times = self.ref_times[self.ref_times>0]
          print(f"Simulation time: {self.time} ms")
          self.is_fitted_ = True

          return self

     def transform(self, X, y=None):
          X = self.normalize_data(X)
          X_s = np.zeros((*X.shape, self.N), dtype=np.uint8)
          for i, inp_vector in enumerate(X):
               feature_indices = np.argsort(inp_vector)[::-1]
               consume_S0 = self.S0[0].copy()
               
               cur_prob = 1
               for j in feature_indices:
                    cur_feature = inp_vector[j]

                    cur_sum = np.sum(consume_S0)
                    if cur_sum == 0:
                         break
                    
                    # we want to use at least one spike
                    num_spikes = max(int(cur_sum * cur_feature / cur_prob), 1)
                    
                    nonzero_idxs = np.where(consume_S0 != 0)[0]
                    
                    chosen_idxs = np.random.choice(nonzero_idxs, num_spikes, replace=False)
                    
                    X_s[i][j][chosen_idxs] = 1
                    consume_S0[chosen_idxs] = 0

                    cur_prob -= cur_feature

          return X_s 
               

def debug():
    from fsnn_classifiers.datasets.load_data import load_data

    X_train, X_test, y_train, y_test = load_data(dataset="mnist1000")

    enc = CorrelationEncoder(rate=7000, resolution=0.1, min_spikes=1)
    X_train = enc.normalize_data(X_train)

    enc.fit(X_train)
    print(np.sum(enc.S0), enc.time)
    for x_i in X_train:
         vec = enc.transform(x_i)
         print(vec.sum())
         break 
if __name__ == "__main__":
     debug()
