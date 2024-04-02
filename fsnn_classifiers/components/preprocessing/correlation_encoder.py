import numpy as np

def generate_reference_sequence(rate, N, interval: int = 5, resolution=0.1):

        S0 = np.zeros(N, dtype=np.uint8)
        for i in range(N):
            if sum(S0)/N > rate*resolution/1000.:
                break
            if i % interval == 0:
                S0[i] = 1
        
        spike_p = (rate*resolution/1000.) * np.exp(-rate*resolution/1000.)
        return S0.reshape((1,-1)), spike_p

def generate_phi_theta(spike_p, inp_vector):
    phi = spike_p * (1 - inp_vector ** 0.5)
    theta = spike_p + (1 - spike_p) * inp_vector ** 0.5

    return phi.reshape((-1,1)), theta.reshape((-1,1))


def generate_correlated_sequence(inp_vector, ref_seq, spike_p, N):
        phi, theta = generate_phi_theta(spike_p, inp_vector)
        s = np.random.rand(len(inp_vector), N)
        der = np.where(((s < ref_seq) & (s < theta)) | ((s >= ref_seq) & (s < phi)), 1, 0)
        
        for i, c in enumerate(inp_vector):
            switch_num = ((1-c) * np.count_nonzero(der[i])).astype(np.int32) # how many spikes to erase
            switch_idx = np.where(der[i] > 0)[0]
            np.random.shuffle(switch_idx)
            der[i][switch_idx[:switch_num]] = 0

        return der.astype(np.uint8)

def spikes_to_times(inp_spikes, time, tau_s, resolution = 0.1):
     spike_times = (np.arange(0, time, resolution).reshape((1,-1)) + resolution + tau_s).round(1)
     spike_times = np.repeat(spike_times, inp_spikes.shape[0], axis=0) # number of neurons
     spike_times = spike_times * inp_spikes
     return spike_times

def get_time_dict(spike_times):
     return [{"spike_times":list(st_i[st_i>0])} for st_i in spike_times]
     

class CorrelationEncoder(object):
     
     def __init__(self, rate, tau_s, time, resolution, interval):
          self.rate = rate
          self.time = time
          self.resolution = resolution
          self.interval = interval
          self.tau_s = tau_s

          self.N = int(time/resolution)

          self.S0, self.spike_p = generate_reference_sequence(
               self.rate,
               self.N,
               self.interval,
          )

          self.ref_times = spikes_to_times(self.S0, self.time, self.tau_s, self.resolution)
          self.ref_times = self.ref_times[self.ref_times>0]

     def __call__(self, X: np.ndarray):
          
          X_s = np.empty((*X.shape, self.N), dtype=np.uint8)
          for i, inp_vector in enumerate(X):
               X_s[i] = generate_correlated_sequence(inp_vector,
                                                     self.S0,
                                                     self.spike_p,
                                                     self.N
                                                     )
          return X_s 
               

def debug():
    import matplotlib.pyplot as plt

    time = 200
    resolution = 0.1
    rate = int(time/resolution)
    N = int(time/resolution)
    S0, spike_p = generate_reference_sequence(rate, N)
    print(S0[0][:100])

    freq = sum(S0[0]) * 10000 / int(time/resolution)
    print(freq, spike_p)

    feature = np.array([0.1, 0.3, 0.7, 0.9])

    S1 = generate_correlated_sequence(feature, S0, spike_p, N)


    fig, ax = plt.subplots(1+len(S1))
    ax[0].bar(x=np.arange(0,int(time/resolution),1), height=S0[0], width=1)
    for i, s_ in enumerate(S1):
        ax[i+1].bar(x=np.arange(0,int(time/resolution),1), height=s_, width=1)

        correlation_matrix = np.corrcoef(S0[0], s_)
        correlation_coefficient = correlation_matrix[0, 1]

        print(f"Pearson's correlation coefficient {i}: {correlation_coefficient}")

        t = np.ravel(spikes_to_times(S0.reshape((1,-1)), time, 0.2, 0.1))
        #print(t[t!=0])

    plt.show()

if __name__ == "__main__":
     debug()
