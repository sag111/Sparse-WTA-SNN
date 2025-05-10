import numpy as np


class CorrelationEncoder(object):
     def __init__(self, epoch_time, full_time, scale, resolution, clip_val, I):
         self.epoch_time = epoch_time
         self.full_time = full_time
         self.scale = scale
         self.resolution = resolution
         self.clip_val = clip_val
         self.I = I

     # def _generate_spike_times(self, inp_vector, vector_number):
     #      inp_vector = np.asarray(inp_vector)
     #      start_time = self.epoch_time + vector_number * self.full_time
     #      # Convert positive input values to spike times.
     #      # Higher values correspond to earlier spikes.
     #      # Times are rounded to the nearest multiple of the resolution.
     #      spike_times = np.where(inp_vector > 0, self.scale / inp_vector, 0)
     #      spike_times = np.round(spike_times / self.resolution) * self.resolution
     #      # Align all spike times so that the first spike occurs at (start_time + resolution),
     #      # to reduce the simulation time.
     #      first_spike = spike_times[spike_times > 0].min()
     #      shifted_spike_times = np.where(
     #           spike_times > 0, 
     #           spike_times - first_spike + self.resolution + start_time,
     #           0
     #      )
     #      return np.expand_dims(shifted_spike_times, 1).round(1)

     def _generate_spike_times(self, inp_vector, vector_number):
          inp = np.asarray(inp_vector)
          start_time = self.epoch_time + vector_number * self.full_time
          raw = np.where(inp > 0, self.scale / inp, 0.0)

          # Convert to ticks
          ticks = np.rint(raw / self.resolution).astype(int)

          # Shift all spikes so that first spike occurs at (start + 1 tick)
          positive = ticks > 0
          if not positive.any():
               return np.zeros((len(inp), 1))
          first_tick = ticks[positive].min()
          shifted_ticks = np.where(positive,
                                   ticks - first_tick + 1 + (start_time / self.resolution),
                                   0).astype(int)

          # Convert to milliseconds
          final_times = (shifted_ticks * self.resolution).reshape(-1, 1)

          return final_times

     def _get_time_dict(self, spike_times):
          return [
               {"spike_times": list(st_i[st_i>0])} 
               for st_i in spike_times
          ]

     def _get_teacher_dict(self, spike_times):
          start_times = np.unique(spike_times[spike_times > 0]).round(1)
          end_times = (start_times + self.resolution).round(1)
          amplitude_times = sorted(list(start_times) + list(end_times))
          amplitude_values = np.tile([self.I, 0.], len(amplitude_times) // 2) # I if i % 2 == 0 else 0
          return [{"amplitude_times": amplitude_times,
                    "amplitude_values": amplitude_values,
                    "allow_offgrid_times": True}]
   
     def __call__(self, X: np.ndarray):
         X_clipped = np.where(X >= self.clip_val, X, 0)
         input_data = []
         teacher_data = []
         for vector_number, x in enumerate(X_clipped):
             spike_times  = self._generate_spike_times(x, vector_number)
             time_dict    = self._get_time_dict(spike_times)
             teacher_dict = self._get_teacher_dict(spike_times)
             input_data.append(time_dict)
             teacher_data.append(teacher_dict)
         return input_data, teacher_data
               

def debug():
     pass


if __name__ == "__main__":
     debug()