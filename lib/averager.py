from collections import defaultdict
import torch


class Averager:
    def __init__(self, pipe, keymap):
        self._pipe = pipe
        self._keymap = keymap
        
        self._averages = defaultdict(lambda: [0, 0])
    
    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            for k in self._keymap.keys():
                v = item[k]
                
                if isinstance(v, torch.Tensor):
                    self._averages[k][0] += torch.sum(v).item()
                    self._averages[k][1] += torch.numel(v)
                else:
                    self._averages[k][0] += v
                    self._averages[k][1] += 1
            
            yield item

        for k0, k1 in self._keymap.items():
            item[k1] = self._averages[k0][0] / self._averages[k0][1]
        

def averager(pipe, *, keymap):
    pipe = Averager(pipe, keymap)
    return pipe

