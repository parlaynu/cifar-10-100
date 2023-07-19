import sys
import torch
import torch.nn.functional as F


class GradStats:
    def __init__(self, pipe, model):
        self._pipe = pipe
        self._model = model
        self._norm_type = 2
    
    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            grads = [p.grad.detach() for p in self._model.parameters() if p.grad is not None]

            g_min = sys.maxsize
            g_max = 0
            for g in grads:
                g_min = min(g_min, torch.min(g))
                g_max = max(g_max, torch.max(g))

            total_norm = torch.norm(torch.stack([torch.norm(g, self._norm_type) for g in grads]), self._norm_type)

            item['gradient_min'] = g_min
            item['gradient_max'] = g_max
            item['total_norm'] = total_norm.item()
            
            yield item


def gradstats(pipe, model):
    pipe = GradStats(pipe, model)
    return pipe

