import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


class IterableDatasetAdapter(IterableDataset):
    def __init__(self, pipe):
        IterableDataset.__init__(self)
        self._pipe = pipe
    
    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            yield item


def dataloader(pipe, *, num_workers, batch_size, device, drop_last):
    
    pin_memory = False if device == torch.device('cpu') else True
    
    pipe = IterableDatasetAdapter(pipe)
    pipe = DataLoader(pipe, num_workers=num_workers, batch_size=batch_size, drop_last=drop_last, pin_memory=pin_memory)
    return pipe

