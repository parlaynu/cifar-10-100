import time
import torch


def log_writer(log_dir="snapshots"):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except:
        print("Warning: module tensorboard not found ... no logging")
        return None
    
    # verify the logdir
    if log_dir.find("/") == -1:
        now = int(time.time())
        log_dir = f"{log_dir}/{now}"
    
    return SummaryWriter(log_dir=log_dir)


class Logger:
    def __init__(self, pipe, writer, prefix, keys, batch_keys):
        self._pipe = pipe
        self._writer = writer
        self._prefix = prefix
        
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self._keys = keys

        if not isinstance(batch_keys, (list, tuple)):
            batch_keys = [batch_keys]
        self._batch_keys = batch_keys
        
        self._step = -1
        self._batch_step = -1

    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            self._batch_step += 1
            self.log_data(item, self._batch_keys, self._batch_step)
            yield item
        
        self._step += 1
        self.log_data(item, self._keys, self._step)

    def log_data(self, item, keys, step):
        if self._writer is None:
            return
        
        for key in keys:
            value = item[key]
            label = ''.join([word.capitalize() for word in key.split('_')])
            self._writer.add_scalar(f'{self._prefix}/{label}', value, global_step=step)


def logger(pipe, *, writer, prefix, keys, batch_keys=[]):
    pipe = Logger(pipe, writer, prefix, keys, batch_keys)
    return pipe

