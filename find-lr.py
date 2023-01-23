#!/usr/bin/env python3
import warnings
import math
from torch.optim.lr_scheduler import _LRScheduler


class FindLrRamp(_LRScheduler):
    
    def __init__(self, pipe, optimizer, initial_lr, final_lr, num_steps):
        
        self._pipe = pipe
        
        self._initial_lrs = [initial_lr for _ in optimizer.param_groups]
        self._final_lr = final_lr
        self._num_steps = num_steps
        
        self._cur_step = -1
        
        _LRScheduler.__init__(self, optimizer, -1)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        self._cur_step += 1

        lin_scale = self._cur_step / (self._num_steps - 1)
        
        return [lr + lin_scale * (self._final_lr - lr) for lr in self._initial_lrs]

    def __len__(self):
        return len(self._pipe)
    
    def __iter__(self):
        for item in self._pipe:
            item['lr'] = self.get_last_lr()[0]
            yield item
            self.step()


class FindLrExp(_LRScheduler):
    
    def __init__(self, pipe, optimizer, initial_lr, final_lr, num_steps):
        
        self._pipe = pipe
        
        self._initial_lrs = [initial_lr for _ in optimizer.param_groups]
        self._final_lr = final_lr
        self._num_steps = num_steps
        
        self._cur_step = -1
        
        _LRScheduler.__init__(self, optimizer, -1)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        self._cur_step += 1

        lin_scale = self._cur_step / (self._num_steps - 1)
        exp_scale = (math.pow(100, lin_scale) - 1) / 99
        
        return [lr + exp_scale * (self._final_lr - lr) for lr in self._initial_lrs]

    def __len__(self):
        return len(self._pipe)
    
    def __iter__(self):
        for item in self._pipe:
            item['lr'] = self.get_last_lr()[0]
            yield item
            self.step()


def build_pipeline(args):
    import lib as cl
    import lib.models as models

    pipe = head = cl.dataset(args.dsroot, split="train", batch_size=args.batch_size)
    mean, std = head.data_norm()

    model = models.create_model(args.model, head.num_classes(), force_cpu=args.force_cpu, pretrained=True, freeze=args.freeze)
    
    pipe = cl.augmenter(pipe, mode="train", mean=mean, std=std, size=args.image_size)
    pipe = cl.dataloader(pipe, device=model.device, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    
    pipe = cl.trainer(pipe, model, label_smoothing=0.1, 
                        lr=args.initial_lr, weight_decay=0.01, 
                        grad_clip_value=None, 
                        grad_max_norm=None
                        )
    
    
    optimizer = pipe._optimizer
    num_steps = len(pipe) * args.num_epochs
    
    pipe = FindLrExp(pipe, optimizer, args.initial_lr, args.final_lr, num_steps)
    
    pipe = cl.averager(pipe, keymap={"loss": "loss"})

    log_writer = cl.log_writer(args.log_dir)
    pipe = cl.logger(pipe, writer=log_writer, prefix="FindLR", keys=[], batch_keys=["loss", "lr"], )
    
    return pipe, head, model, log_writer


def run(args):
    import lib as cl
    
    # build the pipeline
    pipe, head, *_ = build_pipeline(args)
    
    # run the loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch:02d}")
    
        head.shuffle()
        for idx, item in cl.progress(pipe, header="FindLR", end=""):
            pass
        if loss := item.get('loss', None):
            print(f" loss={loss:0.2e}", end="")
        print("")


def main():
    import os, sys
    import argparse
    from datetime import datetime
    import random
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--force-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=0)
    parser.add_argument('-e', '--num-epochs', help='number of epochs', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=16)
    parser.add_argument('-s', '--image-size', help='resize the images before processing', type=int, default=32)
    parser.add_argument('-z', '--freeze', help='number of layers in the model to freeze', type=int, default=0)
    parser.add_argument("-i", "--initial-lr", help="initial learning rate", type=float, default=0.0000001)
    parser.add_argument("-f", "--final-lr", help="final learning rate", type=float, default=0.01)
    parser.add_argument('model', help='the model type to use', type=str)
    parser.add_argument('dsroot', help='path to the cifar-10 or cifar-100 dataset', type=str)
    
    args = parser.parse_args()
    
    # prepare the log dir
    now = datetime.now()
    args.run_id = now.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join("snapshots", f"{args.run_id}-find-lr")
    
    os.makedirs(args.log_dir)
    
    # save the command line
    cmd_file = os.path.join(args.log_dir, "command.txt")
    with open(cmd_file, "w") as f:
        print(" ".join(sys.argv), file=f)
    
    # initialize the random number generators
    torch.manual_seed(1330)
    random.seed(1330)

    # and run...
    run(args)    


if __name__ == "__main__":
    main()

